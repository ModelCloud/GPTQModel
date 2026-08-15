# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch
import torch.nn.functional as F

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.qvq import (
    batched_viterbi_quantize,
    pack_qvq_bank_ids,
    reconstruct_qvq_inner_weight,
)
from gptqmodel.quantization.qvq_codecs import pgc16_codebook
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_mlx import (
    QVQ_MLX_BITS,
    QVQMLXLinear,
    _multirow_vector_width,
    _v4_independent_output_width,
    _v4_row_tile,
    _v4_use_mma,
    qvq_mlx_gemv,
    qvq_mlx_viterbi,
)


def _mlx(tensor: torch.Tensor):
    return mx.array(tensor.numpy())


def _case(
    bits: float,
    m: int,
    *,
    k: int = 32,
    n: int = 32,
    vector_size: int = 2,
    trellis_window: int = 16,
    dual_v2: bool = False,
):
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    generator = torch.Generator().manual_seed(9100 + transition_bits * 10 + m + vector_size)
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (256 // vector_size, tiles),
        generator=generator,
        dtype=torch.int32,
    )
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    x = torch.randn((m, k), generator=generator).half()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=vector_size,
        trellis_window=trellis_window,
        dual_v2=dual_v2,
        in_features=k,
        out_features=n,
    )
    reference = (x.float() @ inner.float()).half()
    return tuple(map(_mlx, (x, trellis))), reference


def _banked_case(bits: float, m: int, *, k: int = 32, n: int = 32):
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    generator = torch.Generator().manual_seed(19100 + transition_bits * 10 + m + k + n)
    tile_count = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (64, tile_count), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    bank_ids = pack_qvq_bank_ids(torch.arange(tile_count, dtype=torch.uint8).remainder_(4))
    x = torch.randn((m, k), generator=generator).half()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=4,
        in_features=k,
        out_features=n,
        bank_ids=bank_ids,
    )
    reference = (x.float() @ inner.float()).half()
    return tuple(map(_mlx, (x, trellis, bank_ids))), reference


@pytest.mark.parametrize(("bits", "vector_size", "banked"), ((2.5, 2, False), (2, 4, True)))
def test_qvq_mlx_linear_runs_full_format_native_forward(bits, vector_size, banked):
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU

    k, n = 32, 48
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    generator = torch.Generator().manual_seed(22100 + transition_bits)
    tile_count = (k // 16) * (n // 16)
    edge_count = 256 // vector_size
    edges = torch.randint(0, 1 << transition_bits, (edge_count, tile_count), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    bank_ids = (
        pack_qvq_bank_ids(torch.arange(tile_count, dtype=torch.uint8).remainder_(4)) if banked else None
    )
    su = torch.randint(0, 2, (k,), generator=generator).mul_(2).sub_(1).float()
    sv = torch.randn(n, generator=generator).mul_(0.1)
    bias = torch.randn(n, generator=generator).mul_(0.1)
    x = torch.randn((2, 3, k), generator=generator).half()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=vector_size,
        in_features=k,
        out_features=n,
        bank_ids=bank_ids,
    )
    transformed = matmul_hadU(x.reshape(-1, k).float() * su)
    expected = matmul_hadU(transformed.half().float() @ inner.float()) * sv + bias
    expected = expected.reshape(2, 3, n).half()
    layer = QVQMLXLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        trellis=_mlx(trellis),
        SU=_mlx(su),
        SV=_mlx(sv),
        bias=_mlx(bias),
        vector_size=vector_size,
        bank_ids=None if bank_ids is None else _mlx(bank_ids),
    )

    actual = layer(_mlx(x))
    mx.eval(actual)
    actual_torch = torch.from_numpy(np.asarray(actual))

    torch.testing.assert_close(actual_torch, expected, rtol=1e-3, atol=1e-3)
    assert actual_torch.dtype == x.dtype


def test_qvq_l18_v4_mlx_linear_runs_full_native_forward():
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU

    bits, k, n = 2, 32, 48
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    generator = torch.Generator().manual_seed(22118)
    tile_count = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (64, tile_count), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    su = torch.randint(0, 2, (k,), generator=generator).mul_(2).sub_(1).float()
    sv = torch.randn(n, generator=generator).mul_(0.1)
    bias = torch.randn(n, generator=generator).mul_(0.1)
    x = torch.randn((2, 3, k), generator=generator).half()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=4,
        trellis_window=18,
        in_features=k,
        out_features=n,
    )
    transformed = matmul_hadU(x.reshape(-1, k).float() * su)
    expected = matmul_hadU(transformed.half().float() @ inner.float()) * sv + bias
    expected = expected.reshape(2, 3, n).half()
    layer = QVQMLXLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        trellis=_mlx(trellis),
        SU=_mlx(su),
        SV=_mlx(sv),
        bias=_mlx(bias),
        vector_size=4,
        trellis_window=18,
    )

    actual = layer(_mlx(x))
    mx.eval(actual)
    actual_torch = torch.from_numpy(np.asarray(actual))

    torch.testing.assert_close(actual_torch, expected, rtol=1e-3, atol=1e-3)
    assert actual_torch.dtype == x.dtype


def test_qvq_mlx_linear_rescales_coherent_input_transform_range():
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU

    width = 2048
    trellis = torch.zeros(((width // 16) ** 2, 16), dtype=torch.int32)
    su = torch.ones(width, dtype=torch.float32)
    sv = torch.full((width,), 1e-4, dtype=torch.float32)
    x = torch.full((1, width), 60000, dtype=torch.float16)
    inner = reconstruct_qvq_inner_weight(trellis, bits=2, in_features=width, out_features=width)
    transformed = matmul_hadU(x.float() * su)
    expected = matmul_hadU(transformed @ inner.float()) * sv
    layer = QVQMLXLinear(
        bits=2,
        in_features=width,
        out_features=width,
        trellis=_mlx(trellis),
        SU=_mlx(su),
        SV=_mlx(sv),
    )

    actual = layer(_mlx(x))
    mx.eval(actual)
    actual_torch = torch.from_numpy(np.asarray(actual)).float()

    assert transformed.abs().max() > torch.finfo(torch.float16).max
    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual_torch).all()
    relative_l2 = (actual_torch - expected).norm() / expected.norm()
    assert relative_l2 < 5e-4


def test_qvq_mlx_loader_conversion_preserves_native_v4_payload():
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch

    bits, k, n = 2.5, 32, 48
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    tile_count = (k // 16) * (n // 16)
    generator = torch.Generator().manual_seed(22250)
    states = torch.randint(0, 1 << transition_bits, (64, tile_count), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(states, transition_bits).T.contiguous()
    bank_ids = pack_qvq_bank_ids(torch.arange(tile_count, dtype=torch.uint8).remainder_(4))
    tensors = {
        "trellis": trellis,
        "SU": torch.randn(k, generator=generator),
        "SV": torch.randn(n, generator=generator),
        "bias": torch.randn(n, generator=generator),
        "bank_ids": bank_ids,
    }
    source = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        name="model.layers.0.self_attn.q_proj",
        tensors=tensors,
        vector_size=4,
        bank_count=4,
    )

    converted = _qvq_mlx_linear_from_torch(source)

    assert isinstance(converted, QVQMLXLinear)
    assert converted.bits == bits
    assert converted.vector_size == 4
    np.testing.assert_array_equal(np.asarray(converted.trellis), trellis.numpy())
    np.testing.assert_array_equal(np.asarray(converted.bank_ids), bank_ids.numpy())
    np.testing.assert_array_equal(np.asarray(converted.SU), tensors["SU"].numpy())
    np.testing.assert_array_equal(np.asarray(converted.SV), tensors["SV"].numpy())


def test_qvq_mlx_loader_conversion_preserves_l18_v4_geometry():
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch

    bits, k, n = 2.5, 32, 48
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    tile_count = (k // 16) * (n // 16)
    generator = torch.Generator().manual_seed(18250)
    edges = torch.randint(0, 1 << transition_bits, (64, tile_count), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    tensors = {
        "trellis": trellis,
        "SU": torch.randn(k, generator=generator),
        "SV": torch.randn(n, generator=generator),
        "bias": torch.randn(n, generator=generator),
    }
    source = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        name="model.layers.0.self_attn.q_proj",
        tensors=tensors,
        vector_size=4,
        trellis_window=18,
    )

    converted = _qvq_mlx_linear_from_torch(source)

    assert isinstance(converted, QVQMLXLinear)
    assert converted.bits == bits
    assert converted.vector_size == 4
    assert converted.trellis_window == 18
    assert converted.bank_ids is None
    np.testing.assert_array_equal(np.asarray(converted.trellis), trellis.numpy())
    np.testing.assert_array_equal(np.asarray(converted.SU), tensors["SU"].numpy())
    np.testing.assert_array_equal(np.asarray(converted.SV), tensors["SV"].numpy())


def test_qvq_l18_v4_mlx_rejects_rates_above_w2_5():
    operands, _ = _case(3, 1, vector_size=4)

    with pytest.raises(ValueError, match="W1 through W2.5"):
        qvq_mlx_gemv(
            *operands,
            3,
            out_features=16,
            vector_size=4,
            trellis_window=18,
        )


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 4, 8))
@pytest.mark.parametrize("m", (1, 4))
def test_qvq_dual_v2_mlx_matches_torch_dense_reference(bits, m):
    operands, reference = _case(bits, m, dual_v2=True)

    actual = qvq_mlx_gemv(
        *operands,
        bits,
        out_features=reference.shape[1],
        dual_v2=True,
    )
    mx.eval(actual)
    actual = torch.from_numpy(np.asarray(actual))

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual.numpy(), reference)


def test_qvq_dual_v2_mlx_linear_runs_full_native_forward():
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch

    bits, k, n = 2, 32, 48
    operands, _ = _case(bits, 6, k=k, n=n, dual_v2=True)
    x, trellis = operands
    generator = torch.Generator().manual_seed(32218)
    su = torch.randint(0, 2, (k,), generator=generator).mul_(2).sub_(1).float()
    sv = torch.randn(n, generator=generator).mul_(0.1)
    bias = torch.randn(n, generator=generator).mul_(0.1)
    torch_layer = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        tensors={
            "trellis": torch.from_numpy(np.asarray(trellis)),
            "SU": su,
            "SV": sv,
            "bias": bias,
        },
        dual_v2=True,
    ).eval()
    layer = _qvq_mlx_linear_from_torch(torch_layer)

    actual = layer(x)
    mx.eval(actual)
    actual = torch.from_numpy(np.asarray(actual))
    expected = torch_layer(torch.from_numpy(np.asarray(x)))

    assert layer.dual_v2 is True
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_qvq_mlx_model_conversion_replaces_matching_linear(monkeypatch):
    import mlx.nn as mlx_nn
    import torch.nn as torch_nn

    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.utils import mlx as mlx_bridge

    class TorchModel(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = QVQLinear(bits=2, in_features=16, out_features=16, register_buffers=True).eval()

    class MLXModel(mlx_nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = mlx_nn.Linear(16, 16, bias=False)

    class ModelArgs:
        @classmethod
        def from_dict(cls, config):
            del config
            return cls()

    monkeypatch.setattr(mlx_bridge, "MLX_AVAILABLE", True)
    monkeypatch.setattr(mlx_bridge, "get_model_path", lambda path: (path, None), raising=False)
    monkeypatch.setattr(
        mlx_bridge,
        "load_config",
        lambda path: {"tie_word_embeddings": False, "model_type": "test"},
        raising=False,
    )
    monkeypatch.setattr(mlx_bridge, "_get_classes", lambda config: (lambda args: MLXModel(), ModelArgs), raising=False)
    monkeypatch.setattr(mlx_bridge, "torch_empty_cache", lambda: None)

    converted = mlx_bridge.convert_qvq_to_mlx_model("unused", TorchModel(), "lm_head")

    assert isinstance(converted.proj, QVQMLXLinear)
    x = mx.zeros((1, 16), dtype=mx.float16)
    actual = converted.proj(x)
    mx.eval(actual)
    assert actual.shape == (1, 16)
    assert actual.dtype == mx.float16
    empty = converted.proj(mx.zeros((0, 16), dtype=mx.float16))
    assert empty.shape == (0, 16)
    assert empty.dtype == mx.float16


@pytest.mark.parametrize("bits", (1, 1.5))
def test_qvq_low_rate_mlx_viterbi_matches_weighted_constrained_cpu_oracle(bits):
    generator = torch.Generator().manual_seed(20261011 + int(bits * 2))
    codebook = pgc16_codebook(dtype=torch.float32)
    sequences = torch.randn((2, 128, 2), generator=generator)
    step_weights = torch.rand((2, 128), generator=generator).add_(0.1)
    transition_bits = qvq_transition_bits(bits)
    overlap = torch.randint(0, 1 << (16 - transition_bits), (2,), generator=generator, dtype=torch.int64)
    expected = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=bits,
        overlap=overlap,
        step_weights=step_weights,
    )

    states, squared_error = qvq_mlx_viterbi(
        _mlx(sequences),
        _mlx(codebook),
        bits,
        mx.array(overlap.numpy(), dtype=mx.uint32),
        _mlx(step_weights),
    )
    mx.eval(states, squared_error)

    assert torch.equal(torch.from_numpy(np.array(states)).long(), expected.states)
    torch.testing.assert_close(
        torch.from_numpy(np.array(squared_error)), expected.squared_error, atol=5e-6, rtol=1e-6
    )


@pytest.mark.parametrize("bits", (1, 1.5))
def test_qvq_mlx_viterbi_minimum_sequence_matches_cpu_oracle(bits):
    sequences = torch.tensor([[[0.25, -0.75]]], dtype=torch.float32)
    codebook = pgc16_codebook(dtype=torch.float32)
    expected = batched_viterbi_quantize(sequences, codebook, bits=bits)

    states, squared_error = qvq_mlx_viterbi(_mlx(sequences), _mlx(codebook), bits)
    mx.eval(states, squared_error)

    assert torch.equal(torch.from_numpy(np.array(states)).long(), expected.states)
    torch.testing.assert_close(
        torch.from_numpy(np.array(squared_error)), expected.squared_error, atol=5e-6, rtol=1e-6
    )


def test_qvq_mlx_viterbi_rejects_invalid_numeric_inputs():
    sequences = mx.zeros((1, 1, 2), dtype=mx.float32)
    codebook = _mlx(pgc16_codebook(dtype=torch.float32))

    bad_sequences = mx.array([[[float("nan"), 0.0]]], dtype=mx.float32)
    with pytest.raises(ValueError, match="sequences and codebook must contain only finite values"):
        qvq_mlx_viterbi(bad_sequences, codebook, 1)

    bad_codebook_torch = pgc16_codebook(dtype=torch.float32)
    bad_codebook_torch[0, 0] = torch.inf
    with pytest.raises(ValueError, match="sequences and codebook must contain only finite values"):
        qvq_mlx_viterbi(sequences, _mlx(bad_codebook_torch), 1)

    for value in (-1.0, float("nan")):
        step_weights = mx.full((1, 1), value, dtype=mx.float32)
        with pytest.raises(ValueError, match="step weights must be finite and nonnegative"):
            qvq_mlx_viterbi(sequences, codebook, 1, step_weights=step_weights)

    overlap_limit = 1 << (16 - qvq_transition_bits(1))
    overlap = mx.array([overlap_limit], dtype=mx.uint32)
    with pytest.raises(ValueError, match=rf"overlap must be in \[0, {overlap_limit - 1}\]"):
        qvq_mlx_viterbi(sequences, codebook, 1, overlap=overlap)


def _assert_dense_accuracy_metrics(actual: np.ndarray, reference: torch.Tensor) -> None:
    actual_f32 = torch.from_numpy(actual).float()
    reference_f32 = reference.float()
    error = actual_f32 - reference_f32
    relative_l2 = error.norm() / reference_f32.norm().clamp_min(torch.finfo(torch.float32).tiny)
    cosine = F.cosine_similarity(actual_f32, reference_f32, dim=-1)
    forward_kld = F.kl_div(
        actual_f32.log_softmax(dim=-1),
        reference_f32.softmax(dim=-1),
        reduction="batchmean",
    )

    assert error.square().mean().item() < 1e-3
    assert relative_l2.item() < 2e-3
    assert torch.all(cosine > 0.99999)
    assert forward_kld.item() < 2e-5
    assert torch.equal(actual_f32.argmax(dim=-1), reference_f32.argmax(dim=-1))
    assert torch.equal(actual_f32.topk(5, dim=-1).indices, reference_f32.topk(5, dim=-1).indices)


@pytest.mark.parametrize("bits", QVQ_MLX_BITS)
@pytest.mark.parametrize("m", (1, 4))
def test_qvq_mlx_all_planar_bits_match_dense_reference(bits, m):
    operands, reference = _case(bits, m)

    actual = np.asarray(qvq_mlx_gemv(*operands, bits, out_features=reference.shape[1]))

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", QVQ_MLX_BITS)
def test_qvq_mlx_fp32_output_all_planar_bits_preserves_fp16_result(bits):
    operands, reference = _case(bits, 1)

    actual = qvq_mlx_gemv(
        *operands, bits, out_features=reference.shape[1], output_fp32=True
    )
    mx.eval(actual)
    actual_numpy = np.asarray(actual)

    assert actual.dtype == mx.float32
    np.testing.assert_allclose(actual_numpy.astype(np.float16), reference.numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("m", (1, 4, 17))
def test_qvq_v4_mlx_matches_dense_reference(bits, m):
    operands, reference = _case(bits, m, vector_size=4)

    actual = np.asarray(
        qvq_mlx_gemv(
            *operands,
            bits,
            out_features=reference.shape[1],
            vector_size=4,
        )
    )

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5))
@pytest.mark.parametrize("m", (1, 4))
def test_qvq_l18_v4_mlx_matches_torch_dense_reference(bits, m):
    operands, reference = _case(bits, m, vector_size=4, trellis_window=18)

    actual = np.asarray(
        qvq_mlx_gemv(
            *operands,
            bits,
            out_features=reference.shape[1],
            vector_size=4,
            trellis_window=18,
        )
    )

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 2.5))
def test_qvq_l18_v4_mlx_fp32_output_matches_torch_dense_reference(bits):
    operands, _ = _case(bits, 3, k=32, n=48, vector_size=4, trellis_window=18)
    x = torch.from_numpy(np.asarray(operands[0]))
    trellis = torch.from_numpy(np.asarray(operands[1]))
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=4,
        trellis_window=18,
        in_features=32,
        out_features=48,
    )
    reference = x.float() @ inner.float()

    actual = qvq_mlx_gemv(
        *operands,
        bits,
        out_features=48,
        vector_size=4,
        trellis_window=18,
        output_fp32=True,
    )
    mx.eval(actual)
    actual = torch.from_numpy(np.asarray(actual))

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("m", (8, 9))
def test_qvq_v4_mlx_mma_matches_dense_reference(bits, m):
    operands, reference = _case(bits, m, k=80, n=64, vector_size=4)

    actual = np.asarray(
        qvq_mlx_gemv(*operands, bits, out_features=reference.shape[1], vector_size=4)
    )

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("m", (1, 4, 17))
def test_qvq_v4_banked_mlx_matches_dense_reference(bits, m):
    (x, trellis, bank_ids), reference = _banked_case(bits, m)

    actual = np.asarray(
        qvq_mlx_gemv(
            x,
            trellis,
            bits,
            out_features=reference.shape[1],
            vector_size=4,
            bank_ids=bank_ids,
        )
    )

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
def test_qvq_v4_banked_mlx_fp32_output_preserves_fp16_result(bits):
    (x, trellis, bank_ids), reference = _banked_case(bits, 1)

    actual = qvq_mlx_gemv(
        x,
        trellis,
        bits,
        out_features=reference.shape[1],
        vector_size=4,
        bank_ids=bank_ids,
        output_fp32=True,
    )
    mx.eval(actual)
    actual_numpy = np.asarray(actual)

    assert actual.dtype == mx.float32
    np.testing.assert_allclose(actual_numpy.astype(np.float16), reference.numpy(), rtol=1e-3, atol=1e-3)


def test_qvq_v4_banked_mlx_fp32_output_preserves_inner_range():
    width = 2048
    out_features = 16
    tile_count = (width // 16) * (out_features // 16)
    trellis = torch.zeros((tile_count, 16), dtype=torch.int32)
    bank_ids = torch.zeros((tile_count + 3) // 4, dtype=torch.uint8)
    x = torch.zeros((1, width), dtype=torch.float16)
    x[0, 0] = 60000
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        vector_size=4,
        in_features=width,
        out_features=out_features,
        bank_ids=bank_ids,
    )

    actual = qvq_mlx_gemv(
        _mlx(x),
        _mlx(trellis),
        2,
        out_features=out_features,
        vector_size=4,
        bank_ids=_mlx(bank_ids),
        output_fp32=True,
    )
    mx.eval(actual)
    actual_torch = torch.from_numpy(np.asarray(actual))
    expected = x.float() @ inner.float()

    assert actual.dtype == mx.float32
    assert torch.isfinite(actual_torch).all()
    assert actual_torch.abs().max() > torch.finfo(torch.float16).max
    torch.testing.assert_close(actual_torch, expected, rtol=2e-5, atol=2e-2)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("m", (8, 9))
def test_qvq_v4_banked_mlx_mma_matches_dense_reference(bits, m):
    (x, trellis, bank_ids), reference = _banked_case(bits, m, k=80, n=64)

    actual = np.asarray(
        qvq_mlx_gemv(
            x,
            trellis,
            bits,
            out_features=reference.shape[1],
            vector_size=4,
            bank_ids=bank_ids,
        )
    )

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 2.5, 4))
def test_qvq_v4_banked_mlx_large_shape_matches_dense_reference(bits):
    (x, trellis, bank_ids), reference = _banked_case(bits, 17, k=144, n=8192)
    actual = np.asarray(
        qvq_mlx_gemv(
            x,
            trellis,
            bits,
            out_features=reference.shape[1],
            vector_size=4,
            bank_ids=bank_ids,
        )
    )
    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 2.5, 4))
def test_qvq_v4_mlx_repeated_launches_are_deterministic(bits):
    operands, reference = _case(bits, 3, vector_size=4)
    outputs = [
        np.asarray(
            qvq_mlx_gemv(
                *operands,
                bits,
                out_features=reference.shape[1],
                vector_size=4,
            )
        )
        for _ in range(10)
    ]

    assert all(np.array_equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.parametrize(
    ("transition_bits", "m", "k", "n", "expected"),
    (
        (8, 4, 2048, 2048, 4), (4, 4, 2048, 2048, 8), (8, 4, 2048, 8192, 4),
        (8, 8, 2048, 2048, 8), (8, 16, 2048, 2048, 8), (8, 16, 2048, 8192, 8),
        (8, 17, 2048, 2048, 4), (8, 24, 2048, 2048, 8), (8, 32, 2048, 2048, 8),
        (8, 32, 2048, 8192, 8), (4, 16, 8192, 2048, 8), (16, 24, 8192, 2048, 8),
        (16, 16, 2048, 8192, 8), (16, 32, 2048, 8192, 8), (4, 32, 8192, 8192, 4),
    ),
)
def test_qvq_v4_mlx_row_tile_dispatch(transition_bits, m, k, n, expected):
    assert _v4_row_tile(transition_bits, m, k, n) == expected


@pytest.mark.parametrize(
    ("transition_bits", "k", "n", "expected"),
    ((4, 2048, 2048, 32), (8, 2048, 2048, 4), (8, 2048, 8192, 32), (8, 8192, 2048, 16)),
)
def test_qvq_v4_mlx_independent_output_width_dispatch(transition_bits, k, n, expected):
    assert _v4_independent_output_width(transition_bits, k, n) == expected


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    ((4, 2048, False), (7, 2048, False), (8, 2048, True), (4, 8192, False), (5, 8192, True)),
)
def test_qvq_v4_mlx_mma_dispatch(m, n, expected):
    assert _v4_use_mma(m, n) is expected


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("k", (64, 80))
def test_qvq_v4_mlx_wide_multirow_matches_dense_reference(bits, k):
    operands, reference = _case(bits, 17, k=k, n=8192, vector_size=4)

    actual = np.asarray(
        qvq_mlx_gemv(
            *operands, bits, out_features=reference.shape[1], vector_size=4,
        )
    )

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
def test_qvq_v4_mlx_wide_multirow_tail_matches_dense_reference(bits):
    operands, reference = _case(bits, 17, k=144, n=8192, vector_size=4)

    actual = np.asarray(
        qvq_mlx_gemv(
            *operands, bits, out_features=reference.shape[1], vector_size=4,
        )
    )

    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", QVQ_MLX_BITS)
def test_qvq_mlx_repeated_launches_are_deterministic(bits):
    operands, reference = _case(bits, 3)
    outputs = [
        np.asarray(qvq_mlx_gemv(*operands, bits, out_features=reference.shape[1]))
        for _ in range(3)
    ]

    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[1], outputs[2])








@pytest.mark.parametrize(
    ("transition_bits", "m", "k", "n", "expected"),
    (
        (4, 4, 2048, 2048, 4),
        (6, 8, 2048, 2048, 4),
        (4, 9, 2048, 2048, 2),
        (6, 16, 2048, 2048, 2),
        (6, 16, 8192, 2048, 4),
        (8, 16, 2048, 2048, 8),
        (14, 16, 8192, 2048, 4),
        (16, 16, 2048, 2048, 8),
        (4, 17, 2048, 2048, 4),
        (4, 32, 2048, 2048, 8),
        (6, 32, 2048, 8192, 4),
        (12, 32, 8192, 2048, 4),
        (16, 32, 8192, 2048, 8),
    ),
)
def test_qvq_mlx_multirow_vector_width_dispatch(transition_bits, m, k, n, expected):
    assert _multirow_vector_width(transition_bits, m, k, n) == expected


def test_qvq_mlx_empty_batch_and_contract_guards():
    x = mx.zeros((0, 16), dtype=mx.float16)
    trellis = mx.zeros((1, 16), dtype=mx.int32)
    assert qvq_mlx_gemv(x, trellis, 2, out_features=16).shape == (0, 16)
    assert qvq_mlx_gemv(x, trellis, 2, out_features=16, output_fp32=True).dtype == mx.float32
    with pytest.raises(TypeError, match="out_features must be an integer"):
        qvq_mlx_gemv(x, trellis, 2, out_features=16.5)
    with pytest.raises(TypeError, match="output_fp32 must be a bool"):
        qvq_mlx_gemv(x, trellis, 2, out_features=16, output_fp32=1)
    with pytest.raises(ValueError, match="pgc16-v1"):
        qvq_mlx_gemv(
            x,
            trellis,
            2,
            out_features=16,
            codebook_version="unsupported",
        )
    with pytest.raises(ValueError, match="W1 through W4"):
        qvq_mlx_gemv(x, trellis, 4.5, out_features=16, vector_size=4)
    with pytest.raises(ValueError, match="vector_size=4"):
        qvq_mlx_gemv(
            x,
            trellis,
            2,
            out_features=16,
            bank_ids=mx.zeros((1,), dtype=mx.uint8),
        )
    with pytest.raises(TypeError, match="packed uint8"):
        qvq_mlx_gemv(
            x,
            trellis,
            2,
            out_features=16,
            vector_size=4,
            bank_ids=mx.zeros((1,), dtype=mx.int32),
        )
    with pytest.raises(ValueError, match="packed shape"):
        qvq_mlx_gemv(
            x,
            trellis,
            2,
            out_features=16,
            vector_size=4,
            bank_ids=mx.zeros((2,), dtype=mx.uint8),
        )

    cases = [
        ((x, trellis, 9), {"out_features": 16}, ValueError, "rate"),
        ((x[:, :, None], trellis, 2), {"out_features": 16}, ValueError, "2D"),
        (
            (mx.zeros((1, 15), dtype=mx.float16), trellis, 2),
            {"out_features": 16},
            ValueError,
            "divisible",
        ),
        (
            (mx.zeros((1, 16), dtype=mx.float16), trellis[:, :8], 2),
            {"out_features": 16},
            ValueError,
            "trellis",
        ),
    ]
    for args, kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            qvq_mlx_gemv(*args, **kwargs)
