# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Dense-reference, edge-case, stream, and free-threaded tests for QVQ CUDA."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import gptqmodel.utils.qvq_cuda as qvq_cuda_utils
from gptqmodel.looper.qvq_output_alignment import _FixedTrellisAlignmentLinear
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear, QVQReferenceLinear
from gptqmodel.quantization.config import ViterbiPruningConfig
from gptqmodel.quantization.qvq import (
    QVQQuantizationTelemetry,
    _batched_v2_banked_viterbi_quantize,
    _canonical_qvq_codebook,
    _canonical_qvq_v2b2_pair_stacks,
    _canonical_qvq_v2b4_bank_stack,
    _canonical_qvq_v2b4_banks,
    _canonical_qvq_v4_banks,
    _yaqa_inner_v2b2_family_batch_cuda,
    batched_viterbi_quantize,
    block_ldlq_inner,
    block_ldlq_inner_banked,
    block_ldlq_inner_banked_candidates,
    block_ldlq_inner_v2b2_p32,
    optimize_qvq_output_channel_scales,
    pack_qvq_bank_ids,
    pack_qvq_binary_bank_ids,
    pack_trellis_states,
    quantize_qvq_linear,
    qvq_proxy_loss,
    reconstruct_qvq_inner_weight,
    unpack_trellis_states,
    yaqa_inner,
    yaqa_inner_v2b2_p32,
    yaqa_proxy_loss,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    canonical_pgc16_levels,
    pgc16_codebook,
    pgc16_codebook_v2_bank,
    pgc16_codebook_v4,
)
from gptqmodel.quantization.qvq_pruning import viterbi_pruning_dispatch_code
from gptqmodel.quantization.qvq_rates import qvq_transition_bits, qvq_words_per_tile
from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b
from gptqmodel.quantization.rotation.hadamard_utils import (
    matmul_hadU,
    matmul_hadU_stable,
)
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_cuda import (
    QVQ_CUDA_BITS,
    _qvq_cuda_viterbi_tail_trusted_op,
    _qvq_cuda_viterbi_trusted,
    _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op,
    _qvq_cuda_viterbi_v2_segment_g_op,
    _qvq_cuda_viterbi_v2_segment_grid_trusted_op,
    _qvq_cuda_viterbi_v2_segment_midpoint_trusted_op,
    _qvq_cuda_viterbi_v2_segment_tail_trusted_op,
    _qvq_cuda_yaqa_feedback_op,
    _qvq_cuda_yaqa_feedback_update_op,
    qvq_cuda_gemv,
    qvq_cuda_hadamard,
    qvq_cuda_hadamard_input_fp16_padded_multiblock,
    qvq_cuda_hadamard_ordered_split16_fp32_to_fp16,
    qvq_cuda_hadamard_pair_fp32_to_fp16,
    qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock,
    qvq_cuda_supported,
    qvq_cuda_swiglu_precondition,
    qvq_cuda_swiglu_precondition_multiblock,
    qvq_cuda_viterbi,
    qvq_cuda_viterbi_banked,
    qvq_cuda_viterbi_v2_segment_banked,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not qvq_cuda_supported(),
        reason="requires NVIDIA CUDA compute capability >= 8.0",
    ),
]


def _case(
    bits: float,
    m: int,
    *,
    k: int = 32,
    n: int = 32,
    dtype: torch.dtype = torch.float16,
    device="cuda:0",
    seed_offset: int = 0,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
):
    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(9100 + transition_bits * 100 + m + k + n + seed_offset)
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (128, tiles), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    x = torch.randn((m, k), generator=generator).to(dtype)
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        in_features=k,
        out_features=n,
        codebook_version=codebook_version,
    )
    reference = (x.float() @ inner.float()).to(dtype)
    return (x.to(device), trellis.to(device)), reference




def _accuracy_metrics(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, float, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    mse = (actual_f - reference_f).square().mean().item()
    forward_kld = torch.nn.functional.kl_div(
        actual_f.log_softmax(dim=-1), reference_f.softmax(dim=-1), reduction="batchmean"
    ).item()
    top1 = (actual_f.argmax(dim=-1) == reference_f.argmax(dim=-1)).float().mean().item()
    actual_top5 = actual_f.topk(5, dim=-1).indices
    reference_top5 = reference_f.topk(5, dim=-1).indices
    top5_overlap = (actual_top5.unsqueeze(-1) == reference_top5.unsqueeze(-2)).any(dim=-1).float().mean().item()
    return mse, forward_kld, top1, top5_overlap


def _assert_accuracy(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_mse: float = 1e-3,
    max_kld: float = 2e-4,
    min_top1: float = 0.96875,
    min_top5: float = 0.96875,
):
    assert torch.isfinite(actual).all()
    mse, kld, top1, top5 = _accuracy_metrics(actual, reference)
    assert mse < max_mse
    assert kld < max_kld
    assert top1 >= min_top1
    assert top5 >= min_top5


@pytest.mark.parametrize("seed", (17, 31, 47))
def test_yaqa_cuda_activation_checkpointing_is_bit_exact_and_deterministic(seed):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, hidden):
            return self.proj(hidden).tanh()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList((Layer(), Layer()))
            self.head = torch.nn.Linear(4, 8, bias=False)

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            hidden = torch.stack(
                (
                    input_ids.float(),
                    input_ids.float() + 1,
                    input_ids.float() - 1,
                    input_ids.float() * 0.5,
                ),
                dim=-1,
            )
            for layer in self.model.layers:
                hidden = layer(hidden)
            return SimpleNamespace(logits=self.head(hidden))

    torch.manual_seed(20260812 + seed)
    baseline_model = Model().eval().cuda()
    checkpointed_model = Model().eval().cuda()
    checkpointed_model.load_state_dict(baseline_model.state_dict())
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[5, 6, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        },
    ]
    baseline_input, baseline_output, _ = capture_yaqa_sketch_b(
        baseline_model,
        batches,
        {"proj": baseline_model.model.layers[0].proj},
        device=torch.device("cuda"),
        seed=seed,
        accumulator_device=torch.device("cpu"),
    )

    for _ in range(10):
        layers = checkpointed_model.model.layers
        actual_input, actual_output, stats = capture_yaqa_sketch_b(
            checkpointed_model,
            batches,
            {"proj": layers[0].proj},
            device=torch.device("cuda"),
            seed=seed,
            checkpoint_modules=tuple(layers),
        )
        torch.cuda.synchronize()

        assert torch.equal(actual_input["proj"], baseline_input["proj"])
        assert torch.equal(actual_output["proj"], baseline_output["proj"])
        assert stats["activation_checkpointing"] is True
        assert stats["checkpointed_modules"] == 2
        assert stats["accumulator_device"] == "cuda"
        assert stats["accumulator_bytes"] == 128
        assert stats["capture_cuda_ms"] > 0
        assert stats["final_host_transfer_seconds"] >= 0
        assert all("forward" not in layer.__dict__ for layer in layers)


@pytest.mark.parametrize("bits", (1, 1.5))
@pytest.mark.parametrize("seed", (17, 31, 47))
def test_qvq_cuda_yaqa_ultralow_rates_are_deterministic_and_match_dense_decoder(bits, seed):
    """Pin YAQA-v3 state selection and unchanged planar reconstruction over repeated CUDA launches."""

    generator = torch.Generator().manual_seed(20261200 + seed + int(bits * 2))
    weight = (torch.randn((32, 16), generator=generator) * 0.1).cuda()
    activations = torch.randn((64, 16), generator=generator).cuda()
    downstream = torch.randn((48, 32), generator=generator).cuda()
    input_hessian = activations.T @ activations / activations.shape[0]
    output_hessian = downstream.T @ downstream / downstream.shape[0]
    held_out = torch.randn((32, 16), generator=generator).cuda()
    reference_trellis = None
    reference_weight = None

    for _ in range(10):
        result = quantize_qvq_linear(
            weight,
            input_hessian,
            output_hessian=output_hessian,
            bits=bits,
            seed=seed,
            trellis_batch_size=16,
            tail_biting_candidates=2,
            rounding="yaqa",
        )
        decoded = reconstruct_qvq_inner_weight(
            result.trellis.cpu(),
            bits=bits,
            in_features=16,
            out_features=32,
        )
        torch.testing.assert_close(decoded, result.inner_weight.cpu())
        layer = QVQReferenceLinear(
            bits=bits,
            in_features=16,
            out_features=32,
            name="proj",
            tensors={"trellis": result.trellis.cpu(), "SU": result.SU.cpu(), "SV": result.SV.cpu()},
            out_dtype=torch.float32,
        )
        actual = layer(held_out.cpu())
        expected = held_out.cpu() @ result.weight.cpu().T

        assert torch.isfinite(result.weight).all()
        assert torch.isfinite(result.kronecker_proxy_loss)
        torch.testing.assert_close(
            result.kronecker_proxy_loss,
            yaqa_proxy_loss(weight, result.weight, input_hessian, output_hessian),
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        if reference_trellis is None:
            reference_trellis = result.trellis.clone()
            reference_weight = result.weight.clone()
        else:
            assert torch.equal(result.trellis, reference_trellis)
            assert torch.equal(result.weight, reference_weight)


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
@pytest.mark.parametrize("m", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_qvq_cuda_all_bits_batches_and_dtypes_match_dense_reference(bits, m, dtype):
    operands, reference = _case(bits, m, dtype=dtype)

    actual = qvq_cuda_gemv(*operands, bits, out_features=reference.shape[1]).cpu()

    assert actual.shape == reference.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    _assert_accuracy(actual, reference)


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
@pytest.mark.parametrize("word", (0, -1, -1431655766, 1431655765))
def test_qvq_cuda_structured_trellis_words_match_pgc16_reference(bits, word):
    """Exercise mixer/codebook extremes that uniformly random tiles can miss."""

    generator = torch.Generator().manual_seed(17000 + qvq_transition_bits(bits))
    x = torch.randn((3, 32), generator=generator, dtype=torch.float16)
    trellis = torch.full((4, qvq_words_per_tile(bits)), word, dtype=torch.int32)
    inner = reconstruct_qvq_inner_weight(trellis, bits=bits, in_features=32, out_features=32)
    reference = (x.float() @ inner.float()).to(torch.float16)

    actual = qvq_cuda_gemv(x.cuda(), trellis.cuda(), bits, out_features=32).cpu()

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    _assert_accuracy(actual, reference)


def test_qvq_cuda_bfloat16_input_keeps_canonical_pgc16_levels_in_float16():
    x = torch.zeros((1, 16), device="cuda", dtype=torch.bfloat16)
    trellis = torch.zeros((1, qvq_words_per_tile(2.5)), device="cuda", dtype=torch.int32)
    captured = {}

    def fake_op(
        input_tensor,
        trellis_tensor,
        levels,
        transition_bits,
        out_features,
        output_fp32,
        bank_ids=None,
        bank_mode=0,
        bank_alt_id=0,
    ):
        del trellis_tensor, bank_ids
        captured["levels"] = levels
        captured["transition_bits"] = transition_bits
        captured["output_fp32"] = output_fp32
        captured["bank_mode"] = bank_mode
        captured["bank_alt_id"] = bank_alt_id
        return torch.empty(
            (input_tensor.shape[0], out_features),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    with patch("gptqmodel.utils.qvq_cuda._qvq_cuda_op", return_value=fake_op):
        qvq_cuda_gemv(x, trellis, 2.5, out_features=16)

    levels = captured["levels"]
    assert levels.dtype == torch.float16
    assert torch.equal(levels.cpu().view(torch.int16), canonical_pgc16_levels().view(torch.int16))
    assert captured["transition_bits"] == 5
    assert captured["output_fp32"] is False
    assert captured["bank_mode"] == 0
    assert captured["bank_alt_id"] == 0


def test_qvq_cuda_fp32_inner_output_preserves_accumulator_range_and_accuracy():
    operands, _ = _case(2, 4, k=256, n=80)
    x, trellis = operands
    inner = reconstruct_qvq_inner_weight(trellis.cpu(), bits=2, in_features=256, out_features=80)
    reference = x.float().cpu() @ inner.float()

    actual = qvq_cuda_gemv(x, trellis, 2, out_features=80, output_fp32=True).cpu()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)


def test_qvq_cuda_range_safe_hadamard_fuses_prescale_before_narrowing():
    x = torch.zeros((1, 32), dtype=torch.float16, device="cuda")
    x[0, 0] = 40000
    pre_scale = torch.full((32,), 2.0, dtype=torch.float16, device="cuda")
    reference = matmul_hadU(x.float() * pre_scale.float()).to(torch.float16)

    actual = qvq_cuda_hadamard(x, pre_scale=pre_scale, scale_mode=2)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-2)


def test_qvq_cuda_range_safe_hadamard_is_bitwise_identical_when_prescale_fits_fp16():
    generator = torch.Generator().manual_seed(20260815)
    x = torch.randn((3, 2048), generator=generator, dtype=torch.float16).cuda()
    pre_scale = torch.randn((2048,), generator=generator, dtype=torch.float16).cuda()

    original = qvq_cuda_hadamard(x, pre_scale=pre_scale, scale_mode=0)
    range_safe = qvq_cuda_hadamard(x, pre_scale=pre_scale, scale_mode=2)

    assert torch.isfinite(original).all()
    assert torch.equal(range_safe, original)


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
def test_qvq_cuda_hadamard_direct_padding_is_exact_and_graph_stable(m):
    n = 2048
    generator = torch.Generator(device="cuda").manual_seed(20261800 + m)
    x = torch.randn(
        (m, n), generator=generator, device="cuda", dtype=torch.float16
    )
    pre_scale = torch.randn(
        (n,), generator=generator, device="cuda", dtype=torch.float16
    )
    expected = qvq_cuda_hadamard(x, pre_scale=pre_scale, scale_mode=2)
    actual = qvq_cuda_hadamard(
        x,
        pre_scale=pre_scale,
        scale_mode=2,
        pad_to_16=True,
    )
    assert actual.shape == (16, n)
    assert torch.equal(actual[:m].view(torch.int16), expected.view(torch.int16))
    assert torch.count_nonzero(actual[m:]).item() == 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_hadamard(
            x,
            pre_scale=pre_scale,
            scale_mode=2,
            pad_to_16=True,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), actual.view(torch.int16))


def test_qvq_cuda_hadamard_direct_padding_guards():
    x = torch.ones((1, 2048), device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="pad_to_16"):
        qvq_cuda_hadamard(x, pad_to_16=1)
    with pytest.raises(ValueError, match="nonempty 2D"):
        qvq_cuda_hadamard(x.expand(17, -1).contiguous(), pad_to_16=True)
    with pytest.raises(ValueError, match="nonempty 2D"):
        qvq_cuda_hadamard(x.reshape(1, 1, 2048), pad_to_16=True)


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
def test_qvq_cuda_hadamard_multiblock_input_is_exact_and_graph_stable(m):
    properties = torch.cuda.get_device_properties(0)
    if properties.name != "NVIDIA H100" or (
        properties.major,
        properties.minor,
    ) != (9, 0):
        pytest.skip("requires the physical H100 Phase-26 path")

    generator = torch.Generator(device="cuda").manual_seed(20262600 + m)
    x = torch.randn(
        (m, 2048), generator=generator, device="cuda", dtype=torch.float16
    )
    pre_scale = torch.randn(
        (2048,), generator=generator, device="cuda", dtype=torch.float16
    )
    expected = qvq_cuda_hadamard(
        x, pre_scale=pre_scale, scale_mode=2, pad_to_16=True
    )
    actual = qvq_cuda_hadamard_input_fp16_padded_multiblock(
        x, pre_scale=pre_scale
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_hadamard_input_fp16_padded_multiblock(
            x, pre_scale=pre_scale
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))


def test_qvq_cuda_hadamard_multiblock_input_rescues_prescale_overflow_on_stream():
    properties = torch.cuda.get_device_properties(0)
    if properties.name != "NVIDIA H100" or (
        properties.major,
        properties.minor,
    ) != (9, 0):
        pytest.skip("requires the physical H100 Phase-26 path")

    x = torch.zeros((1, 2048), device="cuda", dtype=torch.float16)
    pre_scale = torch.ones((2048,), device="cuda", dtype=torch.float16)
    x[0, :2] = 60000
    pre_scale[:2] = 2
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        expected = qvq_cuda_hadamard(
            x, pre_scale=pre_scale, scale_mode=2, pad_to_16=True
        )
        actual = qvq_cuda_hadamard_input_fp16_padded_multiblock(
            x, pre_scale=pre_scale
        )
    stream.synchronize()
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_qvq_cuda_float32_hadamard_preserves_postscale_and_bias_precision():
    generator = torch.Generator().manual_seed(20260813)
    x = torch.randn((3, 32), generator=generator, dtype=torch.float32).cuda()
    post_scale = torch.randn((32,), generator=generator, dtype=torch.float32).cuda()
    bias = torch.randn((32,), generator=generator, dtype=torch.float32).cuda()
    reference = matmul_hadU(x) * post_scale + bias

    actual = qvq_cuda_hadamard(x, post_scale=post_scale, bias=bias, scale_mode=1)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)


def test_qvq_cuda_fp16_emulation_rescues_late_butterfly_and_sv_overflow():
    x = torch.zeros((1, 32), dtype=torch.float32, device="cuda")
    x[0, :2] = 40000
    post_scale = torch.full((32,), 2.0, dtype=torch.float16, device="cuda").float()
    bias = torch.full((32,), -60000.0, dtype=torch.float16, device="cuda").float()
    historical = matmul_hadU(x.to(torch.float16)) * post_scale.to(torch.float16) + bias.to(torch.float16)
    reference = matmul_hadU(x) * post_scale + bias

    actual = qvq_cuda_hadamard(x, post_scale=post_scale, bias=bias, scale_mode=4)
    direct_fp16 = qvq_cuda_hadamard(
        x,
        post_scale=post_scale,
        bias=bias,
        scale_mode=4,
        output_fp16=True,
    )

    assert not torch.isfinite(historical).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=16.0)
    assert torch.equal(direct_fp16.view(torch.int16), actual.half().view(torch.int16))


@pytest.mark.parametrize("scale_mode", (3, 4))
@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
def test_qvq_cuda_hadamard_fp16_final_store_is_bit_exact_and_graph_stable(
    scale_mode, m
):
    n = 2048
    generator = torch.Generator(device="cuda").manual_seed(
        20261900 + 10 * scale_mode + m
    )
    x = torch.randn(
        (m, n), generator=generator, device="cuda", dtype=torch.float32
    )
    post_scale = torch.randn(
        (n,), generator=generator, device="cuda", dtype=torch.float32
    )
    bias = torch.randn(
        (n,), generator=generator, device="cuda", dtype=torch.float32
    )
    expected = qvq_cuda_hadamard(
        x,
        post_scale=post_scale,
        bias=bias,
        scale_mode=scale_mode,
    ).to(torch.float16)
    actual = qvq_cuda_hadamard(
        x,
        post_scale=post_scale,
        bias=bias,
        scale_mode=scale_mode,
        output_fp16=True,
    )
    assert actual.dtype == torch.float16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_hadamard(
            x,
            post_scale=post_scale,
            bias=bias,
            scale_mode=scale_mode,
            output_fp16=True,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))


def test_qvq_cuda_hadamard_fp16_final_store_guards():
    fp16 = torch.ones((1, 32), device="cuda", dtype=torch.float16)
    fp32 = fp16.float()
    with pytest.raises(TypeError, match="output_fp16"):
        qvq_cuda_hadamard(fp32, scale_mode=4, output_fp16=1)
    with pytest.raises(TypeError, match="require float32"):
        qvq_cuda_hadamard(fp16, scale_mode=4, output_fp16=True)
    with pytest.raises(ValueError, match="scale mode 3/4"):
        qvq_cuda_hadamard(fp32, scale_mode=1, output_fp16=True)
    with pytest.raises(ValueError, match="no padding"):
        qvq_cuda_hadamard(
            fp32,
            scale_mode=4,
            pad_to_16=True,
            output_fp16=True,
        )


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("scale_mode", (3, 4))
@pytest.mark.parametrize("bias_enabled", (False, True))
def test_qvq_cuda_ordered_split16_recovery_is_bit_exact_and_graph_stable(
    m, scale_mode, bias_enabled
):
    properties = torch.cuda.get_device_properties(0)
    if properties.name != "NVIDIA H100" or (
        properties.major,
        properties.minor,
    ) != (9, 0):
        pytest.skip("requires the physical H100 Phase-25 path")

    generator = torch.Generator(device="cuda").manual_seed(20262300 + m)
    partials = (
        torch.randn(
            (16, 16, 2048),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    post_scale = torch.randn(
        (2048,), generator=generator, device="cuda", dtype=torch.float32
    )
    bias = (
        torch.randn(
            (2048,), generator=generator, device="cuda", dtype=torch.float32
        )
        if bias_enabled
        else None
    )
    reduced = torch.zeros_like(partials[0])
    for split in range(16):
        reduced = reduced + partials[split]
    expected = qvq_cuda_hadamard(
        reduced[:m],
        post_scale=post_scale,
        bias=bias,
        scale_mode=scale_mode,
        output_fp16=True,
    )
    actual = qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
        partials,
        post_scale=post_scale,
        bias=bias,
        scale_mode=scale_mode,
        logical_rows=m,
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    actual_multiblock = qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
        partials,
        post_scale=post_scale,
        bias=bias,
        scale_mode=scale_mode,
        logical_rows=m,
        multiblock=True,
    )
    assert torch.equal(
        actual_multiblock.view(torch.int16), expected.view(torch.int16)
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
            partials,
            post_scale=post_scale,
            bias=bias,
            scale_mode=scale_mode,
            logical_rows=m,
            multiblock=True,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))

    with pytest.raises(TypeError, match="multiblock"):
        qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
            partials,
            post_scale=post_scale,
            bias=bias,
            scale_mode=scale_mode,
            logical_rows=m,
            multiblock=1,
        )


def test_qvq_cuda_multiblock_ordered_split16_preserves_late_overflow_on_stream():
    properties = torch.cuda.get_device_properties(0)
    if properties.name != "NVIDIA H100" or (
        properties.major,
        properties.minor,
    ) != (9, 0):
        pytest.skip("requires the physical H100 Phase-25 path")

    partials = torch.zeros((16, 16, 2048), device="cuda", dtype=torch.float32)
    partials[0, 0, 0] = 60000.0
    partials[0, 0, 1] = 60000.0
    post_scale = torch.full((2048,), 0.001, device="cuda", dtype=torch.float32)
    reduced = torch.zeros_like(partials[0])
    for split in range(16):
        reduced = reduced + partials[split]
    expected = qvq_cuda_hadamard(
        reduced[:1],
        post_scale=post_scale,
        scale_mode=4,
        output_fp16=True,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
            partials,
            post_scale=post_scale,
            scale_mode=4,
            logical_rows=1,
            multiblock=True,
        )
    stream.synchronize()
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("seed", (20260901, 20260902, 20260903))
def test_qvq_cuda_paired_output_recovery_is_bit_exact_and_repeatable(m, seed):
    n = 8192
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input0 = torch.randn((m, n), generator=generator, device="cuda") * 20
    input1 = torch.randn((m, n), generator=generator, device="cuda") * 20
    scale0 = torch.randn((n,), generator=generator, device="cuda")
    scale1 = torch.randn((n,), generator=generator, device="cuda")
    bias0 = torch.randn((n,), generator=generator, device="cuda")
    bias1 = torch.randn((n,), generator=generator, device="cuda")
    expected = (
        qvq_cuda_hadamard(
            input0, post_scale=scale0, bias=bias0, scale_mode=3
        ).half(),
        qvq_cuda_hadamard(
            input1, post_scale=scale1, bias=bias1, scale_mode=3
        ).half(),
    )

    for _ in range(10):
        actual = qvq_cuda_hadamard_pair_fp32_to_fp16(
            input0,
            input1,
            post_scale0=scale0,
            post_scale1=scale1,
            bias0=bias0,
            bias1=bias1,
        )
        assert actual[0].dtype == torch.float16
        assert actual[1].dtype == torch.float16
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("seed", (20260921, 20260922, 20260923))
@pytest.mark.parametrize("scale_mode", (3, 4))
@pytest.mark.parametrize("warp_low", (False, True))
def test_qvq_cuda_multiblock_paired_recovery_is_bit_exact_and_repeatable(
    m, seed, scale_mode, warp_low
):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("experimental multiblock recovery requires Hopper")
    n = 8192
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input0 = torch.randn((m, n), generator=generator, device="cuda") * 20
    input1 = torch.randn((m, n), generator=generator, device="cuda") * 20
    scale0 = torch.randn((n,), generator=generator, device="cuda")
    scale1 = torch.randn((n,), generator=generator, device="cuda")
    bias0 = torch.randn((n,), generator=generator, device="cuda")
    expected = qvq_cuda_hadamard_pair_fp32_to_fp16(
        input0,
        input1,
        post_scale0=scale0,
        post_scale1=scale1,
        bias0=bias0,
        scale_mode=scale_mode,
    )

    for _ in range(10):
        actual = qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock(
            input0,
            input1,
            post_scale0=scale0,
            post_scale1=scale1,
            bias0=bias0,
            scale_mode=scale_mode,
            warp_low=warp_low,
        )
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("warp_low", (False, True))
def test_qvq_cuda_multiblock_paired_recovery_graph_and_contract_guards(warp_low):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("experimental multiblock recovery requires Hopper")
    n = 8192
    input0 = torch.randn((1, n), device="cuda")
    input1 = torch.randn((1, n), device="cuda")
    scale = torch.ones((n,), device="cuda")
    expected = qvq_cuda_hadamard_pair_fp32_to_fp16(
        input0,
        input1,
        post_scale0=scale,
        post_scale1=scale,
    )
    qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock(
        input0,
        input1,
        post_scale0=scale,
        post_scale1=scale,
        warp_low=warp_low,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock(
            input0,
            input1,
            post_scale0=scale,
            post_scale1=scale,
            warp_low=warp_low,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], expected[0])
    assert torch.equal(captured[1], expected[1])

    with pytest.raises(ValueError, match="last dimension 8192"):
        qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock(
            input0[:, :4096],
            input1[:, :4096],
            post_scale0=scale[:4096],
            post_scale1=scale[:4096],
            warp_low=warp_low,
        )
    with pytest.raises(TypeError, match="warp_low"):
        qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock(
            input0,
            input1,
            post_scale0=scale,
            post_scale1=scale,
            warp_low=1,
        )


@pytest.mark.parametrize("scale_mode", (3, 4))
@pytest.mark.parametrize("warp_low", (False, True))
def test_qvq_cuda_multiblock_paired_recovery_preserves_overflow_rounding_bits(
    scale_mode, warp_low
):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("experimental multiblock recovery requires Hopper")
    n = 8192
    input0 = torch.full((1, n), 60000.0, device="cuda")
    input1 = torch.empty((1, n), device="cuda")
    input1[:, 0::2] = 60000.0
    input1[:, 1::2] = -60000.0
    scale0 = torch.full((n,), 1.0e-4, device="cuda")
    scale1 = torch.full((n,), -1.0e-4, device="cuda")
    bias0 = torch.linspace(-0.125, 0.125, n, device="cuda")
    expected = qvq_cuda_hadamard_pair_fp32_to_fp16(
        input0,
        input1,
        post_scale0=scale0,
        post_scale1=scale1,
        bias0=bias0,
        scale_mode=scale_mode,
    )

    for _ in range(10):
        actual = qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock(
            input0,
            input1,
            post_scale0=scale0,
            post_scale1=scale1,
            bias0=bias0,
            scale_mode=scale_mode,
            warp_low=warp_low,
        )
        assert torch.equal(actual[0].view(torch.int16), expected[0].view(torch.int16))
        assert torch.equal(actual[1].view(torch.int16), expected[1].view(torch.int16))


def test_qvq_cuda_paired_output_recovery_handles_overflow_optional_bias_and_stream():
    n = 32
    input0 = torch.zeros((1, n), device="cuda")
    input1 = torch.zeros((1, n), device="cuda")
    input0[0, :2] = 40000
    input1[0, 0] = -40000
    scale0 = torch.full((n,), 2.0, device="cuda")
    scale1 = torch.full((n,), 0.25, device="cuda")
    bias0 = torch.full((n,), -60000.0, device="cuda")
    expected = (
        qvq_cuda_hadamard(
            input0, post_scale=scale0, bias=bias0, scale_mode=4
        ).half(),
        qvq_cuda_hadamard(input1, post_scale=scale1, scale_mode=4).half(),
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_cuda_hadamard_pair_fp32_to_fp16(
            input0,
            input1,
            post_scale0=scale0,
            post_scale1=scale1,
            bias0=bias0,
            scale_mode=4,
        )
    stream.synchronize()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_qvq_cuda_paired_output_recovery_empty_graph_and_contract_guards():
    n = 32
    empty = torch.empty((0, n), device="cuda")
    scale = torch.ones(n, device="cuda")
    outputs = qvq_cuda_hadamard_pair_fp32_to_fp16(
        empty, empty, post_scale0=scale, post_scale1=scale
    )
    assert tuple(output.shape for output in outputs) == ((0, n), (0, n))
    assert all(output.dtype == torch.float16 for output in outputs)

    input0 = torch.randn((1, n), device="cuda")
    input1 = torch.randn((1, n), device="cuda")
    qvq_cuda_hadamard_pair_fp32_to_fp16(
        input0, input1, post_scale0=scale, post_scale1=scale
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_hadamard_pair_fp32_to_fp16(
            input0, input1, post_scale0=scale, post_scale1=scale
        )
    graph.replay()
    torch.cuda.synchronize()
    reference = (
        qvq_cuda_hadamard(input0, post_scale=scale, scale_mode=3).half(),
        qvq_cuda_hadamard(input1, post_scale=scale, scale_mode=3).half(),
    )
    assert torch.equal(captured[0], reference[0])
    assert torch.equal(captured[1], reference[1])

    with pytest.raises(TypeError, match="float32"):
        qvq_cuda_hadamard_pair_fp32_to_fp16(
            input0.half(), input1.half(), post_scale0=scale, post_scale1=scale
        )
    with pytest.raises(ValueError, match="identical shapes"):
        qvq_cuda_hadamard_pair_fp32_to_fp16(
            input0, input1[:, :-1], post_scale0=scale, post_scale1=scale
        )
    with pytest.raises(ValueError, match="scale_mode"):
        qvq_cuda_hadamard_pair_fp32_to_fp16(
            input0,
            input1,
            post_scale0=scale,
            post_scale1=scale,
            scale_mode=2,
        )


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("seed", (20260911, 20260912, 20260913))
def test_qvq_cuda_swiglu_precondition_is_bit_exact_and_repeatable(m, seed):
    n = 8192
    generator = torch.Generator(device="cuda").manual_seed(seed)
    gate = torch.randn((m, n), generator=generator, device="cuda", dtype=torch.float16)
    up = torch.randn((m, n), generator=generator, device="cuda", dtype=torch.float16)
    pre_scale = torch.randn((n,), generator=generator, device="cuda", dtype=torch.float16)
    activated_gate = torch.nn.functional.silu(gate)
    reference = qvq_cuda_hadamard(
        activated_gate * up,
        pre_scale=pre_scale,
        scale_mode=2,
    )

    for _ in range(10):
        actual = qvq_cuda_swiglu_precondition(activated_gate, up, pre_scale)
        assert actual.dtype == torch.float16
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("seed", (20260931, 20260932, 20260933))
@pytest.mark.parametrize("half2_high", (False, True))
@pytest.mark.parametrize("fuse_silu", (False, True))
@pytest.mark.parametrize("half2_low", (False, True))
def test_qvq_cuda_multiblock_swiglu_precondition_is_bit_exact_and_repeatable(
    m, seed, half2_high, fuse_silu, half2_low
):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("multiblock SwiGLU precondition requires Hopper")
    n = 8192
    generator = torch.Generator(device="cuda").manual_seed(seed)
    gate = torch.randn(
        (m, n), generator=generator, device="cuda", dtype=torch.float16
    )
    up = torch.randn(
        (m, n), generator=generator, device="cuda", dtype=torch.float16
    )
    pre_scale = torch.randn(
        (n,), generator=generator, device="cuda", dtype=torch.float16
    )
    activated_gate = torch.nn.functional.silu(gate)
    expected = qvq_cuda_swiglu_precondition(activated_gate, up, pre_scale)
    gate_input = gate if fuse_silu else activated_gate

    for _ in range(10):
        actual = qvq_cuda_swiglu_precondition_multiblock(
            gate_input,
            up,
            pre_scale,
            half2_high=half2_high,
            fuse_silu=fuse_silu,
            half2_low=half2_low,
        )
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("half2_high", (False, True))
@pytest.mark.parametrize("fuse_silu", (False, True))
@pytest.mark.parametrize("half2_low", (False, True))
def test_qvq_cuda_multiblock_swiglu_precondition_graph_and_guards(
    half2_high, fuse_silu, half2_low
):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("multiblock SwiGLU precondition requires Hopper")
    n = 8192
    gate = torch.randn((1, n), device="cuda", dtype=torch.float16)
    up = torch.randn((1, n), device="cuda", dtype=torch.float16)
    pre_scale = torch.ones((n,), device="cuda", dtype=torch.float16)
    activated_gate = torch.nn.functional.silu(gate) if fuse_silu else gate
    expected = qvq_cuda_swiglu_precondition(activated_gate, up, pre_scale)
    qvq_cuda_swiglu_precondition_multiblock(
        gate,
        up,
        pre_scale,
        half2_high=half2_high,
        fuse_silu=fuse_silu,
        half2_low=half2_low,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_swiglu_precondition_multiblock(
            gate,
            up,
            pre_scale,
            half2_high=half2_high,
            fuse_silu=fuse_silu,
            half2_low=half2_low,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, expected)

    with pytest.raises(ValueError, match="N=8192"):
        qvq_cuda_swiglu_precondition_multiblock(
            gate[:, :4096],
            up[:, :4096],
            pre_scale[:4096],
            half2_high=half2_high,
            fuse_silu=fuse_silu,
            half2_low=half2_low,
        )
    with pytest.raises(TypeError, match="fuse_silu"):
        qvq_cuda_swiglu_precondition_multiblock(
            gate,
            up,
            pre_scale,
            half2_high=half2_high,
            fuse_silu=1,
            half2_low=half2_low,
        )
    with pytest.raises(TypeError, match="half2_low"):
        qvq_cuda_swiglu_precondition_multiblock(
            gate,
            up,
            pre_scale,
            half2_high=half2_high,
            fuse_silu=fuse_silu,
            half2_low=1,
        )
    with pytest.raises(TypeError, match="pad_to_16"):
        qvq_cuda_swiglu_precondition_multiblock(
            gate,
            up,
            pre_scale,
            half2_high=half2_high,
            fuse_silu=fuse_silu,
            half2_low=half2_low,
            pad_to_16=1,
        )
    with pytest.raises(ValueError, match="at most 16 rows"):
        qvq_cuda_swiglu_precondition_multiblock(
            gate.expand(17, -1).contiguous(),
            up.expand(17, -1).contiguous(),
            pre_scale,
            half2_high=half2_high,
            fuse_silu=fuse_silu,
            half2_low=half2_low,
            pad_to_16=True,
        )


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
def test_qvq_cuda_multiblock_swiglu_precondition_direct_padding_is_exact_and_graph_stable(m):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("multiblock SwiGLU precondition requires Hopper")
    n = 8192
    generator = torch.Generator(device="cuda").manual_seed(20261700 + m)
    gate = torch.randn((m, n), generator=generator, device="cuda", dtype=torch.float16)
    up = torch.randn((m, n), generator=generator, device="cuda", dtype=torch.float16)
    pre_scale = torch.randn((n,), generator=generator, device="cuda", dtype=torch.float16)
    expected = qvq_cuda_swiglu_precondition_multiblock(
        gate,
        up,
        pre_scale,
        half2_high=True,
        fuse_silu=True,
        half2_low=True,
    )
    actual = qvq_cuda_swiglu_precondition_multiblock(
        gate,
        up,
        pre_scale,
        half2_high=True,
        fuse_silu=True,
        half2_low=True,
        pad_to_16=True,
    )
    assert actual.shape == (16, n)
    assert torch.equal(actual[:m], expected)
    assert torch.count_nonzero(actual[m:]).item() == 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_swiglu_precondition_multiblock(
            gate,
            up,
            pre_scale,
            half2_high=True,
            fuse_silu=True,
            half2_low=True,
            pad_to_16=True,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[:m], expected)
    assert torch.count_nonzero(captured[m:]).item() == 0


def test_qvq_cuda_multiblock_fused_silu_covers_every_finite_fp16_value():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("multiblock SwiGLU precondition requires Hopper")
    n = 8192
    all_bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.float16)
    finite = all_bits[torch.isfinite(all_bits)]
    gate = torch.zeros((8, n), device="cuda", dtype=torch.float16)
    gate.view(-1)[: finite.numel()].copy_(finite.to(device="cuda"))
    up = torch.ones_like(gate)
    pre_scale = torch.full((n,), 0.125, device="cuda", dtype=torch.float16)
    expected = qvq_cuda_swiglu_precondition_multiblock(
        torch.nn.functional.silu(gate),
        up,
        pre_scale,
        half2_high=True,
    )
    actual = qvq_cuda_swiglu_precondition_multiblock(
        gate,
        up,
        pre_scale,
        half2_high=True,
        fuse_silu=True,
        half2_low=True,
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_qvq_cuda_swiglu_precondition_graph_stream_overflow_and_guards():
    n = 32
    gate = torch.zeros((1, n), device="cuda", dtype=torch.float16)
    up = torch.ones((1, n), device="cuda", dtype=torch.float16)
    gate[0, :2] = 40000
    up[0, :2] = 2
    pre_scale = torch.full((n,), 0.25, device="cuda", dtype=torch.float16)
    reference = qvq_cuda_hadamard(
        gate * up,
        pre_scale=pre_scale,
        scale_mode=2,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_cuda_swiglu_precondition(gate, up, pre_scale)
    stream.synchronize()
    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_cuda_swiglu_precondition(gate, up, pre_scale)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), reference.view(torch.int16))

    empty = qvq_cuda_swiglu_precondition(gate[:0], up[:0], pre_scale)
    assert empty.shape == (0, n)
    with pytest.raises(TypeError, match="float16"):
        qvq_cuda_swiglu_precondition(gate.float(), up.float(), pre_scale.float())
    with pytest.raises(ValueError, match="identical shapes"):
        qvq_cuda_swiglu_precondition(gate, up[:, :-1], pre_scale)


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
def test_qvq_cuda_accumulation_heavy_shape_matches_dense_reference(bits):
    operands, reference = _case(bits, 33, k=256, n=80, dtype=torch.bfloat16)

    actual = qvq_cuda_gemv(*operands, bits, out_features=80).cpu()

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=5e-2)
    _assert_accuracy(actual, reference, max_kld=2e-3, min_top1=0.96, min_top5=0.96)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("m", (1, 32))
def test_qvq_cuda_no_split_threshold_matches_dense_reference(dtype, m):
    """Pin scalar/WMMA dispatch at the 384-base-block no-split threshold."""

    operands, reference = _case(4, m, k=16, n=6144, dtype=dtype)

    actual = qvq_cuda_gemv(*operands, 4, out_features=6144).cpu()

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    _assert_accuracy(actual, reference)


def test_qvq_cuda_split_count_cap_matches_dense_reference():
    """Exercise the 64-way split cap used by the narrowest real projections."""

    operands, reference = _case(4, 1, k=4096, n=16)

    actual = qvq_cuda_gemv(*operands, 4, out_features=16).cpu()

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    _assert_accuracy(actual, reference)


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
@pytest.mark.parametrize("seed_offset", (0, 1, 2))
def test_qvq_cuda_repeated_launches_are_bitwise_deterministic(bits, seed_offset):
    operands, reference = _case(bits, 8, k=64, n=48, seed_offset=seed_offset)
    outputs = [qvq_cuda_gemv(*operands, bits, out_features=reference.shape[1]) for _ in range(10)]
    torch.cuda.synchronize()

    for output in outputs:
        _assert_accuracy(output.cpu(), reference)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_qvq_cuda_uses_current_non_default_stream():
    operands, reference = _case(4, 8, k=64, n=48)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_cuda_gemv(*operands, 4, out_features=48)
        completion = torch.cuda.Event()
        completion.record(stream)
    completion.synchronize()

    torch.testing.assert_close(actual.cpu(), reference, rtol=2e-2, atol=2e-2)
    _assert_accuracy(actual.cpu(), reference)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_qvq_cuda_one_free_threaded_caller_per_device():
    def run(device_index: int, bits: int):
        device = f"cuda:{device_index}"
        with torch.cuda.device(device_index):
            operands, reference = _case(bits, 16, k=128, n=64, dtype=torch.bfloat16, device=device)
            outputs = [qvq_cuda_gemv(*operands, bits, out_features=64).cpu() for _ in range(3)]
        return outputs, reference

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, device, bits) for device, bits in ((0, 1), (1, 8))]
    for future in futures:
        outputs, reference = future.result()
        torch.testing.assert_close(outputs[0], reference, rtol=2e-2, atol=5e-2)
        _assert_accuracy(outputs[0], reference)
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(outputs[1], outputs[2])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_qvq_cuda_viterbi_one_free_threaded_caller_per_device():
    def run(device_index: int, bits: float):
        device = torch.device("cuda", device_index)
        generator = torch.Generator().manual_seed(20260890 + int(bits * 2))
        sequences = torch.randn((16, 128, 2), generator=generator, dtype=torch.float32).to(device)
        codebook = torch.randn((1 << 16, 2), generator=generator, dtype=torch.float32).to(device)
        step_weights = torch.rand((16, 128), generator=generator, dtype=torch.float32).to(device)
        outputs = [qvq_cuda_viterbi(sequences, codebook, bits, step_weights=step_weights) for _ in range(10)]
        torch.cuda.synchronize(device)
        return [(states.cpu(), error.cpu()) for states, error in outputs]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, device, bits) for device, bits in ((0, 1), (1, 1.5))]
    for future in futures:
        outputs = future.result()
        assert all(torch.equal(outputs[0][0], output[0]) for output in outputs[1:])
        assert all(torch.equal(outputs[0][1], output[1]) for output in outputs[1:])


def test_qvq_cuda_viterbi_uses_current_non_default_stream():
    generator = torch.Generator().manual_seed(20260891)
    sequences = torch.randn((4, 128, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = torch.randn((1 << 16, 2), generator=generator, dtype=torch.float32).cuda()
    step_weights = torch.rand((4, 128), generator=generator, dtype=torch.float32).cuda()
    expected = qvq_cuda_viterbi(sequences, codebook, 4, step_weights=step_weights)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_cuda_viterbi(sequences, codebook, 4, step_weights=step_weights)
        completion = torch.cuda.Event()
        completion.record(stream)
    completion.synchronize()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("bits", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
@pytest.mark.parametrize("bank_count,segment_steps", ((2, 16), (4, 32)))
@pytest.mark.parametrize("constrained,weighted", ((False, False), (True, False), (True, True)))
@pytest.mark.parametrize("codebook_dtype", (torch.float16, torch.float32))
def test_qvq_cuda_v2_segment_banked_is_bit_exact_eager_reference(
    monkeypatch,
    bits,
    bank_count,
    segment_steps,
    constrained,
    weighted,
    codebook_dtype,
):
    generator = torch.Generator(device="cuda").manual_seed(
        20260815 + int(bits * 2) * 100 + bank_count * 10 + constrained * 2 + weighted
    )
    sequences = torch.randn((2, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=bits, dtype=torch.float32) for bank in range(bank_count))
    ).to(device="cuda", dtype=codebook_dtype)
    overlap = (
        torch.randint(
            0,
            1 << (16 - qvq_transition_bits(bits, vector_size=2)),
            (2,),
            generator=generator,
            device="cuda",
            dtype=torch.int64,
        )
        if constrained
        else None
    )
    step_weights = (
        (0.1 + torch.rand((2, 128), generator=generator, device="cuda", dtype=torch.float32)).contiguous()
        if weighted
        else None
    )
    native_states, native_loss, native_banks = qvq_cuda_viterbi_v2_segment_banked(
        sequences,
        codebooks,
        bits,
        segment_steps,
        overlap,
        step_weights,
    )
    # Keep all tensors on CUDA while forcing the public quantizer through its
    # eager recurrence. This compares identical FP32 emission arithmetic and
    # avoids using a lower-precision CPU surrogate as the oracle.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    reference = _batched_v2_banked_viterbi_quantize(
        sequences,
        codebooks,
        bits=bits,
        segment_steps=segment_steps,
        overlap=overlap,
        step_weights=step_weights,
    )
    assert torch.equal(native_states, reference.states)
    assert torch.equal(native_banks, reference.segment_bank_ids)
    assert torch.equal(native_loss, reference.squared_error)


@pytest.mark.parametrize("bank_count,segment_steps", ((2, 16), (4, 32)))
def test_qvq_cuda_v2_segment_banked_half_ties_prefer_bank_zero(bank_count, segment_steps):
    sequences = torch.zeros((3, 128, 2), device="cuda", dtype=torch.float32)
    codebook = torch.zeros((1 << 16, 2), device="cuda", dtype=torch.float16)
    codebooks = codebook.unsqueeze(0).expand(bank_count, -1, -1).contiguous()
    states, loss, bank_ids = qvq_cuda_viterbi_v2_segment_banked(
        sequences,
        codebooks,
        bits=1.0,
        segment_steps=segment_steps,
    )
    assert torch.count_nonzero(states) == 0
    assert torch.count_nonzero(loss) == 0
    assert torch.count_nonzero(bank_ids) == 0


@pytest.mark.parametrize("bits", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
@pytest.mark.parametrize("bank_count,segment_steps", ((2, 16), (4, 32)))
def test_qvq_cuda_prevalidated_segmented_v2_matches_public_boundary(bits, bank_count, segment_steps):
    generator = torch.Generator(device="cuda").manual_seed(20260818 + int(bits * 2) * 10 + bank_count)
    sequences = torch.randn((3, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=bits, dtype=torch.float32) for bank in range(bank_count))
    ).to(device="cuda", dtype=torch.float16)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    overlap = torch.randint(
        0,
        1 << (16 - transition_bits),
        (3,),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    step_weights = (0.1 + torch.rand((3, 128), generator=generator, device="cuda")).contiguous()

    expected = qvq_cuda_viterbi_v2_segment_banked(
        sequences,
        codebooks,
        bits,
        segment_steps,
        overlap,
        step_weights,
    )
    actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences,
        codebooks,
        transition_bits,
        segment_steps,
        overlap,
        step_weights,
    )
    assert all(torch.equal(expected_tensor, actual_tensor) for expected_tensor, actual_tensor in zip(expected, actual))


@pytest.mark.parametrize("codebook_dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("constrained", (False, True))
@pytest.mark.parametrize("batch", (1, 7, 8, 9, 17))
def test_qvq_cuda_w25_segmented_cooperative_dispatch_is_exact(
    codebook_dtype,
    weighted,
    constrained,
    batch,
):
    """Cover both cooperative launch geometries and the larger-batch grid fallback."""

    generator = torch.Generator(device="cuda").manual_seed(
        20260826 + batch * 100 + int(weighted) * 10 + int(constrained)
    )
    sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda")
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=2.5, dtype=torch.float32) for bank in range(2))
    ).to(device="cuda", dtype=codebook_dtype)
    overlap = (
        torch.randint(0, 1 << 11, (batch,), generator=generator, device="cuda", dtype=torch.int64)
        if constrained
        else None
    )
    step_weights = (
        (0.1 + torch.rand((batch, 128), generator=generator, device="cuda")).contiguous()
        if weighted
        else None
    )

    expected = qvq_cuda_viterbi_v2_segment_banked(
        sequences,
        codebooks,
        2.5,
        16,
        overlap,
        step_weights,
    )
    for _ in range(3):
        actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
            sequences,
            codebooks,
            5,
            16,
            overlap,
            step_weights,
        )
        assert all(
            torch.equal(expected_tensor, actual_tensor)
            for expected_tensor, actual_tensor in zip(expected, actual)
        )


@pytest.mark.parametrize("bits", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
@pytest.mark.parametrize("bank_count,segment_steps", ((2, 16), (4, 32)))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("batch", (1, 3, 17))
def test_qvq_cuda_segmented_v2_midpoint_matches_full_traceback(
    bits,
    bank_count,
    segment_steps,
    weighted,
    batch,
):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("midpoint-only segmented V2 is an SM80 specialization")

    generator = torch.Generator(device="cuda").manual_seed(
        20260821 + int(bits * 2) * 1_000 + bank_count * 100 + int(weighted) * 10 + batch
    )
    sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=bits, dtype=torch.float32) for bank in range(bank_count))
    ).to(device="cuda", dtype=torch.float16)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    step_weights = (
        (0.1 + torch.rand((batch, 128), generator=generator, device="cuda")).contiguous() if weighted else None
    )

    full_states = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences,
        codebooks,
        transition_bits,
        segment_steps,
        None,
        step_weights,
    )[0]
    expected = full_states[:, 63] & ((1 << (16 - transition_bits)) - 1)
    midpoint_op = _qvq_cuda_viterbi_v2_segment_midpoint_trusted_op()
    for _ in range(3):
        actual = midpoint_op(sequences, codebooks, transition_bits, segment_steps, step_weights)
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("bits", (1.5, 2.0, 2.5, 3.0))
@pytest.mark.parametrize("bank_count,segment_steps", ((2, 16), (4, 32)))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("batch", (1, 7))
def test_qvq_cuda_fused_segmented_tail_matches_two_pass(
    bits,
    bank_count,
    segment_steps,
    weighted,
    batch,
):
    generator = torch.Generator(device="cuda").manual_seed(
        20260824 + int(bits * 2) * 1_000 + bank_count * 100 + int(weighted) * 10 + batch
    )
    sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda")
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=bits, dtype=torch.float32) for bank in range(bank_count))
    ).to(device="cuda", dtype=torch.float16)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    step_weights = (
        (0.1 + torch.rand((batch, 128), generator=generator, device="cuda")).contiguous() if weighted else None
    )
    rotated = torch.roll(sequences, 64, dims=1).contiguous()
    rotated_weights = None if step_weights is None else torch.roll(step_weights, 64, dims=1).contiguous()
    provisional = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        rotated,
        codebooks,
        transition_bits,
        segment_steps,
        None,
        rotated_weights,
    )
    overlap = (provisional[0][:, 63] & ((1 << (16 - transition_bits)) - 1)).contiguous()
    expected = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences,
        codebooks,
        transition_bits,
        segment_steps,
        overlap,
        step_weights,
    )
    actual = _qvq_cuda_viterbi_v2_segment_tail_trusted_op()(
        sequences,
        codebooks,
        transition_bits,
        segment_steps,
        step_weights,
    )
    assert all(torch.equal(expected_tensor, actual_tensor) for expected_tensor, actual_tensor in zip(expected, actual))


@pytest.mark.parametrize("bits", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("batch", (1, 7))
def test_qvq_cuda_fused_canonical_tail_matches_two_pass(bits, weighted, batch):
    generator = torch.Generator(device="cuda").manual_seed(
        20260825 + int(bits * 2) * 100 + int(weighted) * 10 + batch
    )
    sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda")
    codebook = pgc16_codebook_v2_bank(0, bits=bits, dtype=torch.float32).to(
        device="cuda",
        dtype=torch.float16,
    )
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    step_weights = (
        (0.1 + torch.rand((batch, 128), generator=generator, device="cuda")).contiguous() if weighted else None
    )
    rotated = torch.roll(sequences, 64, dims=1).contiguous()
    rotated_weights = None if step_weights is None else torch.roll(step_weights, 64, dims=1).contiguous()
    provisional = _qvq_cuda_viterbi_trusted(rotated, codebook, bits, step_weights=rotated_weights)
    overlap = (provisional[0][:, 63] & ((1 << (16 - transition_bits)) - 1)).contiguous()
    expected = _qvq_cuda_viterbi_trusted(sequences, codebook, bits, overlap, step_weights)
    actual = _qvq_cuda_viterbi_tail_trusted_op()(sequences, codebook, transition_bits, step_weights)
    assert all(torch.equal(expected_tensor, actual_tensor) for expected_tensor, actual_tensor in zip(expected, actual))


@pytest.mark.parametrize("bits", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
@pytest.mark.parametrize("weighted", (False, True))
def test_qvq_cuda_family_batched_segmented_v2_matches_independent_searches(bits, weighted):
    generator = torch.Generator(device="cuda").manual_seed(20260819 + int(bits * 2) + int(weighted) * 100)
    families, batch = 3, 3
    sequences = torch.randn((families, batch, 128, 2), generator=generator, device="cuda")
    codebooks = torch.stack(
        tuple(
            torch.stack(
                (
                    pgc16_codebook_v2_bank(0, bits=bits, dtype=torch.float32),
                    pgc16_codebook_v2_bank(family, bits=bits, dtype=torch.float32),
                )
            )
            for family in (1, 2, 3)
        )
    ).to(device="cuda", dtype=torch.float16)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    overlap = torch.randint(
        0,
        1 << (16 - transition_bits),
        (families, batch),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    step_weights = (
        (0.1 + torch.rand((families, batch, 128), generator=generator, device="cuda")) if weighted else None
    )

    expected = tuple(
        tuple(
            tensor
            for tensor in _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
                sequences[family],
                codebooks[family],
                transition_bits,
                16,
                overlap[family],
                None if step_weights is None else step_weights[family],
            )
        )
        for family in range(families)
    )
    actual = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()(
        sequences,
        codebooks,
        transition_bits,
        16,
        overlap,
        step_weights,
    )

    for family in range(families):
        assert all(torch.equal(expected[family][index], actual[index][family]) for index in range(3))


def _fused_family_grid_dispatch_count() -> int:
    """Process-wide count of fused W2 family-grid kernel dispatches (see qvq_viterbi_cuda.cu)."""

    _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()  # ensure the extension is loaded
    return int(torch.ops.gptqmodel_qvq.fused_family_grid_dispatch_count())


# Round-2 relaxed contract for the fused W2 family-grid kernel: floating-point
# op order is no longer bit-identical to the reference path (squared-difference
# emission, weight folded into an FMA), so the fused path is held to *decision
# equivalence* instead of bit-exactness: discrete outputs match the reference
# except for quantified, score-equal near-tie flips, and the accumulated
# squared error matches within FP32 accumulation error.
#
# Tolerance derivation.  A per-sequence objective is a sum of 128 non-negative
# terms, each built from <= ~10 FP32 ops; with u = 2^-24 and effective op depth
# n ~ 138, the standard forward-error bound gives
# |fl(S) - S| <= gamma_n * sum|terms| = gamma_n * S (all terms >= 0), with
# gamma_138 = 138u / (1 - 138u) ~= 8.2e-6.  Two independently accumulated
# scores (reference vs fused op order) can therefore disagree by up to
# ~2*gamma ~= 1.6e-5 relative while the true objectives are equal or closer:
_FUSED_ACCUMULATION_RTOL = 2e-5
# The reference emission is the expanded form max(tn + cn - 2*dot, 0), which
# cancels catastrophically when a step distance is ~0 (its absolute error per
# step is ~u * (tn + cn), not relative); over 128 steps with |t|,|c| = O(1)
# that is an absolute slack of ~128 * 6e-8 * O(4) ~= 3e-5, relevant for the
# adversarial codebook-snapped cases where the true objective itself is ~0:
_FUSED_ACCUMULATION_ATOL = 1e-4
# Fixed-seed comparison-grid tests: measured worst per-family state-flip
# fraction is 6.6e-4 (an 11-state near-tie cluster at input scale 4.0,
# unweighted, batch 131); bound = measured worst + ~50 % margin:
_FUSED_MAX_FLIP_FRACTION = 1e-3
# Kernel-reported loss vs reference-reported loss for the fixed-seed tests
# (both FP32-accumulated under different op orders): the 2*gamma relative
# bound above, doubled for headroom on the weighted per-term multiply:
_FUSED_LOSS_RTOL = 4e-5
_FUSED_LOSS_ATOL = 1e-4


def _assert_family_grid_decision_equivalent(
    expected, actual, context, max_flip_fraction=_FUSED_MAX_FLIP_FRACTION
):
    """expected/actual: (states, squared_error, segment_bank_ids) triples."""

    states_e, loss_e, banks_e = expected
    states_a, loss_a, banks_a = actual
    flip_fraction = (states_e != states_a).float().mean().item()
    assert flip_fraction <= max_flip_fraction, (context, flip_fraction)
    bank_flip_fraction = (banks_e != banks_a).float().mean().item()
    assert bank_flip_fraction <= max(max_flip_fraction * 16, 16 / banks_e.numel()), (
        context,
        bank_flip_fraction,
    )
    torch.testing.assert_close(
        loss_a, loss_e, rtol=_FUSED_LOSS_RTOL, atol=_FUSED_LOSS_ATOL, msg=lambda m: f"{context}: {m}"
    )


# The fused path is gated on the *flattened* batch (families x batch >= 40): per-family
# batches 1, 2, 3 and 7 (3, 6, 9, 21 sequences) take the reference path, 33+ (99+) the fused kernel.
@pytest.mark.parametrize("batch", (1, 2, 3, 7, 33, 128, 131))
@pytest.mark.parametrize("constrained", (False, True))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("scale", (1.0, 0.05, 4.0))
def test_qvq_cuda_fused_w2_family_grid_is_decision_equivalent_to_reference_grid(batch, constrained, weighted, scale):
    """The W2 <4,2,16,fused> family op runs the fused persistent kernel; the per-family
    ``viterbi_v2_segment_grid_trusted`` op still runs the reference segmented grid kernels.
    Small batches take the (bit-exact) reference path; fused batches are decision-equivalent."""

    bits = 2.0
    generator = torch.Generator(device="cuda").manual_seed(20260823 + batch * 7 + int(constrained) + 2 * int(weighted))
    families = 3
    sequences = torch.randn((families, batch, 128, 2), generator=generator, device="cuda") * scale
    codebooks = torch.stack(
        tuple(
            torch.stack(
                (
                    pgc16_codebook_v2_bank(0, bits=bits, dtype=torch.float32),
                    pgc16_codebook_v2_bank(family, bits=bits, dtype=torch.float32),
                )
            )
            for family in (1, 2, 3)
        )
    ).to(device="cuda", dtype=torch.float16)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    assert transition_bits == 4
    overlap = (
        torch.randint(0, 1 << (16 - transition_bits), (families, batch), generator=generator, device="cuda", dtype=torch.int64)
        if constrained
        else None
    )
    step_weights = (0.1 + torch.rand((families, batch, 128), generator=generator, device="cuda")) if weighted else None

    dispatches_before = _fused_family_grid_dispatch_count()
    actual = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()(
        sequences, codebooks, transition_bits, 16, overlap, step_weights
    )
    expected_fused_dispatches = 1 if families * batch >= 40 else 0
    assert _fused_family_grid_dispatch_count() - dispatches_before == expected_fused_dispatches
    for family in range(families):
        expected = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
            sequences[family],
            codebooks[family],
            transition_bits,
            16,
            None if overlap is None else overlap[family],
            None if step_weights is None else step_weights[family],
        )
        if expected_fused_dispatches == 0:
            for index in range(3):
                assert torch.equal(expected[index], actual[index][family]), (family, index)
        else:
            _assert_family_grid_decision_equivalent(
                expected,
                tuple(tensor[family] for tensor in actual),
                (batch, constrained, weighted, scale, family),
            )


@pytest.mark.parametrize("constrained", (False, True))
@pytest.mark.parametrize("weighted", (False, True))
def test_qvq_cuda_fused_w2_family_grid_general_codebook_fallback_is_decision_equivalent(constrained, weighted):
    """Bank 1 that is not an XOR-permutation of bank 0 must take the in-kernel general path
    (fused_step_general); batch >= 40 per family keeps the fused kernel dispatched."""

    bits = 2.0
    generator = torch.Generator(device="cuda").manual_seed(20260824 + int(constrained) + 2 * int(weighted))
    families, batch = 3, 41
    sequences = torch.randn((families, batch, 128, 2), generator=generator, device="cuda")
    bank0 = pgc16_codebook_v2_bank(0, bits=bits, dtype=torch.float32).to("cuda", torch.float16)
    codebooks = torch.stack(
        tuple(
            torch.stack((bank0, torch.randn((1 << 16, 2), generator=generator, device="cuda").to(torch.float16)))
            for _ in range(families)
        )
    ).contiguous()
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    overlap = (
        torch.randint(0, 1 << (16 - transition_bits), (families, batch), generator=generator, device="cuda", dtype=torch.int64)
        if constrained
        else None
    )
    step_weights = (0.1 + torch.rand((families, batch, 128), generator=generator, device="cuda")) if weighted else None
    dispatches_before = _fused_family_grid_dispatch_count()
    actual = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()(
        sequences, codebooks, transition_bits, 16, overlap, step_weights
    )
    assert _fused_family_grid_dispatch_count() - dispatches_before == 1
    for family in range(families):
        expected = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
            sequences[family],
            codebooks[family],
            transition_bits,
            16,
            None if overlap is None else overlap[family],
            None if step_weights is None else step_weights[family],
        )
        _assert_family_grid_decision_equivalent(
            expected,
            tuple(tensor[family] for tensor in actual),
            (constrained, weighted, family),
        )


@pytest.mark.parametrize("batch", (40, 67, 128))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("xor_related_banks", (True, False))
def test_qvq_cuda_fused_w2_segment_tail_matches_reference_two_pass(batch, weighted, xor_related_banks):
    """viterbi_v2_segment_tail_trusted at W2 with >= 40 sequences runs both passes on the fused
    family-grid kernel; under the round-2 relaxed contract it must be decision-equivalent
    (sequence-wise, see below) to the reference two-pass construction built from the
    (never fused) viterbi_v2_segment_grid_trusted op."""

    bits = 2.0
    generator = torch.Generator(device="cuda").manual_seed(20260826 + batch * 10 + int(weighted) + 2 * int(xor_related_banks))
    sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda")
    bank0 = pgc16_codebook_v2_bank(0, bits=bits, dtype=torch.float32).to("cuda", torch.float16)
    bank1 = (
        pgc16_codebook_v2_bank(2, bits=bits, dtype=torch.float32).to("cuda", torch.float16)
        if xor_related_banks
        else torch.randn((1 << 16, 2), generator=generator, device="cuda").to(torch.float16)
    )
    codebooks = torch.stack((bank0, bank1)).contiguous()
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    assert transition_bits == 4
    step_weights = (
        (0.1 + torch.rand((batch, 128), generator=generator, device="cuda")).contiguous() if weighted else None
    )
    rotated = torch.roll(sequences, 64, dims=1).contiguous()
    rotated_weights = None if step_weights is None else torch.roll(step_weights, 64, dims=1).contiguous()
    reference = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    provisional = reference(rotated, codebooks, transition_bits, 16, None, rotated_weights)
    overlap = (provisional[0][:, 63] & ((1 << (16 - transition_bits)) - 1)).contiguous()
    expected = reference(sequences, codebooks, transition_bits, 16, overlap, step_weights)

    dispatches_before = _fused_family_grid_dispatch_count()
    actual = _qvq_cuda_viterbi_v2_segment_tail_trusted_op()(sequences, codebooks, transition_bits, 16, step_weights)
    # Provisional (rotated) pass + constrained pass, both on the fused kernel.
    assert _fused_family_grid_dispatch_count() - dispatches_before == 2
    # Decision equivalence, sequence-wise: a near-tie flip in the provisional
    # pass changes the derived overlap and legitimately diverges that whole
    # sequence, so diverged sequences are held to equal-or-better final loss
    # instead of state equality.
    states_e, loss_e, banks_e = expected
    states_a, loss_a, banks_a = actual
    diverged = (states_e != states_a).any(dim=1)
    assert diverged.float().mean().item() <= 0.05, diverged.sum().item()
    matched = ~diverged
    torch.testing.assert_close(
        loss_a[matched], loss_e[matched], rtol=_FUSED_LOSS_RTOL, atol=_FUSED_LOSS_ATOL
    )
    assert torch.equal(banks_e[matched], banks_a[matched])
    if bool(diverged.any()):
        assert bool(
            (loss_a[diverged] <= loss_e[diverged] * (1 + _FUSED_LOSS_RTOL) + _FUSED_LOSS_ATOL).all()
        ), (loss_a[diverged], loss_e[diverged])


def test_qvq_cuda_fused_w2_family_grid_disable_env_forces_reference_path():
    """QVQ_DISABLE_FUSED_FAMILY_GRID=1 must route a fused-eligible batch through
    the reference path (0 fused dispatches, bit-exact outputs).  The flag is
    read once per process, so this runs in a subprocess."""

    import os
    import subprocess
    import sys

    script = r"""
import torch
from gptqmodel.quantization.qvq_codecs.pgc16 import pgc16_codebook_v2_bank
from gptqmodel.utils.qvq_cuda import (
    _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op,
    _qvq_cuda_viterbi_v2_segment_grid_trusted_op,
)
fam = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()
ref = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
generator = torch.Generator(device="cuda").manual_seed(20260902)
families, batch = 3, 41
sequences = torch.randn((families, batch, 128, 2), generator=generator, device="cuda")
codebooks = torch.stack(
    tuple(
        torch.stack(
            (
                pgc16_codebook_v2_bank(0, bits=2.0, dtype=torch.float32),
                pgc16_codebook_v2_bank(f + 1, bits=2.0, dtype=torch.float32),
            )
        )
        for f in range(families)
    )
).to(device="cuda", dtype=torch.float16)
actual = fam(sequences, codebooks, 4, 16, None, None)
assert int(torch.ops.gptqmodel_qvq.fused_family_grid_dispatch_count()) == 0
for f in range(families):
    expected = ref(sequences[f], codebooks[f], 4, 16, None, None)
    for index in range(3):
        assert torch.equal(expected[index], actual[index][f]), (f, index)
print("DISABLED-PATH-OK")
"""
    env = dict(os.environ, QVQ_DISABLE_FUSED_FAMILY_GRID="1")
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "DISABLED-PATH-OK" in result.stdout


def _rescore_family_grid_paths_fp64(sequences, codebooks, states, bank_ids, step_weights):
    """Independent re-scoring of a returned discrete path under ONE common
    reference objective: sum_s w_s * ||t_s - codebook[bank(s), state_s]||^2,
    evaluated in fp64 (the clip is a no-op for a sum of squares).  Both
    kernels' paths go through this identical arithmetic, so comparing rescored
    values proves score equality of the *decisions* independently of either
    kernel's own FP32 accumulation."""

    steps = states.shape[1]
    banks = bank_ids.to(torch.long).repeat_interleave(steps // bank_ids.shape[1], dim=1)
    codes = codebooks.to(torch.float64)[banks, states]
    diff = sequences.to(torch.float64) - codes
    emission = (diff * diff).sum(dim=-1)
    if step_weights is not None:
        emission = emission * step_weights.to(torch.float64)
    return emission.sum(dim=1)


def _assert_valid_trellis_paths(states):
    """Consecutive states must chain through the 12-bit suffix window."""

    assert bool(((states[:, :-1] & 0xFFF) == (states[:, 1:] >> 4)).all())


def test_qvq_cuda_fused_w2_family_grid_randomized_stress_decision_equivalence():
    """Round-2 relaxed-contract stress test: >= 10k fused-path random sequences
    through the family-grid op, spanning the batch gate boundary,
    weighted/unweighted, constrained/unconstrained, XOR-related and arbitrary
    bank pairs, plus adversarial near-tie cases (duplicated codes,
    codebook-snapped targets).  Both kernels' returned discrete paths are
    independently RESCORED under one common fp64 objective
    (_rescore_family_grid_paths_fp64): sequences without flips must rescore
    bitwise-identically, sequences with flips must rescore within the FP32
    accumulation bound (the proof that every flip is a genuine near-tie), and
    each kernel's own reported loss must match its own path's rescore (a
    traceback/reporting consistency check)."""

    fam = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()
    ref = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    families = 3
    fused_sequences = 0
    fused_states = 0
    flip_deltas = []
    per_category_stats = {}
    generator = torch.Generator(device="cuda").manual_seed(20260901)

    def bank_pair(family, xor_related, adversarial_dup):
        bank0 = pgc16_codebook_v2_bank(0, bits=2.0, dtype=torch.float32)
        if xor_related:
            bank1 = pgc16_codebook_v2_bank(family + 1, bits=2.0, dtype=torch.float32)
        else:
            bank1 = torch.randn((1 << 16, 2), generator=generator, device="cuda").cpu()
        pair = torch.stack((bank0, bank1)).to(device="cuda", dtype=torch.float16)
        if adversarial_dup:
            # Duplicate a block of codes so exact score ties are guaranteed.
            pair[:, 1024:2048] = pair[:, 0:1024]
        return pair

    cases = []
    for call in range(24):  # 24 x (3 x 128) = 9216 sequences on the fused path
        cases.append((128, call % 2 == 0, call % 3 == 0, True, call % 4 == 0, False))
    for batch in (13, 14, 33, 40, 41, 67):  # around/above the flattened gate (>= 40)
        cases.append((batch, True, True, True, False, False))
        cases.append((batch, False, False, False, False, False))
    for _ in range(4):  # adversarial: targets snapped to codebook entries
        cases.append((64, True, False, True, False, True))

    for batch, weighted, constrained, xor_related, adversarial_dup, adversarial_snap in cases:
        codebooks = torch.stack(
            tuple(bank_pair(f, xor_related, adversarial_dup) for f in range(families))
        ).contiguous()
        sequences = torch.randn((families, batch, 128, 2), generator=generator, device="cuda")
        if adversarial_snap:
            picks = torch.randint(0, 1 << 16, (families, batch, 128), generator=generator, device="cuda")
            for f in range(families):
                sequences[f] = codebooks[f, 0].float()[picks[f]]
        overlap = (
            torch.randint(0, 1 << 12, (families, batch), generator=generator, device="cuda", dtype=torch.int64)
            if constrained
            else None
        )
        step_weights = (
            (0.1 + torch.rand((families, batch, 128), generator=generator, device="cuda")) if weighted else None
        )
        dispatches_before = _fused_family_grid_dispatch_count()
        actual = fam(sequences, codebooks, 4, 16, overlap, step_weights)
        expected_fused = 1 if families * batch >= 40 else 0
        assert _fused_family_grid_dispatch_count() - dispatches_before == expected_fused, batch
        case_key = (
            ("xor" if xor_related else "arbitrary")
            + ("+dup" if adversarial_dup else "")
            + ("+snap" if adversarial_snap else "")
            + ("+w" if weighted else "")
            + ("+c" if constrained else "")
        )
        for f in range(families):
            expected = ref(
                sequences[f],
                codebooks[f],
                4,
                16,
                None if overlap is None else overlap[f],
                None if step_weights is None else step_weights[f],
            )
            states_e, loss_e, banks_e = expected
            states_a, loss_a, banks_a = (tensor[f] for tensor in actual)
            if expected_fused == 0:
                assert torch.equal(states_e, states_a) and torch.equal(loss_e, loss_a)
                assert torch.equal(banks_e, banks_a)
                continue
            context = (batch, case_key, f)
            _assert_valid_trellis_paths(states_e)
            _assert_valid_trellis_paths(states_a)
            weights_f = None if step_weights is None else step_weights[f]
            rescore_e = _rescore_family_grid_paths_fp64(
                sequences[f], codebooks[f], states_e, banks_e, weights_f)
            rescore_a = _rescore_family_grid_paths_fp64(
                sequences[f], codebooks[f], states_a, banks_a, weights_f)
            # Each kernel's own reported FP32 loss must match its own path's
            # fp64 rescore within the accumulation bound (catches traceback or
            # loss-reporting bugs independently of any flips).
            for own_loss, own_rescore, which in (
                (loss_e, rescore_e, "reference"), (loss_a, rescore_a, "fused")):
                own_delta = (own_loss.to(torch.float64) - own_rescore).abs()
                own_bound = _FUSED_ACCUMULATION_ATOL + _FUSED_ACCUMULATION_RTOL * own_rescore.abs()
                assert bool((own_delta <= own_bound).all()), (
                    context, which, float(own_delta.max()), float(own_rescore.max()))
            flipped = ((states_e != states_a).any(dim=1)) | ((banks_e != banks_a).any(dim=1))
            # Identical discrete paths must rescore bitwise-identically.
            assert torch.equal(rescore_e[~flipped], rescore_a[~flipped]), context
            # Flipped paths are score-equal iff their fp64 rescores agree
            # within the FP32 accumulation bound derived above: two candidates
            # can only swap order when their FP32-accumulated costs are inside
            # each other's rounding envelopes.
            if bool(flipped.any()):
                delta = (rescore_a[flipped] - rescore_e[flipped]).abs()
                bound = _FUSED_ACCUMULATION_ATOL + _FUSED_ACCUMULATION_RTOL * rescore_e[flipped].abs()
                assert bool((delta <= bound).all()), (
                    context, float(delta.max()), float(rescore_e[flipped].min()))
                flip_deltas.extend(
                    (delta / rescore_e[flipped].abs().clamp_min(1e-9)).tolist())
            fused_sequences += batch
            fused_states += states_e.numel()
            stats = per_category_stats.setdefault(
                case_key, {"sequences": 0, "states": 0, "flip_sequences": 0, "flip_states": 0})
            stats["sequences"] += batch
            stats["states"] += states_e.numel()
            stats["flip_sequences"] += int(flipped.sum())
            stats["flip_states"] += int((states_e != states_a).sum())

    # Deterministic totals of the fused-path comparison, derived from `cases`:
    # only calls with families * batch >= 40 dispatch the fused kernel.
    expected_sequences = sum(families * b for (b, *_rest) in cases if families * b >= 40)
    assert fused_sequences == expected_sequences == 11_154, (fused_sequences, expected_sequences)
    assert fused_states == expected_sequences * 128 == 1_427_712, fused_states
    flip_sequences = sum(s["flip_sequences"] for s in per_category_stats.values())
    flip_states = sum(s["flip_states"] for s in per_category_stats.values())
    report = {
        "fused_sequences": fused_sequences,
        "fused_states": fused_states,
        "flip_sequences": flip_sequences,
        "flip_states": flip_states,
        "max_flip_rescore_rel": max(flip_deltas, default=0.0),
        "per_case_category": per_category_stats,
    }
    print("fused family-grid stress report:", report)
    # Flip-rate bound: measured-distribution-plus-margin (see the report print
    # for the measured values; historically ~1e-5 of states, ~1e-3 of
    # sequences).  An order of magnitude above measured still sits far below
    # any rate that could move real-workload quality.
    assert flip_states / fused_states <= 1e-4, report
    assert flip_sequences / fused_sequences <= 1e-2, report


def test_qvq_cuda_sampled_yaqa_family_batch_matches_serial_selection(monkeypatch):
    bits = 2.5
    generator = torch.Generator(device="cuda").manual_seed(20260820)
    source = (torch.randn((32, 32), generator=generator, device="cuda") * 0.05).to(torch.float16)
    input_samples = torch.randn((41, 32), generator=generator, device="cuda")
    output_samples = torch.randn((37, 32), generator=generator, device="cuda")
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    input_hessian.diagonal().add_(0.1)
    output_hessian.diagonal().add_(0.1)
    banks = _canonical_qvq_v2b4_banks(
        device=source.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float16,
    )
    pair_stacks = _canonical_qvq_v2b2_pair_stacks(
        device=source.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float16,
    )

    native_resolver = qvq_cuda_utils._qvq_cuda_viterbi_v2_segment_family_grid_trusted_op

    def serial_family_op(sequences, codebooks, transition_bits, segment_steps, overlap, step_weights):
        results = tuple(
            _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
                sequences[family],
                codebooks[family],
                transition_bits,
                segment_steps,
                None if overlap is None else overlap[family],
                None if step_weights is None else step_weights[family],
            )
            for family in range(sequences.shape[0])
        )
        return tuple(torch.stack(tuple(result[index] for result in results)) for index in range(3))

    common = {
        "bits": bits,
        "family_mode": "reselect",
        "sample_strategy": "32_16x16",
        "block_family_id": 1,
        "bank_codebook_pair_stacks": pair_stacks,
    }
    monkeypatch.setattr(
        qvq_cuda_utils,
        "_qvq_cuda_viterbi_v2_segment_family_grid_trusted_op",
        lambda: serial_family_op,
    )
    expected = yaqa_inner_v2b2_p32(source, input_hessian, output_hessian, banks, **common)
    monkeypatch.setattr(qvq_cuda_utils, "_qvq_cuda_viterbi_v2_segment_family_grid_trusted_op", native_resolver)
    actual = yaqa_inner_v2b2_p32(source, input_hessian, output_hessian, banks, **common)

    assert all(torch.equal(expected_tensor, actual_tensor) for expected_tensor, actual_tensor in zip(expected, actual))


@pytest.mark.parametrize("bank_count,segment_steps", ((2, 16), (4, 32)))
def test_qvq_cuda_v2_segment_banked_multiwave_grid_is_deterministic(bank_count, segment_steps):
    """Cover more bank CTAs than one 124-SM A100 wave with identical work."""

    generator = torch.Generator(device="cuda").manual_seed(20260817 + bank_count)
    one_sequence = torch.randn((1, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    sequences = one_sequence.expand(64, -1, -1).contiguous()
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=1.5, dtype=torch.float32) for bank in range(bank_count))
    ).to(device="cuda", dtype=torch.float16)
    expected = _qvq_cuda_viterbi_v2_segment_g_op()(
        sequences,
        codebooks,
        qvq_transition_bits(1.5, vector_size=2),
        segment_steps,
        None,
        None,
    )
    for _ in range(5):
        actual = qvq_cuda_viterbi_v2_segment_banked(sequences, codebooks, 1.5, segment_steps)
        assert all(
            torch.equal(expected_tensor, actual_tensor)
            for expected_tensor, actual_tensor in zip(expected, actual)
        )


@pytest.mark.parametrize("bank_count,segment_steps", ((2, 16), (4, 32)))
def test_qvq_cuda_v2_segment_banked_w3_5_uses_current_non_default_stream(bank_count, segment_steps):
    generator = torch.Generator(device="cuda").manual_seed(20260816 + bank_count)
    sequences = torch.randn((3, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=3.5, dtype=torch.float32) for bank in range(bank_count))
    ).cuda()
    weights = (0.1 + torch.rand((3, 128), generator=generator, device="cuda")).contiguous()
    overlap = torch.randint(0, 1 << 9, (3,), generator=generator, device="cuda", dtype=torch.int64)
    expected = qvq_cuda_viterbi_v2_segment_banked(
        sequences, codebooks, 3.5, segment_steps, overlap, weights
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_cuda_viterbi_v2_segment_banked(
            sequences, codebooks, 3.5, segment_steps, overlap, weights
        )
        completion = torch.cuda.Event()
        completion.record(stream)
    completion.synchronize()
    assert all(torch.equal(expected_tensor, actual_tensor) for expected_tensor, actual_tensor in zip(expected, actual))


def test_qvq_cuda_v2_segment_banked_public_quantizer_routes_native():
    generator = torch.Generator(device="cuda").manual_seed(20260817)
    sequences = torch.randn((2, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=2.0, dtype=torch.float32) for bank in range(2))
    ).cuda()
    with patch(
        "gptqmodel.utils.qvq_cuda.qvq_cuda_viterbi_v2_segment_banked",
        wraps=qvq_cuda_viterbi_v2_segment_banked,
    ) as native:
        result = _batched_v2_banked_viterbi_quantize(
            sequences,
            codebooks,
            bits=2.0,
            segment_steps=16,
        )
    assert native.call_count == 1
    assert result.states.shape == (2, 128)
    assert result.segment_bank_ids.shape == (2, 8)


@pytest.mark.parametrize(
    "bank_count,segment_steps,error",
    ((2, 32, "two P32 banks"), (4, 16, "four P64 banks"), (3, 16, "shape")),
)
def test_qvq_cuda_v2_segment_banked_rejects_invalid_format(bank_count, segment_steps, error):
    sequences = torch.zeros((1, 128, 2), device="cuda", dtype=torch.float32)
    codebooks = torch.zeros((bank_count, 1 << 16, 2), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match=error):
        qvq_cuda_viterbi_v2_segment_banked(sequences, codebooks, 2.0, segment_steps)


def test_qvq_v4_banked_viterbi_matches_four_serial_reference_runs():
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    sequences = torch.randn((8, 17, 4), generator=generator).cuda()
    banks = torch.stack(
        _canonical_qvq_v4_banks(
            device=sequences.device,
            bits=2.0,
            codebook_version=PGC16_CODEBOOK_VERSION,
            dtype=torch.float16,
        )
    ).contiguous()

    banked_states, banked_error = qvq_cuda_viterbi_banked(sequences, banks, bits=2.0)
    serial_states = []
    serial_error = []
    for bank in range(4):
        states, error = qvq_cuda_viterbi(sequences, banks[bank], bits=2.0, vector_size=4)
        serial_states.append(states)
        serial_error.append(error)

    assert torch.equal(banked_states, torch.stack(serial_states))
    assert torch.equal(banked_error, torch.stack(serial_error))


def test_qvq_v4_banked_viterbi_supports_three_active_propagation_banks():
    generator = torch.Generator(device="cpu").manual_seed(20260820)
    sequences = torch.randn((5, 19, 4), generator=generator, dtype=torch.float32).cuda()
    banks = torch.randn((3, 1 << 16, 4), generator=generator, dtype=torch.float16).cuda()

    banked_states, banked_error = qvq_cuda_viterbi_banked(sequences, banks, bits=2.0)
    serial = [
        qvq_cuda_viterbi(sequences, banks[bank], bits=2.0, vector_size=4)
        for bank in range(3)
    ]
    torch.cuda.synchronize()
    assert torch.equal(banked_states, torch.stack([result[0] for result in serial]))
    assert torch.equal(banked_error, torch.stack([result[1] for result in serial]))


def test_qvq_v4_banked_memoryless_w4_matches_serial_runs():
    generator = torch.Generator(device="cpu").manual_seed(20260815)
    sequences = torch.randn((3, 9, 4), generator=generator, dtype=torch.float32).cuda()
    banks = torch.randn((4, 1 << 16, 4), generator=generator, dtype=torch.float16).cuda()

    banked_states, banked_error = qvq_cuda_viterbi_banked(sequences, banks, bits=4.0)
    serial = [
        qvq_cuda_viterbi(sequences, banks[bank], bits=4.0, vector_size=4)
        for bank in range(4)
    ]
    torch.cuda.synchronize()
    assert torch.equal(banked_states, torch.stack([result[0] for result in serial]))
    torch.testing.assert_close(banked_error, torch.stack([result[1] for result in serial]), rtol=0, atol=0)


def test_qvq_v4_banked_independent_sequences_match_serial_runs():
    generator = torch.Generator(device="cpu").manual_seed(20260819)
    sequences = torch.randn((4, 2, 13, 4), generator=generator, dtype=torch.float32).cuda()
    banks = torch.randn((4, 1 << 16, 4), generator=generator, dtype=torch.float16).cuda()

    banked_states, banked_error = qvq_cuda_viterbi_banked(sequences, banks, bits=2.0)
    serial = [
        qvq_cuda_viterbi(sequences[bank], banks[bank], bits=2.0, vector_size=4)
        for bank in range(4)
    ]
    torch.cuda.synchronize()
    assert torch.equal(banked_states, torch.stack([result[0] for result in serial]))
    assert torch.equal(banked_error, torch.stack([result[1] for result in serial]))


def test_qvq_v4_banked_weighted_constrained_non_default_stream_matches_serial():
    generator = torch.Generator(device="cpu").manual_seed(20260816)
    sequences = torch.randn((2, 11, 4), generator=generator, dtype=torch.float32).cuda()
    banks = torch.randn((4, 1 << 16, 4), generator=generator, dtype=torch.float16).cuda()
    weights = (0.1 + torch.rand((2, 11), generator=generator, dtype=torch.float32)).cuda()
    transition_bits = qvq_transition_bits(2.0, vector_size=4)
    overlap_bits = 16 - transition_bits
    overlap = torch.randint(0, 1 << overlap_bits, (8,), generator=generator, dtype=torch.int64).cuda()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        banked = qvq_cuda_viterbi_banked(sequences, banks, 2.0, overlap=overlap, step_weights=weights)
    stream.synchronize()
    serial = [
        qvq_cuda_viterbi(
            sequences,
            banks[bank],
            2.0,
            overlap=overlap[bank * sequences.shape[0] : (bank + 1) * sequences.shape[0]],
            step_weights=weights,
            vector_size=4,
        )
        for bank in range(4)
    ]
    torch.cuda.synchronize()
    assert torch.equal(banked[0], torch.stack([result[0] for result in serial]))
    assert torch.equal(banked[1], torch.stack([result[1] for result in serial]))


def test_qvq_banked_viterbi_norm_cache_uses_versioned_codebooks():
    codebook = torch.linspace(-1, 1, 4 * (1 << 16) * 4, dtype=torch.float32, device="cuda").to(
        torch.float16
    ).reshape(4, 1 << 16, 4)
    sequences = torch.zeros((4, 8, 4), dtype=torch.float32, device="cuda")

    original_states, _ = qvq_cuda_viterbi_banked(sequences, codebook, bits=2.0)
    mutated = codebook.clone()
    mutated[:, 0, 0].add_(torch.as_tensor(4096, device="cuda", dtype=codebook.dtype))
    codebook[:, 0, 0].add_(4096)

    updated_mutated_states, updated_mutated_error = qvq_cuda_viterbi_banked(sequences, codebook, bits=2.0)
    updated_expected_states, updated_expected_error = qvq_cuda_viterbi_banked(sequences, mutated, bits=2.0)

    torch.cuda.synchronize()
    # The mutation need not change the argmin state for this deliberately
    # simple zero-input fixture; it must nevertheless produce the same result
    # as a fresh tensor with the mutated values, rather than a stale cache hit.
    assert torch.equal(updated_mutated_states, updated_expected_states)
    torch.testing.assert_close(updated_mutated_error, updated_expected_error, rtol=0, atol=0)


def test_qvq_viterbi_accepts_inference_mode_codebooks():
    generator = torch.Generator(device="cpu").manual_seed(20260817)
    sequences = torch.randn((2, 7, 4), generator=generator, dtype=torch.float32).cuda()
    codebook = torch.randn((1 << 16, 4), generator=generator, dtype=torch.float16).cuda()
    with torch.inference_mode():
        states, error = qvq_cuda_viterbi(sequences, codebook, bits=2.0, vector_size=4)
    assert states.shape == (2, 7)
    assert torch.isfinite(error).all()


def test_qvq_viterbi_inference_mode_codebook_mutation_rebuilds_norms():
    generator = torch.Generator(device="cpu").manual_seed(20260823)
    sequences = torch.randn((2, 9, 4), generator=generator, dtype=torch.float32).cuda()
    source = torch.randn((1 << 16, 4), generator=generator, dtype=torch.float16).cuda()
    with torch.inference_mode():
        codebook = source.clone()
        qvq_cuda_viterbi(sequences, codebook, bits=2.0, vector_size=4)
        codebook[0, 0].add_(1.0)
        mutated_states, mutated_error = qvq_cuda_viterbi(sequences, codebook, bits=2.0, vector_size=4)
        fresh_states, fresh_error = qvq_cuda_viterbi(sequences, codebook.clone(), bits=2.0, vector_size=4)
    torch.cuda.synchronize()
    assert torch.equal(mutated_states, fresh_states)
    torch.testing.assert_close(mutated_error, fresh_error, rtol=0, atol=0)


def test_qvq_native_banked_viterbi_rejects_mismatched_sequence_bank_count():
    sequences = torch.zeros((3, 2, 5, 4), device="cuda", dtype=torch.float32)
    codebooks = torch.zeros((4, 1 << 16, 4), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="bank_count"):
        torch.ops.gptqmodel_qvq.viterbi_banked(sequences, codebooks, 4)


def test_qvq_norm_cache_eviction_waits_for_cross_stream_consumer():
    """Evicting norm entries must not race an outstanding consumer stream."""

    generator = torch.Generator(device="cpu").manual_seed(20260814)
    sequences = torch.randn((1, 33, 4), generator=generator, dtype=torch.float32).cuda()
    codebooks = [
        torch.randn((1 << 16, 4), generator=generator, dtype=torch.float16).cuda() for _ in range(33)
    ]
    consumer_stream = torch.cuda.Stream()
    with torch.cuda.stream(consumer_stream):
        first_states, first_error = qvq_cuda_viterbi(
            sequences, codebooks[0], bits=2.0, vector_size=4
        )
    for codebook in codebooks[1:]:
        qvq_cuda_viterbi(sequences, codebook, bits=2.0, vector_size=4)
    consumer_stream.synchronize()

    expected_states, expected_error = qvq_cuda_viterbi(
        sequences, codebooks[0], bits=2.0, vector_size=4
    )
    torch.cuda.synchronize()
    assert torch.equal(first_states, expected_states)
    torch.testing.assert_close(first_error, expected_error, rtol=0, atol=0)


def test_qvq_propagated_banked_candidates_parallel_matches_serial():
    generator = torch.Generator(device="cpu").manual_seed(20260818)
    inner = torch.randn((32, 32), generator=generator, dtype=torch.float32).cuda()
    hessian_source = torch.randn((32, 32), generator=generator, dtype=torch.float32).cuda()
    hessian = hessian_source @ hessian_source.T + torch.eye(32, dtype=torch.float32, device="cuda") * 0.25
    banks = tuple(
        torch.randn((1 << 16, 4), generator=generator, dtype=torch.float16).cuda() for _ in range(4)
    )
    parallel_weights, parallel_states = block_ldlq_inner_banked_candidates(
        inner,
        hessian,
        banks,
        bits=2.0,
        tile_rows=16,
        tile_cols=16,
        trellis_batch_size=1,
    )
    serial_results = [
        block_ldlq_inner(
            inner,
            hessian,
            bank,
            bits=2.0,
            tile_rows=16,
            tile_cols=16,
            trellis_batch_size=1,
            telemetry=QVQQuantizationTelemetry(),
        )
        for bank in banks
    ]
    serial_weights = torch.stack([result[0] for result in serial_results])
    serial_states = torch.stack([result[1] for result in serial_results])
    torch.cuda.synchronize()
    assert torch.equal(parallel_weights, serial_weights)
    assert torch.equal(parallel_states, serial_states)


def test_qvq_v4_banked_block_ldlq_native_matches_cpu_selection_and_oracle():
    """Native bank-loss reuse must preserve mixed selectors and bank-0 rollback."""

    generator = torch.Generator(device="cpu").manual_seed(20260822)
    cpu_weight = torch.randn((32, 32), generator=generator, dtype=torch.float32)
    hessian_source = torch.randn((32, 32), generator=generator, dtype=torch.float32)
    cpu_hessian = hessian_source @ hessian_source.T + torch.eye(32) * 0.5
    cpu_stack = torch.randn((4, 1 << 16, 4), generator=generator, dtype=torch.float32).contiguous()
    cpu_banks = tuple(cpu_stack[bank] for bank in range(4))
    cpu_result = block_ldlq_inner_banked(
        cpu_weight,
        cpu_hessian,
        cpu_banks,
        bits=2.0,
        tile_rows=16,
        tile_cols=16,
        trellis_batch_size=1,
        return_bank0_oracle=True,
    )
    cuda_stack = cpu_stack.cuda()
    cuda_result = block_ldlq_inner_banked(
        cpu_weight.cuda(),
        cpu_hessian.cuda(),
        tuple(cuda_stack[bank] for bank in range(4)),
        bank_codebook_stack=cuda_stack,
        bits=2.0,
        tile_rows=16,
        tile_cols=16,
        trellis_batch_size=1,
        return_bank0_oracle=True,
    )
    torch.cuda.synchronize()
    for cpu_value, cuda_value in zip(cpu_result, cuda_result, strict=True):
        if cpu_value.dtype in (torch.float16, torch.float32, torch.float64):
            torch.testing.assert_close(cuda_value.cpu(), cpu_value, rtol=0, atol=0)
        else:
            assert torch.equal(cuda_value.cpu(), cpu_value)


@pytest.mark.parametrize("bits", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
@pytest.mark.parametrize("trellis_batch_size", [1, 3])
@pytest.mark.parametrize("viterbi_objective", ["euclidean", "hessian_diagonal"])
def test_qvq_propagated_banked_candidates_nontrivial_hessian_rate_objective_parity(
    bits, trellis_batch_size, viterbi_objective
):
    generator = torch.Generator(device="cpu").manual_seed(20260819 + int(bits * 10))
    inner = torch.randn((32, 64), generator=generator, dtype=torch.float32).cuda()
    hessian_source = torch.randn((32, 32), generator=generator, dtype=torch.float32).cuda()
    hessian = hessian_source @ hessian_source.T + torch.eye(32, dtype=torch.float32, device="cuda") * 0.5
    bank_stack = torch.randn((4, 1 << 16, 4), generator=generator, dtype=torch.float16).cuda().contiguous()
    banks = tuple(bank_stack[bank] for bank in range(4))
    baseline = block_ldlq_inner(
        inner,
        hessian,
        banks[0],
        bits=bits,
        tile_rows=16,
        tile_cols=16,
        trellis_batch_size=trellis_batch_size,
        viterbi_objective=viterbi_objective,
        telemetry=QVQQuantizationTelemetry(),
    )
    native_telemetry = QVQQuantizationTelemetry()
    native_weights, native_states = block_ldlq_inner_banked_candidates(
        inner,
        hessian,
        banks,
        bits=bits,
        tile_rows=16,
        tile_cols=16,
        bank_codebook_stack=bank_stack,
        trellis_batch_size=trellis_batch_size,
        viterbi_objective=viterbi_objective,
        telemetry=native_telemetry,
        baseline_weight=baseline[0],
        baseline_states=baseline[1],
    )
    serial_results = [baseline] + [
        block_ldlq_inner(
            inner,
            hessian,
            bank,
            bits=bits,
            tile_rows=16,
            tile_cols=16,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=viterbi_objective,
            telemetry=QVQQuantizationTelemetry(),
        )
        for bank in banks[1:]
    ]
    serial_weights = torch.stack([result[0] for result in serial_results])
    serial_states = torch.stack([result[1] for result in serial_results])
    torch.cuda.synchronize()
    assert native_telemetry.counters["native_viterbi_launches"] > 0
    assert torch.equal(native_weights, serial_weights)
    assert torch.equal(native_states, serial_states)


def _nontrivial_yaqa_fixture(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = (torch.randn((16, 16), generator=generator) * 0.05).cuda()
    input_samples = torch.randn((41, 16), generator=generator).cuda()
    output_samples = torch.randn((37, 16), generator=generator).cuda()
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0] + torch.eye(16, device="cuda") * 0.1
    output_hessian = (
        output_samples.T @ output_samples / output_samples.shape[0] + torch.eye(16, device="cuda") * 0.1
    )
    return weight, input_hessian, output_hessian


def test_qvq_cuda_production_yaqa_dispatch_enables_exact_incremental_feedback():
    weight, input_hessian, output_hessian = _nontrivial_yaqa_fixture(20260879)
    with patch("gptqmodel.quantization.qvq.yaqa_inner", wraps=yaqa_inner) as wrapped:
        quantize_qvq_linear(
            weight,
            input_hessian,
            bits=2,
            output_hessian=output_hessian,
            rounding="yaqa",
            trellis_batch_size=1,
        )

    assert wrapped.call_count == 1
    assert wrapped.call_args.kwargs["_incremental_cuda_feedback"] is True


@pytest.mark.parametrize("bits", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
def test_qvq_cuda_incremental_yaqa_feedback_is_bit_exact_for_canonical_b2_and_b4(bits):
    generator = torch.Generator(device="cpu").manual_seed(20260877 + int(bits * 10))
    weight = (torch.randn((32, 32), generator=generator) * 0.05).cuda()
    input_samples = torch.randn((47, 32), generator=generator).cuda()
    output_samples = torch.randn((43, 32), generator=generator).cuda()
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0] + torch.eye(32, device="cuda") * 0.1
    output_hessian = (
        output_samples.T @ output_samples / output_samples.shape[0] + torch.eye(32, device="cuda") * 0.1
    )
    canonical_codebook = _canonical_qvq_codebook(
        device=weight.device,
        vector_size=2,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float32,
    )
    pair_stack = _canonical_qvq_v2b2_pair_stacks(
        device=weight.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=canonical_codebook.dtype,
    )[0]
    v2b4_stack = _canonical_qvq_v2b4_bank_stack(
        device=weight.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=canonical_codebook.dtype,
    )
    common = {
        "bits": bits,
        "trellis_batch_size": 1,
        "_defer_segmented_cuda_checks": True,
    }
    formats = (
        (canonical_codebook, {}),
        (
            pair_stack[0],
            {
                "bank_codebooks": tuple(pair_stack[bank] for bank in range(2)),
                "segmented_bank_stack": pair_stack,
                "v2b2_p32": True,
            },
        ),
        (
            v2b4_stack[0],
            {
                "bank_codebooks": tuple(v2b4_stack[bank] for bank in range(4)),
                "segmented_bank_stack": v2b4_stack,
                "v2b4_p64": True,
            },
        ),
    )

    for codebook, format_kwargs in formats:
        reference = yaqa_inner(
            weight,
            input_hessian,
            output_hessian,
            codebook,
            **common,
            **format_kwargs,
        )
        incremental = yaqa_inner(
            weight,
            input_hessian,
            output_hessian,
            codebook,
            _incremental_cuda_feedback=True,
            **common,
            **format_kwargs,
        )
        assert len(incremental) == len(reference)
        assert all(torch.equal(actual, expected) for actual, expected in zip(incremental, reference, strict=True))


@pytest.mark.parametrize("shape", ((32, 48), (48, 32)))
@pytest.mark.parametrize("bits", (1.0, 2.5, 3.5))
def test_qvq_cuda_factored_incremental_yaqa_feedback_is_exact_for_rectangular_b2(shape, bits):
    """Factored anti-diagonal GEMMs must preserve the complete rectangular B2 artifact."""

    in_features, out_features = shape
    generator = torch.Generator(device="cpu").manual_seed(20260822 + in_features + int(bits * 10))
    weight = (torch.randn(shape, generator=generator) * 0.05).cuda()
    input_samples = torch.randn((47, in_features), generator=generator).cuda()
    output_samples = torch.randn((43, out_features), generator=generator).cuda()
    input_hessian = (
        input_samples.T @ input_samples / input_samples.shape[0]
        + torch.eye(in_features, device="cuda") * 0.1
    )
    output_hessian = (
        output_samples.T @ output_samples / output_samples.shape[0]
        + torch.eye(out_features, device="cuda") * 0.1
    )
    pair_stack = _canonical_qvq_v2b2_pair_stacks(
        device=weight.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float32,
    )[1]
    kwargs = {
        "bits": bits,
        "trellis_batch_size": 1,
        "bank_codebooks": tuple(pair_stack),
        "segmented_bank_stack": pair_stack,
        "v2b2_p32": True,
        "_defer_segmented_cuda_checks": True,
    }
    reference = yaqa_inner(weight, input_hessian, output_hessian, pair_stack[0], **kwargs)
    actual = yaqa_inner(
        weight,
        input_hessian,
        output_hessian,
        pair_stack[0],
        _incremental_cuda_feedback=True,
        **kwargs,
    )
    assert all(torch.equal(candidate, expected) for candidate, expected in zip(actual, reference, strict=True))
    factored_incremental = yaqa_inner(
        weight,
        input_hessian,
        output_hessian,
        pair_stack[0],
        _incremental_cuda_factored_feedback=True,
        **kwargs,
    )
    assert all(
        torch.equal(candidate, expected)
        for candidate, expected in zip(factored_incremental, reference, strict=True)
    )


@pytest.mark.parametrize("with_bias", (False, True))
def test_qvq_cuda_factored_yaqa_feedback_matches_fp32_reference_on_nondefault_stream(with_bias):
    """The grouped anti-diagonal contraction stays within the quantization FP32 gate."""

    generator = torch.Generator(device="cuda").manual_seed(20260824 + with_bias)
    source = torch.randn((32, 64), generator=generator, device="cuda", dtype=torch.float32) * 0.05
    left = torch.randn_like(source, generator=generator) * 0.01
    right = torch.randn_like(source, generator=generator) * 0.01
    output_feedback = torch.tril(
        torch.randn((64, 64), generator=generator, device="cuda", dtype=torch.float32) * 0.01,
        diagonal=-1,
    )
    bias = torch.randn_like(source, generator=generator) * 0.001 if with_bias else None
    expected = []
    for input_block, output_block in ((0, 3), (1, 2)):
        input_start = input_block * 16
        output_start = output_block * 16
        tile = source[input_start : input_start + 16, output_start : output_start + 16]
        if bias is not None:
            tile = tile + bias[input_start : input_start + 16, output_start : output_start + 16]
        expected.append(
            tile
            + left[input_start : input_start + 16, output_start:]
            @ output_feedback[output_start:, output_start : output_start + 16]
            + left[input_start : input_start + 16, output_start : output_start + 16]
            + right[input_start : input_start + 16, output_start : output_start + 16]
        )
    expected = torch.stack(expected)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = _qvq_cuda_yaqa_feedback_op()(source, left, right, output_feedback, 0, 3, 2, bias)
        repeated = _qvq_cuda_yaqa_feedback_op()(source, left, right, output_feedback, 0, 3, 2, bias)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual, repeated)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_qvq_cuda_factored_yaqa_feedback_rejects_invalid_geometry():
    source = torch.zeros((32, 32), device="cuda", dtype=torch.float32)
    output_feedback = torch.zeros((32, 32), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="anti-diagonal geometry"):
        _qvq_cuda_yaqa_feedback_op()(source, source, source, output_feedback, 0, 0, 2, None)


def test_qvq_cuda_factored_yaqa_cache_update_matches_fp32_reference_on_nondefault_stream():
    generator = torch.Generator(device="cuda").manual_seed(20260825)
    left = torch.randn((32, 64), generator=generator, device="cuda", dtype=torch.float32) * 0.01
    right = torch.randn_like(left, generator=generator) * 0.01
    input_feedback = torch.randn((32, 32), generator=generator, device="cuda") * 0.01
    output_feedback = torch.randn((64, 64), generator=generator, device="cuda") * 0.01
    reconstructed = torch.randn((2, 16, 16), generator=generator, device="cuda") * 0.05
    expected_left = left.clone()
    expected_right = right.clone()
    for tile, (input_block, output_block) in enumerate(((0, 3), (1, 2))):
        input_start = input_block * 16
        output_start = output_block * 16
        expected_left[:, output_start : output_start + 16] -= (
            input_feedback[input_start : input_start + 16].T @ reconstructed[tile]
        )
        expected_right[input_start : input_start + 16] -= (
            reconstructed[tile] @ output_feedback[output_start : output_start + 16]
        )

    actual_left = left.clone()
    actual_right = right.clone()
    repeated_left = left.clone()
    repeated_right = right.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _qvq_cuda_yaqa_feedback_update_op()(
            actual_left, actual_right, input_feedback, output_feedback, reconstructed, 0, 3, 2
        )
        _qvq_cuda_yaqa_feedback_update_op()(
            repeated_left, repeated_right, input_feedback, output_feedback, reconstructed, 0, 3, 2
        )
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual_left, repeated_left)
    assert torch.equal(actual_right, repeated_right)
    torch.testing.assert_close(actual_left, expected_left, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(actual_right, expected_right, rtol=0.0, atol=1e-6)


def test_qvq_cuda_factored_yaqa_family_batch_matches_independent_candidates():
    """A leading candidate dimension must not couple independent YAQA histories."""

    generator = torch.Generator(device="cuda").manual_seed(20260826)
    families = 4
    source = torch.randn((32, 64), generator=generator, device="cuda") * 0.05
    left = torch.randn((families, 32, 64), generator=generator, device="cuda") * 0.01
    right = torch.randn_like(left, generator=generator) * 0.01
    input_feedback = torch.randn((32, 32), generator=generator, device="cuda") * 0.01
    output_feedback = torch.randn((64, 64), generator=generator, device="cuda") * 0.01
    reconstructed = torch.randn((families, 2, 16, 16), generator=generator, device="cuda") * 0.05

    expected_tiles = torch.stack(
        tuple(
            _qvq_cuda_yaqa_feedback_op()(
                source, left[family], right[family], output_feedback, 0, 3, 2, None
            )
            for family in range(families)
        )
    )
    expected_left = left.clone()
    expected_right = right.clone()
    for family in range(families):
        _qvq_cuda_yaqa_feedback_update_op()(
            expected_left[family],
            expected_right[family],
            input_feedback,
            output_feedback,
            reconstructed[family],
            0,
            3,
            2,
        )

    actual_left = left.clone()
    actual_right = right.clone()
    actual_tiles = _qvq_cuda_yaqa_feedback_op()(source, left, right, output_feedback, 0, 3, 2, None)
    _qvq_cuda_yaqa_feedback_update_op()(
        actual_left, actual_right, input_feedback, output_feedback, reconstructed, 0, 3, 2
    )

    assert torch.equal(actual_tiles, expected_tiles)
    assert torch.equal(actual_left, expected_left)
    assert torch.equal(actual_right, expected_right)


def test_qvq_cuda_factored_yaqa_cache_update_rejects_invalid_geometry():
    left = torch.zeros((32, 32), device="cuda", dtype=torch.float32)
    feedback = torch.zeros_like(left)
    reconstructed = torch.zeros((2, 16, 16), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="update anti-diagonal geometry"):
        _qvq_cuda_yaqa_feedback_update_op()(
            left, left.clone(), feedback, feedback, reconstructed, 0, 0, 2
        )


def test_qvq_cuda_b2_yaqa_candidate_batch_is_bit_exact():
    weight, input_hessian, output_hessian = _nontrivial_yaqa_fixture(20260823)
    bits = 2.5
    banks = _canonical_qvq_v2b4_banks(
        device=weight.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float32,
    )
    pairs = _canonical_qvq_v2b2_pair_stacks(
        device=weight.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float32,
    )
    kwargs = {
        "bits": bits,
        "block_family_id": 2,
        "family_mode": "reselect",
        "bank_codebook_pair_stacks": pairs,
        "trellis_batch_size": 1,
        "_incremental_cuda_factored_feedback": True,
    }
    reference = yaqa_inner_v2b2_p32(
        weight,
        input_hessian,
        output_hessian,
        banks,
        _parallel_candidates=False,
        **kwargs,
    )
    actual = yaqa_inner_v2b2_p32(
        weight,
        input_hessian,
        output_hessian,
        banks,
        _parallel_candidates=True,
        **kwargs,
    )
    assert all(torch.equal(candidate, expected) for candidate, expected in zip(actual, reference, strict=True))


@pytest.mark.parametrize("bits", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
@pytest.mark.parametrize("objective", ("euclidean", "hessian_diagonal"))
def test_qvq_cuda_b2_block_ldl_family_batch_matches_serial(bits, objective):
    generator = torch.Generator(device="cuda").manual_seed(20260827 + int(bits * 2))
    weight = (torch.randn((32, 64), generator=generator, device="cuda") * 0.05).to(torch.float16)
    samples = torch.randn((47, 32), generator=generator, device="cuda")
    hessian = samples.T @ samples / samples.shape[0]
    hessian.diagonal().add_(0.1)
    banks = _canonical_qvq_v2b4_banks(
        device=weight.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float16,
    )
    pair_stacks = _canonical_qvq_v2b2_pair_stacks(
        device=weight.device,
        bits=bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float16,
    )
    common = {
        "bits": bits,
        "trellis_batch_size": 3,
        "viterbi_objective": objective,
        "bank_codebook_pair_stacks": pair_stacks,
    }
    expected = block_ldlq_inner_v2b2_p32(weight, hessian, banks, _family_batch=False, **common)
    actual = block_ldlq_inner_v2b2_p32(weight, hessian, banks, _family_batch=True, **common)
    assert all(torch.equal(candidate, reference) for candidate, reference in zip(actual, expected, strict=True))


@pytest.mark.parametrize("bits", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
def test_qvq_v2b2_p32_yaqa_fixed_and_reselected_are_exact_and_baseline_safe(bits):
    weight, input_hessian, output_hessian = _nontrivial_yaqa_fixture(20260830 + int(bits * 10))
    common = {
        "bits": bits,
        "output_hessian": output_hessian,
        "rounding": "yaqa",
        "trellis_batch_size": 1,
    }
    canonical = quantize_qvq_linear(weight, input_hessian, **common)
    fixed = quantize_qvq_linear(
        weight,
        input_hessian,
        bank_count=2,
        v2b2_p32=True,
        yaqa_v2b2_family_mode="fixed_block_ldlq",
        telemetry=QVQQuantizationTelemetry(),
        **common,
    )
    reselected = quantize_qvq_linear(
        weight,
        input_hessian,
        bank_count=2,
        v2b2_p32=True,
        yaqa_v2b2_family_mode="reselect",
        telemetry=QVQQuantizationTelemetry(),
        **common,
    )
    assert fixed.kronecker_proxy_loss <= canonical.kronecker_proxy_loss
    assert reselected.kronecker_proxy_loss <= fixed.kronecker_proxy_loss
    for result in (fixed, reselected):
        tensors = result.serialized_tensors()
        decoded = reconstruct_qvq_inner_weight(
            result.trellis,
            bits=bits,
            in_features=16,
            out_features=16,
            bank_ids=tensors["bank_ids"],
            bank_alt_id=tensors["bank_alt_id"],
            v2b2_p32=True,
        )
        assert torch.equal(decoded, result.inner_weight)
        assert result.telemetry["counters"]["yaqa_segmented_v2_chunks"] >= 1


def test_qvq_v2b2_family_batch_telemetry_reports_phase1_reuse_and_consumption():
    generator = torch.Generator(device="cuda").manual_seed(20260823)
    weight = torch.randn((32, 32), generator=generator, device="cuda", dtype=torch.float16)
    hessian = torch.eye(32, device="cuda", dtype=torch.float32)
    pair_stacks = torch.stack(
        _canonical_qvq_v2b2_pair_stacks(
            device=weight.device,
            bits=2.0,
            codebook_version=PGC16_CODEBOOK_VERSION,
            dtype=torch.float16,
        )
    ).contiguous()
    telemetry = QVQQuantizationTelemetry()

    _yaqa_inner_v2b2_family_batch_cuda(
        weight,
        hessian,
        hessian,
        pair_stacks,
        bits=2.0,
        factorization=None,
        rounding_bias=None,
        telemetry=telemetry,
    )
    counters = telemetry.finalize()["counters"]

    assert counters["viterbi_family_grid_calls"] > 0
    assert counters["viterbi_family_state_steps"] > 0
    assert counters["viterbi_exact_reuse_candidates"] == 0
    assert counters["viterbi_reselection_revisits"] == 0
    assert counters["viterbi_logical_solve_ids"] == counters["viterbi_unique_logical_solve_ids"]
    assert counters["viterbi_provisional_states_produced"] == (
        counters["viterbi_provisional_states_consumed"] * 128
    )


def test_qvq_v2b2_p32_yaqa_reselection_uses_non_default_producer_stream_safely():
    weight, input_hessian, output_hessian = _nontrivial_yaqa_fixture(20260876)
    kwargs = {
        "bits": 2.5,
        "output_hessian": output_hessian,
        "rounding": "yaqa",
        "trellis_batch_size": 1,
        "bank_count": 2,
        "v2b2_p32": True,
        "yaqa_v2b2_family_mode": "reselect",
    }
    expected = quantize_qvq_linear(weight, input_hessian, **kwargs)
    torch.cuda.synchronize()

    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        actual = quantize_qvq_linear(weight, input_hessian, **kwargs)
        completion = torch.cuda.Event()
        completion.record(producer)
    completion.synchronize()

    assert torch.equal(actual.inner_weight, expected.inner_weight)
    assert torch.equal(actual.trellis, expected.trellis)
    assert torch.equal(actual.bank_ids, expected.bank_ids)
    assert torch.equal(actual.bank_alt_id, expected.bank_alt_id)
    assert actual.kronecker_proxy_loss == expected.kronecker_proxy_loss


@pytest.mark.parametrize("bits", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
def test_qvq_v2b4_p64_yaqa_is_exact_and_independent_v2_yaqa_safe(bits):
    weight, input_hessian, output_hessian = _nontrivial_yaqa_fixture(20260840 + int(bits * 10))
    common = {
        "bits": bits,
        "output_hessian": output_hessian,
        "rounding": "yaqa",
        "trellis_batch_size": 1,
    }
    canonical = quantize_qvq_linear(weight, input_hessian, **common)
    banked = quantize_qvq_linear(
        weight,
        input_hessian,
        bank_count=4,
        v2b4_p64=True,
        telemetry=QVQQuantizationTelemetry(),
        **common,
    )
    assert banked.kronecker_proxy_loss <= canonical.kronecker_proxy_loss
    tensors = banked.serialized_tensors()
    decoded = reconstruct_qvq_inner_weight(
        banked.trellis,
        bits=bits,
        in_features=16,
        out_features=16,
        bank_ids=tensors["bank_ids"],
        v2b4_p64=True,
    )
    assert torch.equal(decoded, banked.inner_weight)
    assert banked.telemetry["counters"]["yaqa_segmented_v2_chunks"] == 1


@pytest.mark.parametrize("format_name", ["v2b2-p32", "v2b4-p64"])
def test_qvq_banked_v2_yaqa_multitile_partial_batch_is_exact_and_safe(format_name):
    generator = torch.Generator(device="cpu").manual_seed(20260850)
    weight = (torch.randn((64, 32), generator=generator) * 0.05).cuda()
    input_samples = torch.randn((53, 32), generator=generator).cuda()
    output_samples = torch.randn((47, 64), generator=generator).cuda()
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0] + torch.eye(32, device="cuda") * 0.1
    output_hessian = (
        output_samples.T @ output_samples / output_samples.shape[0] + torch.eye(64, device="cuda") * 0.1
    )
    common = {
        "bits": 2.0,
        "output_hessian": output_hessian,
        "rounding": "yaqa",
        "trellis_batch_size": 3,
    }
    canonical = quantize_qvq_linear(weight, input_hessian, **common)
    format_kwargs = (
        {
                "bank_count": 2,
                "v2b2_p32": True,
                "yaqa_v2b2_family_mode": "fixed_block_ldlq",
            }
        if format_name == "v2b2-p32"
        else {"bank_count": 4, "v2b4_p64": True}
    )
    banked = quantize_qvq_linear(
        weight,
        input_hessian,
        telemetry=QVQQuantizationTelemetry(),
        **common,
        **format_kwargs,
    )
    assert banked.kronecker_proxy_loss <= canonical.kronecker_proxy_loss
    tensors = banked.serialized_tensors()
    decoded = reconstruct_qvq_inner_weight(
        banked.trellis,
        bits=2.0,
        in_features=32,
        out_features=64,
        bank_ids=tensors["bank_ids"],
        bank_alt_id=tensors.get("bank_alt_id"),
        v2b2_p32=format_name == "v2b2-p32",
        v2b4_p64=format_name == "v2b4-p64",
    )
    assert torch.equal(decoded, banked.inner_weight)
    expected_selectors = 8 * (8 if format_name == "v2b2-p32" else 4)
    assert banked.bank_ids.numel() == expected_selectors
    assert banked.telemetry["counters"]["yaqa_segmented_v2_chunks"] == 5


def test_qvq_cuda_empty_batch_and_contract_guards():
    x = torch.zeros((0, 16), device="cuda", dtype=torch.float16)
    trellis = torch.zeros((1, 16), device="cuda", dtype=torch.int32)
    assert qvq_cuda_gemv(x, trellis, 2, out_features=16).shape == (0, 16)
    assert qvq_cuda_gemv(x, trellis, 2, out_features=16, output_fp32=True).dtype == torch.float32
    empty_hadamard = qvq_cuda_hadamard(torch.empty((0, 16), device="cuda", dtype=torch.float16))
    assert empty_hadamard.shape == (0, 16)
    assert empty_hadamard.dtype == torch.float16
    with pytest.raises(ValueError, match="pgc16-v1"):
        qvq_cuda_gemv(
            x,
            trellis,
            2,
            out_features=16,
            codebook_version="unsupported",
        )

    cases = [
        ((x, trellis, 9), {"out_features": 16}, ValueError, "rate"),
        ((x, trellis, True), {"out_features": 16}, TypeError, "rate"),
        ((x, trellis, 2), {"out_features": True}, TypeError, "integer"),
        ((x, trellis, 2), {"out_features": 16, "output_fp32": 1}, TypeError, "boolean"),
        ((x[:, :, None], trellis, 2), {"out_features": 16}, ValueError, "2D"),
        ((x.cpu(), trellis, 2), {"out_features": 16}, ValueError, "CUDA device"),
        (
            (x.to(torch.float32), trellis, 2),
            {"out_features": 16},
            TypeError,
            "float16 or bfloat16",
        ),
        ((x, trellis.to(torch.int16), 2), {"out_features": 16}, TypeError, "int32"),
        (
            (torch.zeros((16, 2), device="cuda", dtype=torch.float16).T, trellis, 2),
            {"out_features": 16},
            ValueError,
            "contiguous",
        ),
        (
            (torch.zeros((1, 15), device="cuda", dtype=torch.float16), trellis, 2),
            {"out_features": 16},
            ValueError,
            "divisible",
        ),
        ((x, trellis[:, :8], 2), {"out_features": 16}, ValueError, "trellis"),
    ]
    for args, kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            qvq_cuda_gemv(*args, **kwargs)

    banked_trellis = torch.zeros((1, 8), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match=r"selectors must be in \[0, 3\]"):
        qvq_cuda_gemv(
            x,
            banked_trellis,
            1,
            out_features=16,
            vector_size=4,
            bank_ids=torch.tensor([4], device="cuda", dtype=torch.uint8),
        )

    with pytest.raises(TypeError, match="mode 2 requires float16"):
        qvq_cuda_hadamard(torch.ones((1, 16), device="cuda"), scale_mode=2)


def test_qvq_output_alignment_preserves_range_before_fp16_rounding():
    n = 2048
    layer = _FixedTrellisAlignmentLinear(
        inner_weight=torch.eye(n, device="cuda", dtype=torch.float32) * 70000.0,
        SU=torch.ones(n, device="cuda"),
        SV=torch.full((n,), 0.001, device="cuda"),
        bias=None,
        output_dtype=torch.float16,
    ).eval()
    inputs = torch.zeros((1, n), device="cuda", dtype=torch.float16)
    inputs[0, 0] = 1
    with torch.no_grad():
        output = layer(inputs)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert output.abs().max() == 70


def test_qvq_cuda_bfloat16_recovery_uses_completed_output_finiteness():
    tensors = {
        "trellis": torch.zeros((1, 16), device="cuda", dtype=torch.int32),
        "SU": torch.ones(16, device="cuda"),
        "SV": torch.ones(16, device="cuda"),
    }
    layer = QVQLinear(bits=2, in_features=16, out_features=16, tensors=tensors).eval()
    calls = []

    def fake_compute(inputs, compute_dtype):
        calls.append((inputs.dtype, compute_dtype))
        return torch.full((inputs.shape[0], 16), float("inf"), device="cuda") if len(calls) == 1 else torch.ones(
            (inputs.shape[0], 16), device="cuda"
        )

    with patch.object(layer, "_forward_compute_dtype", side_effect=fake_compute):
        output = layer(torch.ones((1, 16), device="cuda", dtype=torch.bfloat16))
    assert len(calls) == 2
    assert torch.isfinite(output).all()


def test_qvq_cuda_rejects_unsupported_compute_capability_before_launch():
    x = torch.zeros((1, 16), device="cuda", dtype=torch.float16)
    trellis = torch.zeros((1, 16), device="cuda", dtype=torch.int32)
    with (
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        pytest.raises(RuntimeError, match="8.0"),
    ):
        qvq_cuda_gemv(x, trellis, 2, out_features=16)


def test_qvq_cuda_viterbi_contract_guards():
    sequences = torch.zeros((2, 3, 2), device="cuda", dtype=torch.float32)
    codebook = torch.zeros((1 << 16, 2), device="cuda", dtype=torch.float32)
    overlap = torch.zeros(2, device="cuda", dtype=torch.int64)
    step_weights = torch.ones((2, 3), device="cuda", dtype=torch.float32)
    negative_step_weights = step_weights.clone()
    negative_step_weights[0, 0] = -1
    nonfinite_step_weights = step_weights.clone()
    nonfinite_step_weights[0, 0] = torch.nan
    extreme_sequences = torch.full_like(sequences, 1.0e20)
    cases = [
        ((sequences, codebook, True), {}, TypeError, "rate"),
        ((sequences, codebook, 9), {}, ValueError, "rate"),
        ((sequences[..., 0], codebook, 4), {}, ValueError, "shape"),
        ((sequences, codebook[:10], 4), {}, ValueError, "65536"),
        ((sequences.cpu(), codebook, 4), {}, ValueError, "one CUDA device"),
        ((sequences.to(torch.float16), codebook, 4), {}, TypeError, "float32"),
        ((extreme_sequences, codebook, 4), {}, ValueError, "magnitudes"),
        (
            (_noncontiguous_viterbi_tensor(sequences), codebook, 4),
            {},
            ValueError,
            "contiguous",
        ),
        (
            (sequences, codebook, 4, overlap.to(torch.int32)),
            {},
            ValueError,
            "contiguous int64",
        ),
        ((sequences, codebook, 4, overlap[:1]), {}, ValueError, "shape"),
        ((sequences[:, :0], codebook, 4), {}, RuntimeError, "at least one"),
        (
            (sequences, codebook, 4),
            {"step_weights": step_weights[:1]},
            ValueError,
            "shape",
        ),
        (
            (sequences, codebook, 4),
            {"step_weights": step_weights.cpu()},
            ValueError,
            "sequence device",
        ),
        (
            (sequences, codebook, 4),
            {"step_weights": step_weights.half()},
            ValueError,
            "float32",
        ),
        (
            (sequences, codebook, 4),
            {"step_weights": _noncontiguous_viterbi_weights(step_weights)},
            ValueError,
            "contiguous",
        ),
        (
            (sequences, codebook, 4),
            {"step_weights": negative_step_weights},
            ValueError,
            "nonnegative",
        ),
        (
            (sequences, codebook, 4),
            {"step_weights": nonfinite_step_weights},
            ValueError,
            "finite",
        ),
    ]
    for args, kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            qvq_cuda_viterbi(*args, **kwargs)


@pytest.mark.parametrize("bits", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
def test_qvq_cuda_trusted_v2_viterbi_is_bit_exact_after_yaqa_style_prevalidation(bits):
    generator = torch.Generator(device="cpu").manual_seed(20260817 + int(bits * 10))
    sequences = torch.randn((3, 128, 2), generator=generator, dtype=torch.float32).cuda().contiguous()
    codebook = pgc16_codebook_v2_bank(
        0,
        bits=bits,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()
    transition_bits = qvq_transition_bits(bits, vector_size=2)

    public_states, public_loss = qvq_cuda_viterbi(sequences, codebook, bits)
    trusted_states, trusted_loss = _qvq_cuda_viterbi_trusted(sequences, codebook, bits)
    overlap = (public_states[:, -1] & ((1 << (16 - transition_bits)) - 1)).contiguous()
    public_constrained = qvq_cuda_viterbi(sequences, codebook, bits, overlap)
    trusted_constrained = _qvq_cuda_viterbi_trusted(sequences, codebook, bits, overlap)

    assert torch.equal(trusted_states, public_states)
    assert torch.equal(trusted_loss, public_loss)
    assert torch.equal(trusted_constrained[0], public_constrained[0])
    assert torch.equal(trusted_constrained[1], public_constrained[1])


def _noncontiguous_viterbi_tensor(values: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(
        (*values.shape[:-1], values.shape[-1] * 2),
        device=values.device,
        dtype=values.dtype,
    )
    view = storage[..., ::2]
    view.copy_(values)
    assert not view.is_contiguous()
    return view


def _noncontiguous_viterbi_weights(values: torch.Tensor) -> torch.Tensor:
    storage = torch.empty((values.shape[0], values.shape[1] * 2), device=values.device, dtype=values.dtype)
    view = storage[:, ::2]
    view.copy_(values)
    assert not view.is_contiguous()
    return view


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
@pytest.mark.parametrize("constrained", (False, True))
def test_qvq_cuda_weighted_viterbi_matches_eager_reference(bits, constrained):
    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(20260920 + 10 * transition_bits + constrained)
    sequences = torch.randn((2, 7, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = torch.randn((1 << 16, 2), generator=generator, dtype=torch.float32).cuda()
    step_weights = (0.05 + 2.0 * torch.rand((2, 7), generator=generator, dtype=torch.float32)).cuda()
    overlap = None
    if constrained:
        overlap_bits = 16 - transition_bits
        overlap_limit = 1 << overlap_bits
        overlap = torch.randint(0, overlap_limit, (2,), generator=generator, dtype=torch.int64).cuda()

    actual_states, actual_error = qvq_cuda_viterbi(
        sequences,
        codebook,
        bits,
        overlap,
        step_weights,
    )
    dispatched = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=bits,
        overlap=overlap,
        step_weights=step_weights,
    )
    reference = batched_viterbi_quantize(
        _noncontiguous_viterbi_tensor(sequences),
        codebook,
        bits=bits,
        overlap=overlap,
        step_weights=step_weights,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_states, reference.states)
    torch.testing.assert_close(actual_error, reference.squared_error, rtol=2e-5, atol=2e-5)
    assert torch.equal(dispatched.states, actual_states)
    assert torch.equal(dispatched.squared_error, actual_error)


@pytest.mark.parametrize("seed", (2026081201, 2026081202))
@pytest.mark.parametrize("constrained", (False, True))
def test_qvq_cuda_w8_scalar_recurrence_matches_long_pgc16_eager_paths(seed, constrained):
    """Exercise every production-codebook tile and step in the W8 specialization."""

    generator = torch.Generator().manual_seed(seed)
    sequences = torch.randn((7, 128, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = pgc16_codebook(device="cuda", dtype=torch.float32)
    step_weights = (0.05 + 2.0 * torch.rand((7, 128), generator=generator, dtype=torch.float32)).cuda()
    overlap = torch.zeros(7, device="cuda", dtype=torch.int64) if constrained else None

    actual_states, actual_error = qvq_cuda_viterbi(
        sequences,
        codebook,
        8,
        overlap,
        step_weights,
    )
    reference = batched_viterbi_quantize(
        _noncontiguous_viterbi_tensor(sequences),
        codebook,
        bits=8,
        overlap=overlap,
        step_weights=step_weights,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_states, reference.states)
    torch.testing.assert_close(actual_error, reference.squared_error, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("bits", (2.5, 3.5, 4.5, 5.5, 6.5, 7.5))
@pytest.mark.parametrize("constrained", (False, True))
def test_qvq_cuda_fused_suffix_recurrence_matches_long_pgc16_eager_paths(bits, constrained):
    """Cover every half-step fused-recurrence shape at production length."""

    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(2026081250 + transition_bits + constrained)
    sequences = torch.randn((3, 128, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = pgc16_codebook(device="cuda", dtype=torch.float32)
    step_weights = (0.05 + 2.0 * torch.rand((3, 128), generator=generator, dtype=torch.float32)).cuda()
    overlap = None
    if constrained:
        overlap = torch.randint(
            0,
            1 << (16 - transition_bits),
            (3,),
            generator=generator,
            dtype=torch.int64,
        ).cuda()

    actual_states, actual_error = qvq_cuda_viterbi(sequences, codebook, bits, overlap, step_weights)
    reference = batched_viterbi_quantize(
        _noncontiguous_viterbi_tensor(sequences),
        codebook,
        bits=bits,
        overlap=overlap,
        step_weights=step_weights,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_states, reference.states)
    torch.testing.assert_close(actual_error, reference.squared_error, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
def test_qvq_cuda_unit_step_weights_are_exactly_disabled_control(bits):
    generator = torch.Generator().manual_seed(20261000 + qvq_transition_bits(bits))
    sequences = torch.randn((3, 11, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = torch.randn((1 << 16, 2), generator=generator, dtype=torch.float32).cuda()
    unit_weights = torch.ones((3, 11), device="cuda", dtype=torch.float32)

    control = qvq_cuda_viterbi(sequences, codebook, bits)
    weighted = qvq_cuda_viterbi(sequences, codebook, bits, step_weights=unit_weights)
    torch.cuda.synchronize()

    assert torch.equal(control[0], weighted[0])
    assert torch.equal(control[1], weighted[1])


def test_qvq_accuracy_upgrade_helpers_reject_mixed_cpu_cuda_devices():
    weight = torch.eye(16)
    reconstructed = torch.eye(16, device="cuda")
    with pytest.raises(ValueError, match="share one device"):
        qvq_proxy_loss(weight, reconstructed, torch.eye(16))

    with pytest.raises(ValueError, match="share one device"):
        optimize_qvq_output_channel_scales(
            weight,
            torch.eye(16, device="cuda"),
            torch.eye(16),
            torch.ones(16, device="cuda"),
            torch.ones(16, device="cuda"),
        )

    with pytest.raises(ValueError, match="acceptance Hessian device"):
        optimize_qvq_output_channel_scales(
            reconstructed,
            torch.eye(16, device="cuda"),
            torch.eye(16, device="cuda"),
            torch.ones(16, device="cuda"),
            torch.ones(16, device="cuda"),
            optimization_H=torch.eye(16),
        )

    with pytest.raises(ValueError, match="share the sequence device"):
        batched_viterbi_quantize(
            torch.zeros((1, 3, 2), device="cuda"),
            torch.zeros((1 << 16, 2), device="cuda"),
            bits=2,
            step_weights=torch.ones((1, 3)),
        )


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
def test_qvq_cuda_viterbi_zero_ties_choose_lowest_valid_states(bits):
    sequences = torch.zeros((2, 3, 2), device="cuda", dtype=torch.float32)
    codebook = torch.zeros((1 << 16, 2), device="cuda", dtype=torch.float32)
    shift = qvq_transition_bits(bits)
    overlap_bits = 16 - shift
    maximum_overlap = (1 << overlap_bits) - 1 if overlap_bits else 0
    overlap = torch.tensor((0, maximum_overlap), device="cuda", dtype=torch.int64)

    states, squared_error = qvq_cuda_viterbi(sequences, codebook, bits, overlap)
    torch.cuda.synchronize()

    assert torch.equal(states[:, 0] >> shift, overlap)
    if overlap_bits:
        assert torch.equal(states[:, -1] & ((1 << overlap_bits) - 1), overlap)
    assert torch.equal(squared_error, torch.zeros_like(squared_error))


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
def test_qvq_cuda_viterbi_single_step_chooses_lowest_tied_state(bits):
    sequences = torch.zeros((2, 1, 2), device="cuda", dtype=torch.float32)
    codebook = torch.zeros((1 << 16, 2), device="cuda", dtype=torch.float32)

    states, squared_error = qvq_cuda_viterbi(sequences, codebook, bits)
    torch.cuda.synchronize()

    assert torch.equal(states, torch.zeros_like(states))
    assert torch.equal(squared_error, torch.zeros_like(squared_error))


def test_qvq_cuda_viterbi_rejects_unsupported_compute_capability_before_launch():
    sequences = torch.zeros((1, 1, 2), device="cuda", dtype=torch.float32)
    codebook = torch.zeros((1 << 16, 2), device="cuda", dtype=torch.float32)
    with (
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        pytest.raises(RuntimeError, match="8.0"),
    ):
        qvq_cuda_viterbi(sequences, codebook, 4)


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
def test_qvq_cuda_linear_inference_matches_portable_dense_layer(bits):
    operands, _ = _case(bits, 4, k=32, n=32)
    x, trellis = operands
    generator = torch.Generator().manual_seed(14000 + qvq_transition_bits(bits))
    tensors = {
        "trellis": trellis,
        "SU": torch.randn(32, generator=generator, dtype=torch.float16).cuda(),
        "SV": torch.randn(32, generator=generator, dtype=torch.float16).cuda(),
        "bias": torch.randn(32, generator=generator, dtype=torch.float16).cuda(),
    }
    reference_layer = QVQReferenceLinear(
        bits=bits, in_features=32, out_features=32, name="proj", tensors=tensors
    ).eval()
    cuda_layer = QVQLinear(bits=bits, in_features=32, out_features=32, name="proj", tensors=tensors).eval()

    shaped_x = x.reshape(2, 2, 32)
    reference_layer.train()
    differentiable_reference = reference_layer(shaped_x)
    fp32_oracle = reference_layer._forward_compute_dtype(
        shaped_x.float().reshape(-1, 32), torch.float32
    ).reshape_as(shaped_x)
    reference_layer.eval()
    reference = reference_layer(shaped_x)
    actual = cuda_layer(shaped_x)

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    _assert_accuracy(differentiable_reference.cpu().reshape(-1, 32), fp32_oracle.cpu().reshape(-1, 32))
    _assert_accuracy(reference.cpu().reshape(-1, 32), fp32_oracle.cpu().reshape(-1, 32))
    _assert_accuracy(actual.cpu().reshape(-1, 32), fp32_oracle.cpu().reshape(-1, 32))
    assert set(dict(cuda_layer.named_buffers())) == {"trellis", "SU", "SV", "bias"}


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
def test_qvq_cuda_linear_uses_more_accurate_fp16_compute_for_in_range_bfloat16(bits):
    operands, _ = _case(bits, 32, k=32, n=32, dtype=torch.bfloat16)
    x, trellis = operands
    tensors = {
        "trellis": trellis,
        "SU": torch.ones(32, dtype=torch.bfloat16, device="cuda"),
        "SV": torch.ones(32, dtype=torch.bfloat16, device="cuda"),
    }
    reference_layer = QVQReferenceLinear(
        bits=bits,
        in_features=32,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.bfloat16,
    ).eval()
    cuda_layer = QVQLinear(
        bits=bits,
        in_features=32,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.bfloat16,
    ).eval()
    observed = {}
    native = qvq_cuda_gemv

    def capture_dtype(inner_x, *args, **kwargs):
        observed["dtype"] = inner_x.dtype
        return native(inner_x, *args, **kwargs)

    with patch("gptqmodel.utils.qvq_cuda.qvq_cuda_gemv", side_effect=capture_dtype):
        actual = cuda_layer(x)
    reference = reference_layer._forward_compute_dtype(x.float(), torch.float32)

    assert observed["dtype"] == torch.float16
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=2e-2)
    # Two rows have FP32 margins smaller than one BF16 ULP and therefore become
    # exact BF16 ties. Keep reporting raw argmax agreement, but separately
    # require every FP32 winner to remain among the tied BF16 maxima.
    _assert_accuracy(actual.cpu(), reference.cpu(), max_kld=2e-3, min_top1=0.9375, min_top5=0.96)
    reference_winners = reference.argmax(dim=-1, keepdim=True)
    actual_winners = actual == actual.max(dim=-1, keepdim=True).values
    assert actual_winners.gather(-1, reference_winners).all()


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
@pytest.mark.parametrize("basis_index", (0, 15, 31))
def test_qvq_cuda_linear_retries_extreme_bfloat16_in_native_bfloat16(bits, basis_index):
    operands, _ = _case(bits, 1, k=32, n=32, dtype=torch.bfloat16)
    _, trellis = operands
    x = torch.zeros((1, 32), dtype=torch.bfloat16, device="cuda")
    x[0, basis_index] = 1e5
    tensors = {
        "trellis": trellis,
        "SU": torch.ones(32, dtype=torch.bfloat16, device="cuda"),
        "SV": torch.ones(32, dtype=torch.bfloat16, device="cuda"),
    }
    reference_layer = QVQReferenceLinear(
        bits=bits,
        in_features=32,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.bfloat16,
    ).eval()
    cuda_layer = QVQLinear(
        bits=bits,
        in_features=32,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.bfloat16,
    ).eval()
    observed = []
    native = qvq_cuda_gemv

    def capture_dtype(inner_x, *args, **kwargs):
        observed.append(inner_x.dtype)
        return native(inner_x, *args, **kwargs)

    with patch("gptqmodel.utils.qvq_cuda.qvq_cuda_gemv", side_effect=capture_dtype):
        actual = cuda_layer(x)
    reference = reference_layer._forward_compute_dtype(x.float(), torch.float32)

    assert observed == [torch.float16, torch.bfloat16]
    assert torch.isfinite(actual).all()
    assert actual.dtype == torch.bfloat16
    relative_l2 = ((actual.float() - reference).square().sum() / reference.square().sum()).sqrt()
    assert relative_l2 < 0.01
    # At this magnitude two distinct FP32 values can round to the same BF16
    # maximum. The FP32 winner must remain in the BF16 winner set; requiring
    # argmax's first tied index would incorrectly treat an exact tie as drift.
    reference_winner = reference.argmax(dim=-1, keepdim=True)
    assert (actual == actual.max(dim=-1, keepdim=True).values).gather(-1, reference_winner).all()


def test_qvq_cuda_linear_retries_narrow_fp16_factorization_overflow():
    """Narrow transforms retain their original BF16 fallback for adversarial inputs."""

    operands, _ = _case(1.5, 1, k=32, n=32, dtype=torch.float16)
    _, trellis = operands
    x = torch.zeros((1, 32), dtype=torch.float16, device="cuda")
    x[0, 0] = 40000
    tensors = {
        "trellis": trellis,
        "SU": torch.full((32,), 2.0, dtype=torch.float32, device="cuda"),
        "SV": torch.full((32,), 1e-4, dtype=torch.float32, device="cuda"),
    }
    reference_layer = QVQReferenceLinear(
        bits=1.5,
        in_features=32,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).eval()
    cuda_layer = QVQLinear(
        bits=1.5,
        in_features=32,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).eval()
    observed = []
    native = qvq_cuda_gemv

    def capture_dtype(inner_x, *args, **kwargs):
        observed.append(inner_x.dtype)
        return native(inner_x, *args, **kwargs)

    with patch("gptqmodel.utils.qvq_cuda.qvq_cuda_gemv", side_effect=capture_dtype):
        actual = cuda_layer(x)
    reference = reference_layer._forward_compute_dtype(x.float(), torch.float32)

    assert observed == [torch.float16, torch.bfloat16]
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=2e-2)


def test_qvq_cuda_graph_preserves_narrow_fp16_overflow_rescue():
    """Graph capture must not bypass the narrow-transform BF16 rescue."""

    operands, _ = _case(1.5, 1, k=32, n=32, dtype=torch.float16)
    _, trellis = operands
    static_x = torch.zeros((1, 32), dtype=torch.float16, device="cuda")
    static_x[0, 0] = 40000
    tensors = {
        "trellis": trellis,
        "SU": torch.full((32,), 2.0, dtype=torch.float32, device="cuda"),
        "SV": torch.full((32,), 1e-4, dtype=torch.float32, device="cuda"),
    }
    layer = QVQLinear(
        bits=1.5,
        in_features=32,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).eval()

    eager = layer(static_x).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(static_x)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(eager).all()
    assert torch.isfinite(captured).all()
    assert torch.equal(captured, eager)


def test_qvq_cuda_wide_factorization_uses_one_range_safe_fp16_decode():
    operands, _ = _case(1.5, 1, k=2048, n=32, dtype=torch.float16)
    _, trellis = operands
    x = torch.zeros((1, 2048), dtype=torch.float16, device="cuda")
    x[0, 0] = 40000
    tensors = {
        "trellis": trellis,
        "SU": torch.ones((2048,), dtype=torch.float32, device="cuda"),
        "SV": torch.full((32,), 1e-4, dtype=torch.float32, device="cuda"),
    }
    reference_layer = QVQReferenceLinear(
        bits=1.5,
        in_features=2048,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).train()
    cuda_layer = QVQLinear(
        bits=1.5,
        in_features=2048,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).eval()
    observed = []
    native = qvq_cuda_gemv

    def capture_dtype(inner_x, *args, **kwargs):
        observed.append((inner_x.dtype, kwargs["output_fp32"]))
        return native(inner_x, *args, **kwargs)

    with patch("gptqmodel.utils.qvq_cuda.qvq_cuda_gemv", side_effect=capture_dtype):
        actual = cuda_layer(x)
    reference = reference_layer._forward_compute_dtype(x.float(), torch.float32)

    assert observed == [(torch.float16, True)]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=2e-2)


def test_qvq_cuda_wide_finite_path_is_bitwise_identical_to_original_fp16_rounding():
    operands, _ = _case(2, 2, k=2048, n=2048, dtype=torch.float16)
    x, trellis = operands
    generator = torch.Generator().manual_seed(20260814)
    tensors = {
        "trellis": trellis,
        "SU": torch.randn(2048, generator=generator, dtype=torch.float16).cuda(),
        "SV": torch.randn(2048, generator=generator, dtype=torch.float16).cuda(),
        "bias": torch.randn(2048, generator=generator, dtype=torch.float16).cuda(),
    }
    layer = QVQLinear(
        bits=2,
        in_features=2048,
        out_features=2048,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).eval()

    old_input = qvq_cuda_hadamard(x, pre_scale=tensors["SU"], scale_mode=0)
    old_inner = qvq_cuda_gemv(old_input, trellis, 2, out_features=2048)
    old_output = qvq_cuda_hadamard(
        old_inner,
        post_scale=tensors["SV"],
        bias=tensors["bias"],
        scale_mode=0,
    )
    actual = layer(x)

    assert torch.isfinite(old_output).all()
    assert torch.equal(actual, old_output)


def test_qvq_cuda_composite_output_width_preserves_original_fp16_rounding():
    operands, _ = _case(2, 2, k=32, n=448, dtype=torch.float16)
    x, trellis = operands
    generator = torch.Generator().manual_seed(20260816)
    tensors = {
        "trellis": trellis,
        "SU": torch.randn(32, generator=generator, dtype=torch.float16).cuda(),
        "SV": torch.randn(448, generator=generator, dtype=torch.float16).cuda(),
        "bias": torch.randn(448, generator=generator, dtype=torch.float16).cuda(),
    }
    layer = QVQLinear(
        bits=2,
        in_features=32,
        out_features=448,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).eval()

    old_input = qvq_cuda_hadamard(x, pre_scale=tensors["SU"], scale_mode=1)
    old_inner = qvq_cuda_gemv(old_input, trellis, 2, out_features=448)
    old_output = matmul_hadU(old_inner) * tensors["SV"] + tensors["bias"]
    actual = layer(x)

    assert torch.isfinite(old_output).all()
    assert torch.equal(actual, old_output)


def test_qvq_cuda_composite_input_width_retries_overflow_in_bfloat16():
    operands, _ = _case(2, 1, k=448, n=32, dtype=torch.float16)
    _, trellis = operands
    x = torch.zeros((1, 448), dtype=torch.float16, device="cuda")
    x[0, 0] = 40000
    tensors = {
        "trellis": trellis,
        "SU": torch.full((448,), 2.0, dtype=torch.float32, device="cuda"),
        "SV": torch.full((32,), 1e-4, dtype=torch.float32, device="cuda"),
    }
    reference_layer = QVQReferenceLinear(
        bits=2,
        in_features=448,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).train()
    cuda_layer = QVQLinear(
        bits=2,
        in_features=448,
        out_features=32,
        name="proj",
        tensors=tensors,
        dtype=torch.float16,
    ).eval()
    observed = []
    native = qvq_cuda_gemv

    def capture_dtype(inner_x, *args, **kwargs):
        observed.append((inner_x.dtype, kwargs["output_fp32"]))
        return native(inner_x, *args, **kwargs)

    with patch("gptqmodel.utils.qvq_cuda.qvq_cuda_gemv", side_effect=capture_dtype):
        actual = cuda_layer(x)
    reference = reference_layer._forward_compute_dtype(x.float(), torch.float32)

    assert observed == [(torch.float16, True), (torch.bfloat16, True)]
    assert torch.isfinite(actual).all()
    relative_l2 = ((actual.float() - reference).square().sum() / reference.square().sum()).sqrt()
    assert relative_l2 < 0.01
    _assert_accuracy(actual.float().cpu(), reference.cpu(), max_kld=2e-3, min_top1=0.96875, min_top5=0.96875)


@pytest.mark.parametrize("width", (8192, 14336))
def test_qvq_stable_fp16_hadamard_avoids_delayed_normalization_overflow(width):
    source = torch.full((1, width), 56.0, dtype=torch.float16, device="cuda")
    reference = matmul_hadU(source.float())

    delayed = matmul_hadU(source)
    stable = matmul_hadU_stable(source)

    assert not torch.isfinite(delayed).all()
    assert torch.isfinite(stable).all()
    torch.testing.assert_close(stable.float(), reference, rtol=3e-3, atol=2e-2)




def test_qvq_cuda_linear_training_uses_differentiable_reference_path():
    operands, _ = _case(4, 2, k=32, n=32)
    x, trellis = operands
    tensors = {
        "trellis": trellis,
        "SU": torch.ones(32, dtype=torch.float16, device="cuda"),
        "SV": torch.ones(32, dtype=torch.float16, device="cuda"),
    }
    layer = QVQLinear(bits=4, in_features=32, out_features=32, name="proj", tensors=tensors).train()
    with patch(
        "gptqmodel.utils.qvq_cuda.qvq_cuda_gemv",
        side_effect=AssertionError("CUDA inference path called"),
    ):
        output = layer(x.requires_grad_(True))

    output.float().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_qvq_v4_cuda_gemv_multitile_matches_reference():
    """V4 must cover multiple K tiles; one-tile parity does not catch staging bugs."""
    bits = 2
    vector_size = 4
    k, n, m = 64, 32, 3
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    generator = torch.Generator(device="cpu").manual_seed(4491)
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (64, tiles), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous().cuda()
    x = torch.randn((m, k), generator=generator, dtype=torch.float16).cuda()
    inner = reconstruct_qvq_inner_weight(trellis, bits=bits, vector_size=vector_size, in_features=k, out_features=n)
    reference = (x.float() @ inner.float()).to(torch.float16)
    actual = qvq_cuda_gemv(x, trellis, bits, out_features=n, vector_size=vector_size)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("bits", tuple(rate for rate in QVQ_CUDA_BITS if rate <= 4))
def test_qvq_v4_cuda_bank_dispatch_matches_selected_reference_tiles(bits):
    vector_size = 4
    k, n, m = 64, 64, 5
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    generator = torch.Generator(device="cpu").manual_seed(4492)
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (64, tiles), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous().cuda()
    bank_ids = torch.arange(tiles, dtype=torch.uint8, device="cuda") % 4
    x = torch.randn((m, k), generator=generator, dtype=torch.float16).cuda()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=vector_size,
        in_features=k,
        out_features=n,
        bank_ids=bank_ids,
    )
    reference = x.float() @ inner.float()
    actual = qvq_cuda_gemv(
        x,
        trellis,
        bits,
        out_features=n,
        vector_size=vector_size,
        output_fp32=True,
        bank_ids=bank_ids,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)


@pytest.mark.parametrize("bits", (3.0, 3.5))
@pytest.mark.parametrize("format_name", ("v2b2-p32", "v2b4-p64"))
@pytest.mark.parametrize("m", (1, 17))
def test_qvq_segmented_v2_cuda_gemv_matches_dense_reference(bits, format_name, m):
    """W3/W3.5 packed selectors must decode natively without dense reconstruction."""

    k, n = 64, 64
    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator(device="cpu").manual_seed(5330 + int(bits * 10) + m + len(format_name))
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (128, tiles), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous().cuda()
    x = torch.randn((m, k), generator=generator, dtype=torch.float16).cuda()
    if format_name == "v2b2-p32":
        dense_selectors = (torch.arange(tiles * 8, dtype=torch.uint8) % 2).contiguous()
        bank_ids = pack_qvq_binary_bank_ids(dense_selectors).cuda()
        bank_alt_id = 3
        format_kwargs = {
            "v2b2_p32": True,
            "bank_alt_id": torch.tensor([bank_alt_id], dtype=torch.uint8, device="cuda"),
        }
        kernel_kwargs = {"v2b2_p32": True, "bank_alt_id": bank_alt_id}
    else:
        dense_selectors = (torch.arange(tiles * 4, dtype=torch.uint8) % 4).contiguous()
        bank_ids = pack_qvq_bank_ids(dense_selectors).cuda()
        format_kwargs = {"v2b4_p64": True}
        kernel_kwargs = {"v2b4_p64": True}
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        in_features=k,
        out_features=n,
        bank_ids=bank_ids,
        **format_kwargs,
    )
    reference = x.float() @ inner.float()
    actual = qvq_cuda_gemv(
        x,
        trellis,
        bits,
        out_features=n,
        output_fp32=True,
        bank_ids=bank_ids,
        **kernel_kwargs,
    )
    error = (actual - reference).abs()
    assert torch.isfinite(actual).all()
    assert error.max().item() <= 2e-3


@pytest.mark.parametrize("m", (1, 8, 17))
@pytest.mark.parametrize(
    ("output_widths", "alternative_banks"),
    (
        ((32, 64), (1, 3)),
        ((32, 64, 48), (1, 3, 2)),
    ),
)
def test_qvq_grouped_p32_cuda_gemv_preserves_independent_alternative_banks(
    m, output_widths, alternative_banks
):
    bits = 2.0
    k = 64
    transition_bits = qvq_transition_bits(bits)
    words_per_tile = qvq_words_per_tile(bits)
    generator = torch.Generator(device="cpu").manual_seed(5371 + m)
    k_tiles = k // 16
    x = torch.randn((m, k), generator=generator, dtype=torch.float16).cuda()
    trellises = []
    bank_selectors = []
    references = []
    for out_features, bank_alt_id in zip(
        output_widths, alternative_banks, strict=True
    ):
        n_tiles = out_features // 16
        tile_count = k_tiles * n_tiles
        edges = torch.randint(
            0,
            1 << transition_bits,
            (128, tile_count),
            generator=generator,
            dtype=torch.int32,
        )
        trellis = planar_pack_rows(edges, transition_bits).T.contiguous().cuda()
        bank_ids = pack_qvq_binary_bank_ids(
            torch.randint(
                0,
                2,
                (tile_count * 8,),
                generator=generator,
                dtype=torch.uint8,
            )
        ).cuda()
        inner = reconstruct_qvq_inner_weight(
            trellis,
            bits=bits,
            in_features=k,
            out_features=out_features,
            bank_ids=bank_ids,
            v2b2_p32=True,
            bank_alt_id=torch.tensor(
                [bank_alt_id], dtype=torch.uint8, device="cuda"
            ),
        )
        trellises.append(trellis.view(k_tiles, n_tiles, words_per_tile))
        bank_selectors.append(bank_ids.view(k_tiles, n_tiles))
        references.append(x.float() @ inner.float())

    grouped_trellis = torch.cat(trellises, dim=1).reshape(-1, words_per_tile)
    grouped_bank_ids = torch.cat(bank_selectors, dim=1).reshape(-1)
    grouped_bank_alt_ids = torch.tensor(
        alternative_banks, dtype=torch.uint8, device="cuda"
    )
    grouped_bank_alt_boundaries = tuple(
        sum(output_widths[:index]) // 16
        for index in range(1, len(output_widths))
    )
    actual = qvq_cuda_gemv(
        x,
        grouped_trellis,
        bits,
        out_features=sum(output_widths),
        output_fp32=True,
        bank_ids=grouped_bank_ids,
        v2b2_p32=True,
        bank_alt_ids=grouped_bank_alt_ids,
        bank_alt_boundaries=grouped_bank_alt_boundaries,
    )
    reference = torch.cat(references, dim=-1)

    assert torch.isfinite(actual).all()
    assert (actual - reference).abs().max().item() <= 2e-3


def test_qvq_segmented_v2_cuda_gemv_rejects_invalid_contracts():
    x = torch.zeros((1, 16), dtype=torch.float16, device="cuda")
    trellis_w3 = torch.zeros((1, 24), dtype=torch.int32, device="cuda")
    trellis_w4 = torch.zeros((1, 32), dtype=torch.int32, device="cuda")
    selectors = torch.zeros((1,), dtype=torch.uint8, device="cuda")

    with pytest.raises(ValueError, match="exactly one packed selector"):
        qvq_cuda_gemv(x, trellis_w3, 3.0, out_features=16, v2b4_p64=True)
    with pytest.raises(ValueError, match="W1 through W3.5"):
        qvq_cuda_gemv(
            x,
            trellis_w4,
            4.0,
            out_features=16,
            bank_ids=selectors,
            v2b4_p64=True,
        )
    with pytest.raises(ValueError, match="bank_alt_id must be in"):
        qvq_cuda_gemv(
            x,
            trellis_w3,
            3.0,
            out_features=16,
            bank_ids=selectors,
            v2b2_p32=True,
            bank_alt_id=4,
        )
    grouped_trellis_w3 = torch.zeros((2, 24), dtype=torch.int32, device="cuda")
    grouped_selectors = torch.zeros((2,), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="grouped alternative-bank IDs must be in"):
        qvq_cuda_gemv(
            x,
            grouped_trellis_w3,
            3.0,
            out_features=32,
            bank_ids=grouped_selectors,
            v2b2_p32=True,
            bank_alt_ids=torch.zeros((2,), dtype=torch.uint8, device="cuda"),
            bank_alt_boundaries=(1,),
        )
    with pytest.raises(ValueError, match="strictly increasing N16 indices"):
        qvq_cuda_gemv(
            x,
            grouped_trellis_w3,
            3.0,
            out_features=32,
            bank_ids=grouped_selectors,
            v2b2_p32=True,
            bank_alt_ids=torch.ones((2,), dtype=torch.uint8, device="cuda"),
            bank_alt_boundaries=(2,),
        )


@pytest.mark.parametrize(("bits", "format_name"), ((3.0, "v2b2-p32"), (3.5, "v2b4-p64")))
def test_qvq_segmented_v2_linear_routes_w3_rates_to_native_cuda(bits, format_name):
    k = n = 16
    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator(device="cpu").manual_seed(5390 + int(bits * 10))
    edges = torch.randint(0, 1 << transition_bits, (128, 1), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous().cuda()
    if format_name == "v2b2-p32":
        bank_ids = pack_qvq_binary_bank_ids(torch.arange(8, dtype=torch.uint8) % 2).cuda()
        tensors = {
            "trellis": trellis,
            "SU": torch.ones(k, dtype=torch.float16, device="cuda"),
            "SV": torch.ones(n, dtype=torch.float16, device="cuda"),
            "bank_ids": bank_ids,
            "bank_alt_id": torch.tensor([3], dtype=torch.uint8, device="cuda"),
        }
        format_kwargs = {"bank_count": 2, "v2b2_p32": True}
    else:
        bank_ids = pack_qvq_bank_ids(torch.arange(4, dtype=torch.uint8)).cuda()
        tensors = {
            "trellis": trellis,
            "SU": torch.ones(k, dtype=torch.float16, device="cuda"),
            "SV": torch.ones(n, dtype=torch.float16, device="cuda"),
            "bank_ids": bank_ids,
        }
        format_kwargs = {"bank_count": 4, "v2b4_p64": True}
    layer = QVQLinear(bits=bits, in_features=k, out_features=n, tensors=tensors, **format_kwargs).eval()
    x = torch.randn((3, k), generator=generator, dtype=torch.float16).cuda()
    reference = x.float() @ layer.get_inner_weight_tensor(dtype=torch.float32)
    with patch("gptqmodel.utils.qvq_cuda.qvq_cuda_gemv", wraps=qvq_cuda_gemv) as native:
        actual = layer._inner_forward(x)
    assert native.call_count == 1
    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)


@pytest.mark.parametrize("bits", tuple(rate for rate in QVQ_CUDA_BITS if rate <= 4))
@pytest.mark.parametrize("m", (1, 4, 16, 32))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_qvq_v4_cuda_all_rates_batches_and_dtypes_match_dense_reference(bits, m, dtype):
    """V4 inference covers W1 through W4 at every half-rate, not only W2."""
    vector_size = 4
    k, n = 64, 48
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    generator = torch.Generator(device="cpu").manual_seed(18000 + transition_bits * 100 + m)
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (64, tiles), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous().cuda()
    x = torch.randn((m, k), generator=generator, dtype=torch.float32).to(device="cuda", dtype=dtype)
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=vector_size,
        in_features=k,
        out_features=n,
    )
    reference = x.float() @ inner.float()
    actual = qvq_cuda_gemv(x, trellis, bits, out_features=n, vector_size=vector_size, output_fp32=True)
    delta = (actual.float() - reference.float()).abs()
    assert delta.max().item() <= 2e-3
    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)


def test_qvq_v4_cuda_quantization_uses_reference_until_native_dispatch():
    """V4 CUDA Viterbi must return finite, correctly shaped production paths."""
    from gptqmodel.quantization.qvq import batched_viterbi_quantize
    sequences = torch.randn((1, 64, 4), dtype=torch.float32, device="cuda")
    codebook = pgc16_codebook_v4(device="cuda", dtype=torch.float32)
    result = batched_viterbi_quantize(sequences, codebook, bits=2)
    assert result.states.shape == (1, 64)
    assert torch.isfinite(result.squared_error).all()


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
def test_qvq_v4_device_local_trellis_pack_is_bit_exact(bits):
    """Device-local packing must equal CPU packing at every V4 transition width."""

    transition_bits = qvq_transition_bits(bits, vector_size=4)
    generator = torch.Generator().manual_seed(20260820 + transition_bits)
    packed_reference = torch.randint(
        -(1 << 31),
        1 << 31,
        (9, transition_bits * 2),
        generator=generator,
        dtype=torch.int32,
    )
    states = unpack_trellis_states(packed_reference, bits=bits, vector_size=4)
    expected = pack_trellis_states(states, bits=bits, vector_size=4)
    actual = pack_trellis_states(states.cuda(), bits=bits, vector_size=4)
    torch.cuda.synchronize()
    assert actual.is_cuda
    assert torch.equal(actual.cpu(), expected)


def test_qvq_v4_cuda_telemetry_reports_native_dispatch_without_changing_math():
    generator = torch.Generator().manual_seed(20260821)
    weight = torch.randn((16, 16), generator=generator, dtype=torch.float32).cuda()
    hessian = torch.eye(16, dtype=torch.float32, device="cuda")
    control = quantize_qvq_linear(weight, hessian, bits=2, vector_size=4, trellis_batch_size=1)
    measured = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        vector_size=4,
        trellis_batch_size=1,
        telemetry=QVQQuantizationTelemetry(),
    )
    assert torch.equal(measured.trellis, control.trellis)
    assert torch.equal(measured.weight, control.weight)
    assert measured.telemetry["counters"]["native_viterbi_launches"] == 2
    assert measured.telemetry["phases"]["block_ldl_viterbi"]["gpu_ms"] >= 0
    assert measured.telemetry["phases"]["pack_trellis"]["gpu_ms"] >= 0


@pytest.mark.parametrize("vector_size", (2, 4))
def test_qvq_production_codebook_cache_reuses_cuda_storage(vector_size):
    """Repeated module quantization must share one immutable table per format."""

    first = _canonical_qvq_codebook(
        device=torch.device("cuda"),
        vector_size=vector_size,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float16,
    )
    second = _canonical_qvq_codebook(
        device=torch.device("cuda"),
        vector_size=vector_size,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float16,
    )
    assert first.data_ptr() == second.data_ptr()
    assert first.is_cuda and first.dtype == torch.float16


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("steps", (1, 3, 17))
@pytest.mark.parametrize("codebook_dtype", (torch.float16, torch.float32))
def test_qvq_v4_cuda_viterbi_path_matches_cpu_reference(bits, steps, codebook_dtype):
    """The production V4 CUDA quantizer must choose the eager oracle's exact path."""

    generator = torch.Generator().manual_seed(20260813 + int(bits * 100) + steps)
    cpu_sequences = torch.randn((2, steps, 4), generator=generator, dtype=torch.float32)
    cpu_codebook = pgc16_codebook_v4(device="cpu", dtype=torch.float32)
    expected = batched_viterbi_quantize(cpu_sequences, cpu_codebook, bits=bits)

    actual_states, actual_error = qvq_cuda_viterbi(
        cpu_sequences.cuda(),
        cpu_codebook.to(device="cuda", dtype=codebook_dtype),
        bits,
        vector_size=4,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_states.cpu(), expected.states)
    # The selected path is the exact contract.  The loss scalar uses CUDA
    # FP32 FMA order while the eager oracle uses the host BLAS reduction;
    # those legal reduction orders differ by a few ulps.
    torch.testing.assert_close(actual_error.cpu(), expected.squared_error, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("bits", QVQ_CUDA_BITS)
def test_qvq_v2_cuda_fp16_codebook_matches_cpu_reference(bits):
    """Frozen FP16 PGC16 storage must preserve every V2 rate's selected path."""

    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(20260830 + transition_bits)
    cpu_sequences = torch.randn((2, 3, 2), generator=generator, dtype=torch.float32)
    cpu_codebook = pgc16_codebook(device="cpu", dtype=torch.float32)
    expected = batched_viterbi_quantize(cpu_sequences, cpu_codebook, bits=bits)

    actual_states, actual_error = qvq_cuda_viterbi(
        cpu_sequences.cuda(),
        cpu_codebook.to(device="cuda", dtype=torch.float16),
        bits,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_states.cpu(), expected.states)
    torch.testing.assert_close(actual_error.cpu(), expected.squared_error, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("codebook_dtype", (torch.float16, torch.float32))
def test_qvq_v4_cuda_viterbi_covers_overlap_and_weighted_edges(codebook_dtype):
    """Overlap constraints, zero weights, and nontrivial weights share the V4 kernel."""

    generator = torch.Generator().manual_seed(20260814)
    sequences = torch.randn((2, 9, 4), generator=generator, dtype=torch.float32)
    codebook = pgc16_codebook_v4(device="cpu", dtype=torch.float32)
    overlap_bits = 16 - 4 * 2
    overlap = torch.tensor([0, (1 << overlap_bits) - 1], dtype=torch.int64)
    weights = torch.tensor(
        [[0.0, 1.0, 0.5, 2.0, 1.0, 0.25, 3.0, 1.0, 0.75],
         [1.0, 0.0, 2.0, 0.5, 1.5, 1.0, 0.25, 2.0, 1.0]],
        dtype=torch.float32,
    )
    expected = batched_viterbi_quantize(sequences, codebook, bits=2, overlap=overlap, step_weights=weights)
    states, error = qvq_cuda_viterbi(
        sequences.cuda(),
        codebook.to(device="cuda", dtype=codebook_dtype),
        2,
        overlap=overlap.cuda(),
        step_weights=weights.cuda(),
        vector_size=4,
    )
    torch.cuda.synchronize()
    assert torch.equal(states.cpu(), expected.states)
    torch.testing.assert_close(error.cpu(), expected.squared_error, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("codebook_dtype", (torch.float16, torch.float32))
def test_qvq_v4_cuda_w4_memoryless_weighted_matches_reference(codebook_dtype):
    """W4's independent-step fast path preserves weighted selection and loss."""

    generator = torch.Generator().manual_seed(20260815)
    sequences = torch.randn((3, 7, 4), generator=generator, dtype=torch.float32)
    codebook = pgc16_codebook_v4(device="cpu", dtype=torch.float32)
    weights = torch.tensor(
        [[0.0, 1.0, 0.5, 2.0, 1.0, 0.25, 3.0],
         [1.0, 0.0, 2.0, 0.5, 1.5, 1.0, 0.25],
         [2.0, 1.0, 0.0, 0.75, 1.25, 0.5, 1.0]],
        dtype=torch.float32,
    )
    expected = batched_viterbi_quantize(sequences, codebook, bits=4, step_weights=weights)
    states, error = qvq_cuda_viterbi(
        sequences.cuda(),
        codebook.to(device="cuda", dtype=codebook_dtype),
        4,
        step_weights=weights.cuda(),
        vector_size=4,
    )
    torch.cuda.synchronize()
    assert torch.equal(states.cpu(), expected.states)
    torch.testing.assert_close(error.cpu(), expected.squared_error, rtol=1e-6, atol=1e-5)


# ---------------------------------------------------------------------------
# Norm-rank contiguous-band grid recurrence (W2.5/W3.0 half-codebook fast path)
# ---------------------------------------------------------------------------


def _norm_rank_case(seed, batch, bits, bank_count, codebook_dtype=torch.float16, sequence_scale=None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    if sequence_scale is not None:
        sequences = sequences * sequence_scale
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=bits, dtype=torch.float32) for bank in range(bank_count))
    ).to(device="cuda", dtype=codebook_dtype)
    return sequences, codebooks


def _norm_rank_dispatch_count():
    _qvq_cuda_viterbi_v2_segment_grid_trusted_op()  # ensure the extension is loaded
    return int(torch.ops.gptqmodel_qvq.norm_rank_grid_dispatch_count())


@pytest.mark.parametrize("bits,bank_count,segment_steps", ((2.5, 2, 16), (2.5, 4, 32), (3.0, 2, 16), (3.0, 4, 32)))
def test_qvq_cuda_norm_rank_grid_dispatches_and_is_bit_exact(monkeypatch, bits, bank_count, segment_steps):
    """The norm-rank band path must actually dispatch for supported configs and
    agree bit-for-bit with the segmented banked reference."""

    sequences, codebooks = _norm_rank_case(20260825 + bank_count, 9, bits, bank_count)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    before = _norm_rank_dispatch_count()
    actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences, codebooks, transition_bits, segment_steps, None, None
    )
    assert _norm_rank_dispatch_count() == before + 1
    expected = qvq_cuda_viterbi_v2_segment_banked(sequences, codebooks, bits, segment_steps, None, None)
    assert all(torch.equal(e, a) for e, a in zip(expected, actual))
    # This is an independent eager recurrence oracle, not another CUDA kernel.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    eager = _batched_v2_banked_viterbi_quantize(
        sequences, codebooks, bits=bits, segment_steps=segment_steps
    )
    assert torch.equal(actual[0], eager.states)
    assert torch.equal(actual[1], eager.squared_error)
    assert torch.equal(actual[2], eager.segment_bank_ids)


@pytest.mark.parametrize(
    "bits,bank_count,segment_steps,constrained,weighted,codebook_dtype",
    (
        (2.0, 2, 16, False, False, torch.float16),   # shift 4: candidate list too short to pay for banding
        (3.5, 2, 16, False, False, torch.float16),   # shift 7: unsupported rate
        (2.5, 2, 16, True, False, torch.float16),    # constrained keeps the exact fallback
        (2.5, 2, 16, False, True, torch.float16),    # weighted keeps the exact fallback
        (3.0, 4, 32, True, True, torch.float16),
        (3.0, 2, 16, False, False, torch.float32),   # fp32 codebooks keep the reference path
    ),
)
def test_qvq_cuda_norm_rank_grid_leaves_unsupported_configs_on_reference_path(
    bits, bank_count, segment_steps, constrained, weighted, codebook_dtype
):
    sequences, codebooks = _norm_rank_case(20260826, 5, bits, bank_count, codebook_dtype=codebook_dtype)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    generator = torch.Generator(device="cuda").manual_seed(20260827)
    overlap = (
        torch.randint(0, 1 << (16 - transition_bits), (5,), generator=generator, device="cuda", dtype=torch.int64)
        if constrained
        else None
    )
    step_weights = (
        (0.1 + torch.rand((5, 128), generator=generator, device="cuda")).contiguous() if weighted else None
    )
    before = _norm_rank_dispatch_count()
    actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences, codebooks, transition_bits, segment_steps, overlap, step_weights
    )
    assert _norm_rank_dispatch_count() == before
    expected = qvq_cuda_viterbi_v2_segment_banked(
        sequences, codebooks, bits, segment_steps, overlap, step_weights
    )
    assert all(torch.equal(e, a) for e, a in zip(expected, actual))


@pytest.mark.parametrize(
    "value,enabled",
    ((None, True), ("", True), ("0", True), ("1", False), ("00", False), ("0foo", False), ("false", False)),
)
def test_qvq_cuda_norm_rank_grid_disable_env_parser(value, enabled):
    """Only unset, empty, and exact ``0`` keep the eligible path enabled.

    Each spelling runs in a subprocess so the cases stay independent; the
    variable itself is re-read on every dispatch (see the same-process
    mutation tests below).
    """

    import os
    import subprocess
    import sys

    script = r"""
import torch
from gptqmodel.quantization.qvq_codecs.pgc16 import pgc16_codebook_v2_bank
from gptqmodel.utils.qvq_cuda import (
    _qvq_cuda_viterbi_v2_segment_grid_trusted_op,
    qvq_cuda_viterbi_v2_segment_banked,
)
op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
generator = torch.Generator(device="cuda").manual_seed(20260828)
sequences = torch.randn((7, 128, 2), generator=generator, device="cuda")
codebooks = torch.stack(
    (pgc16_codebook_v2_bank(0, bits=3.0, dtype=torch.float32),
     pgc16_codebook_v2_bank(1, bits=3.0, dtype=torch.float32))
).to(device="cuda", dtype=torch.float16)
actual = op(sequences, codebooks, 6, 16, None, None)
assert int(torch.ops.gptqmodel_qvq.norm_rank_grid_dispatch_count()) == EXPECTED
expected = qvq_cuda_viterbi_v2_segment_banked(sequences, codebooks, 3.0, 16, None, None)
for index in range(3):
    assert torch.equal(expected[index], actual[index]), index
print("ENV-PARSER-OK")
"""
    env = dict(os.environ)
    if value is None:
        env.pop("GPTQMODEL_QVQ_DISABLE_OCTET_GRID", None)
    else:
        env["GPTQMODEL_QVQ_DISABLE_OCTET_GRID"] = value
    script = f"EXPECTED = {int(enabled)}\n" + script
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ENV-PARSER-OK" in result.stdout


def test_qvq_cuda_norm_rank_codebook_mutation_rebuilds_tables():
    """An in-place codebook mutation bumps the version counter, so the cached
    sorted tables must be rebuilt rather than served stale."""

    sequences, codebooks = _norm_rank_case(20260829, 6, 3.0, 2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    op(sequences, codebooks, 6, 16, None, None)
    with torch.no_grad():
        codebooks[:, :1024].mul_(-1.0)
    mutated = op(sequences, codebooks, 6, 16, None, None)
    fresh = op(sequences, codebooks.clone(), 6, 16, None, None)
    torch.cuda.synchronize()
    assert all(torch.equal(m, f) for m, f in zip(mutated, fresh))


def test_qvq_cuda_norm_rank_inference_mode_codebook_mutation_is_not_stale():
    """Inference-mode tensors keep no useful version counter, so the norm-rank
    tables must be rebuilt (uncached) every call for them."""

    sequences, source = _norm_rank_case(20260830, 4, 2.5, 2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    with torch.inference_mode():
        codebooks = source.clone()
        op(sequences, codebooks, 5, 16, None, None)
        codebooks[:, :512].mul_(-1.0)
        mutated = op(sequences, codebooks, 5, 16, None, None)
        fresh = op(sequences, codebooks.clone(), 5, 16, None, None)
    torch.cuda.synchronize()
    assert all(torch.equal(m, f) for m, f in zip(mutated, fresh))


def test_qvq_cuda_norm_rank_grid_uses_current_non_default_stream():
    sequences, codebooks = _norm_rank_case(20260831, 6, 3.0, 2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    expected = op(sequences, codebooks, 6, 16, None, None)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # A fresh clone on the side stream also exercises the table build and
        # cache insertion off the default stream.
        actual = op(sequences, codebooks.clone(), 6, 16, None, None)
        completion = torch.cuda.Event()
        completion.record(stream)
    completion.synchronize()
    assert all(torch.equal(e, a) for e, a in zip(expected, actual))


def test_qvq_cuda_norm_rank_table_cache_survives_stream_order_reuse():
    """A cache hit from a second stream must wait on the producing stream's
    ready event instead of reading half-built tables."""

    sequences, codebooks = _norm_rank_case(20260832, 6, 2.5, 4)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        first = op(sequences, codebooks, 5, 32, None, None)
    with torch.cuda.stream(consumer):
        # No cross-stream synchronisation on purpose: the cached-table hit must
        # order itself after the producer's build.
        second = op(sequences, codebooks, 5, 32, None, None)
        done = torch.cuda.Event()
        done.record(consumer)
    done.synchronize()
    producer.synchronize()
    assert all(torch.equal(f, s) for f, s in zip(first, second))


@pytest.mark.parametrize(
    "bits,bank_count,segment_steps",
    ((2.5, 2, 16), (2.5, 4, 32), (3.0, 2, 16), (3.0, 4, 32)),
)
def test_qvq_cuda_viterbi_norm_rank_repeated_later_segment_initialization(
    bits, bank_count, segment_steps
):
    """Repeatedly exercise initialization of every segment after segment zero."""

    sequences, codebooks = _norm_rank_case(20260839 + bank_count, 9, bits, bank_count)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    dispatches_before = _norm_rank_dispatch_count()
    for repeat in range(8):
        candidate_sequences = sequences.clone()
        candidate_sequences[:, repeat::segment_steps, 0].add_(repeat * 0.03125)
        actual = op(candidate_sequences, codebooks, transition_bits, segment_steps, None, None)
        expected = qvq_cuda_viterbi_v2_segment_banked(
            candidate_sequences, codebooks, bits, segment_steps, None, None
        )
        assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    # Both the trusted candidate and public banked comparison dispatch the
    # eligible norm-rank grid once per repeat at this batch size.
    assert _norm_rank_dispatch_count() == dispatches_before + 16


def test_qvq_cuda_norm_rank_cache_eviction_lifetime_and_boundedness():
    """>8 entries must remain valid across queued consumers and allocator reuse."""

    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    sequences, source = _norm_rank_case(20260836, 2, 2.5, 2)
    codebooks = [source.clone() for _ in range(10)]
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        first = op(sequences, codebooks[0], 5, 16, None, None)
    with torch.cuda.stream(consumer):
        # The cache-hit wait is the only producer-to-consumer dependency.
        queued = op(sequences, codebooks[0], 5, 16, None, None)
    for codebook in codebooks[1:]:
        op(sequences, codebook, 5, 16, None, None)
    del codebooks
    pressure = [torch.empty((1 << 20,), device="cuda", dtype=torch.uint8) for _ in range(32)]
    for tensor in pressure:
        tensor.fill_(0x5A)
    consumer.synchronize()
    producer.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(first, queued))
    assert int(torch.ops.gptqmodel_qvq.norm_rank_cache_size()) <= 8

    # Repeated versions of one storage must evict old versions and stay exact.
    _, mutable = _norm_rank_case(20260837, 2, 2.5, 2)
    for index in range(12):
        mutation_stream = torch.cuda.Stream()
        consumer_stream = torch.cuda.Stream()
        with torch.cuda.stream(mutation_stream):
            mutable[:, index, 0].add_(torch.tensor(0.125, device="cuda", dtype=torch.float16))
            mutation_done = torch.cuda.Event()
            mutation_done.record()
        # Mutation is asynchronous; make its cross-stream dependency explicit.
        consumer_stream.wait_event(mutation_done)
        with torch.cuda.stream(consumer_stream):
            mutated = op(sequences, mutable, 5, 16, None, None)
        consumer_stream.synchronize()
        fresh = op(sequences, mutable.clone(), 5, 16, None, None)
        assert all(torch.equal(a, b) for a, b in zip(mutated, fresh))
        assert int(torch.ops.gptqmodel_qvq.norm_rank_cache_size()) <= 8


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_qvq_cuda_mixed_device_cache_eviction_destroys_events_on_owner_device():
    """Both event-owning caches may evict an entry created on another GPU."""

    original_device = torch.cuda.current_device()
    for index in range(33):
        device = torch.device("cuda", 0 if index == 0 else 1)
        generator = torch.Generator(device=device).manual_seed(20260900 + index)
        sequences = torch.randn((1, 2, 4), generator=generator, device=device)
        codebook = torch.randn((1 << 16, 4), generator=generator, device=device, dtype=torch.float16)
        qvq_cuda_viterbi(sequences, codebook, bits=2.0, vector_size=4)
    with torch.cuda.device(0):
        op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
        for index in range(9):
            generator = torch.Generator(device="cuda:0").manual_seed(20261000 + index)
            sequences = torch.randn((1, 128, 2), generator=generator, device="cuda:0")
            codebooks = torch.randn((2, 1 << 16, 2), generator=generator, device="cuda:0", dtype=torch.float16)
            op(sequences, codebooks, 5, 16, None, None)
    with torch.cuda.device(1):
        generator = torch.Generator(device="cuda:1").manual_seed(20261100)
        sequences = torch.randn((1, 128, 2), generator=generator, device="cuda:1")
        codebooks = torch.randn((2, 1 << 16, 2), generator=generator, device="cuda:1", dtype=torch.float16)
        op(sequences, codebooks, 5, 16, None, None)
        assert torch.cuda.current_device() == 1
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    assert int(torch.ops.gptqmodel_qvq.norm_cache_size()) <= 32
    assert int(torch.ops.gptqmodel_qvq.norm_rank_cache_size()) <= 8
    assert torch.cuda.current_device() == original_device


def test_qvq_cuda_norm_rank_w25_bank4_nextafter_chunk_boundary_matches_eager(monkeypatch):
    """Irregular finite FP16 values around a width-4 rank boundary stay exact."""

    bits, bank_count, segment_steps = 2.5, 4, 32
    sequences, codebooks = _norm_rank_case(20260838, 3, bits, bank_count)
    lo = torch.tensor(0.5, device="cuda", dtype=torch.float16)
    hi = torch.nextafter(lo, torch.tensor(torch.inf, device="cuda", dtype=torch.float16))
    below = torch.nextafter(lo, torch.tensor(-torch.inf, device="cuda", dtype=torch.float16))
    # Prefixes 3/4 straddle a sorted width-4 chunk boundary for suffix zero.
    suffix_count = 1 << (16 - 5)
    codebooks[:, 3 * suffix_count, :] = torch.stack((below, hi))
    codebooks[:, 4 * suffix_count, :] = torch.stack((lo, below))
    sequence_values = torch.stack((below.float(), lo.float(), hi.float(), -hi.float()))
    repeated = sequence_values.repeat((sequences.numel() + 3) // 4)
    sequences.copy_(repeated[: sequences.numel()].reshape_as(sequences))
    actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(sequences, codebooks, 5, segment_steps, None, None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    eager = _batched_v2_banked_viterbi_quantize(
        sequences, codebooks, bits=bits, segment_steps=segment_steps
    )
    assert torch.equal(actual[0], eager.states)
    assert torch.equal(actual[1], eager.squared_error)
    assert torch.equal(actual[2], eager.segment_bank_ids)


@pytest.mark.parametrize("bits,bank_count,segment_steps", ((2.5, 2, 16), (3.0, 2, 16), (3.0, 4, 32)))
@pytest.mark.parametrize("pattern", ("ties", "tiny", "large"))
def test_qvq_cuda_norm_rank_rounding_edges_and_ties_match_banked_reference(bits, bank_count, segment_steps, pattern):
    """Adversarial inputs for the directed-rounding band derivation: massive
    exact value ties (the lowest original prefix must win), denormal-scale
    sequences, and large sequences near the finite-accumulation contract."""

    transition_bits = qvq_transition_bits(bits, vector_size=2)
    if pattern == "ties":
        # Repeat one small code table across every prefix of each suffix column:
        # every candidate value ties, so selections exercise pure tie-breaking.
        generator = torch.Generator(device="cuda").manual_seed(20260833)
        base = torch.randn((bank_count, 1 << (16 - transition_bits), 2), generator=generator, device="cuda")
        codebooks = (
            base.repeat(1, 1 << transition_bits, 1).to(torch.float16).contiguous()
        )
        sequences = torch.randn((5, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    elif pattern == "tiny":
        sequences, codebooks = _norm_rank_case(20260834, 5, bits, bank_count, sequence_scale=1e-30)
    else:
        # Just inside the finite FP32 squared-distance contract of the op.
        sequences, codebooks = _norm_rank_case(20260835, 5, bits, bank_count, sequence_scale=1e15)
    actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences, codebooks, transition_bits, segment_steps, None, None
    )
    expected = qvq_cuda_viterbi_v2_segment_banked(sequences, codebooks, bits, segment_steps, None, None)
    assert all(torch.equal(e, a) for e, a in zip(expected, actual))


# ---------------------------------------------------------------------------
# ViterbiPruningConfig policy -> CUDA V2 segmented grid dispatch
# ---------------------------------------------------------------------------


def _pruning_code(**kwargs):
    """Resolve a public `ViterbiPruningConfig` into its native policy code."""

    return viterbi_pruning_dispatch_code(ViterbiPruningConfig(**kwargs))


@pytest.mark.parametrize("bits,bank_count,segment_steps", ((2.5, 2, 16), (2.5, 4, 32), (3.0, 2, 16), (3.0, 4, 32)))
@pytest.mark.parametrize("mode", ("auto", "required"))
def test_qvq_pruning_policy_dispatches_eligible_cells(bits, bank_count, segment_steps, mode):
    """`auto` and `required` both norm-band dispatch every eligible cell and
    stay bit-exact against the unmodified reference."""

    sequences, codebooks = _norm_rank_case(20260901 + bank_count, 9, bits, bank_count)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    # The oracle explicitly forces the pristine baseline recurrence with
    # `mode="off"`; a default-`auto` reference could itself take the norm-band
    # path and the comparison would prove nothing.
    reference = qvq_cuda_viterbi_v2_segment_banked(
        sequences, codebooks, bits, segment_steps, None, None, _pruning_code(mode="off")
    )
    before = _norm_rank_dispatch_count()
    actual = op(
        sequences, codebooks, transition_bits, segment_steps, None, None, _pruning_code(mode=mode)
    )
    assert _norm_rank_dispatch_count() == before + 1
    assert all(torch.equal(e, a) for e, a in zip(reference, actual))


@pytest.mark.parametrize("bits,bank_count,segment_steps", ((2.5, 2, 16), (3.0, 4, 32)))
def test_qvq_pruning_policy_off_suppresses_eligible_dispatch(bits, bank_count, segment_steps):
    """`off` deterministically suppresses norm-band dispatch on a cell that
    `auto` would have dispatched, and the baseline result is identical."""

    sequences, codebooks = _norm_rank_case(20260902 + bank_count, 9, bits, bank_count)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()

    before = _norm_rank_dispatch_count()
    dispatched = op(
        sequences, codebooks, transition_bits, segment_steps, None, None, _pruning_code(mode="auto")
    )
    assert _norm_rank_dispatch_count() == before + 1

    before = _norm_rank_dispatch_count()
    suppressed = op(
        sequences, codebooks, transition_bits, segment_steps, None, None, _pruning_code(mode="off")
    )
    assert _norm_rank_dispatch_count() == before
    assert all(torch.equal(d, s) for d, s in zip(dispatched, suppressed))


# W1.5/W2/W3.5 stay outside the benchmark-supported W2.5/W3 set, and fp32
# codebooks, constrained calls, and weighted calls keep the exact baseline.
_UNSUPPORTED_PRUNING_CELLS = (
    pytest.param(2.0, 2, 16, False, False, torch.float16, "transition_bits=4", id="w2-rate"),
    pytest.param(1.5, 2, 16, False, False, torch.float16, "transition_bits=3", id="w1p5-rate"),
    pytest.param(3.5, 2, 16, False, False, torch.float16, "transition_bits=7", id="w3p5-rate"),
    pytest.param(3.0, 2, 16, False, False, torch.float32, "float16 codebooks", id="fp32-codebooks"),
    pytest.param(2.5, 2, 16, True, False, torch.float16, "constrained", id="constrained"),
    pytest.param(3.0, 4, 32, False, True, torch.float16, "weighted", id="weighted"),
)


@pytest.mark.parametrize(
    "bits,bank_count,segment_steps,constrained,weighted,codebook_dtype,expected",
    _UNSUPPORTED_PRUNING_CELLS,
)
def test_qvq_pruning_policy_auto_leaves_unsupported_cells_on_baseline(
    bits, bank_count, segment_steps, constrained, weighted, codebook_dtype, expected
):
    """`auto` with the default `baseline` fallback keeps every unsupported cell
    on the unmodified reference recurrence."""

    sequences, codebooks = _norm_rank_case(20260903, 5, bits, bank_count, codebook_dtype=codebook_dtype)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    generator = torch.Generator(device="cuda").manual_seed(20260904)
    overlap = (
        torch.randint(0, 1 << (16 - transition_bits), (5,), generator=generator, device="cuda", dtype=torch.int64)
        if constrained
        else None
    )
    step_weights = (
        (0.1 + torch.rand((5, 128), generator=generator, device="cuda")).contiguous() if weighted else None
    )
    before = _norm_rank_dispatch_count()
    actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences, codebooks, transition_bits, segment_steps, overlap, step_weights, _pruning_code(mode="auto")
    )
    assert _norm_rank_dispatch_count() == before
    expected_triple = qvq_cuda_viterbi_v2_segment_banked(
        sequences, codebooks, bits, segment_steps, overlap, step_weights
    )
    assert all(torch.equal(e, a) for e, a in zip(expected_triple, actual))


@pytest.mark.parametrize(
    "bits,bank_count,segment_steps,constrained,weighted,codebook_dtype,expected",
    _UNSUPPORTED_PRUNING_CELLS,
)
@pytest.mark.parametrize(
    "policy",
    (
        pytest.param({"mode": "required"}, id="required"),
        pytest.param({"mode": "auto", "fallback": "error"}, id="auto-fallback-error"),
    ),
)
def test_qvq_pruning_policy_rejects_unsupported_cells_before_silent_fallback(
    bits, bank_count, segment_steps, constrained, weighted, codebook_dtype, expected, policy
):
    """`required` and `auto`+`fallback="error"` must raise, naming the reason,
    instead of silently using the baseline recurrence."""

    sequences, codebooks = _norm_rank_case(20260905, 5, bits, bank_count, codebook_dtype=codebook_dtype)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    generator = torch.Generator(device="cuda").manual_seed(20260906)
    overlap = (
        torch.randint(0, 1 << (16 - transition_bits), (5,), generator=generator, device="cuda", dtype=torch.int64)
        if constrained
        else None
    )
    step_weights = (
        (0.1 + torch.rand((5, 128), generator=generator, device="cuda")).contiguous() if weighted else None
    )
    before = _norm_rank_dispatch_count()
    with pytest.raises(RuntimeError) as excinfo:
        _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
            sequences, codebooks, transition_bits, segment_steps, overlap, step_weights,
            _pruning_code(**policy),
        )
    assert "cannot use it" in str(excinfo.value)
    assert expected in str(excinfo.value)
    assert _norm_rank_dispatch_count() == before


def test_qvq_pruning_policy_code_is_validated_natively():
    sequences, codebooks = _norm_rank_case(20260907, 4, 3.0, 2)
    with pytest.raises(RuntimeError, match="pruning policy must be 0"):
        _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(sequences, codebooks, 6, 16, None, None, 4)


def test_qvq_pruning_policy_default_argument_matches_auto():
    """Direct low-level callers that omit the argument keep today's behavior."""

    sequences, codebooks = _norm_rank_case(20260908, 6, 3.0, 2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    before = _norm_rank_dispatch_count()
    legacy = op(sequences, codebooks, 6, 16, None, None)
    assert _norm_rank_dispatch_count() == before + 1
    explicit = op(sequences, codebooks, 6, 16, None, None, _pruning_code(mode="auto"))
    assert all(torch.equal(a, b) for a, b in zip(legacy, explicit))


def test_qvq_pruning_telemetry_reports_candidate_reduction(monkeypatch):
    """Opt-in telemetry counts real candidate work, not a timing estimate."""

    monkeypatch.setenv("GPTQMODEL_QVQ_TELEMETRY", "1")
    sequences, codebooks = _norm_rank_case(20260915, 9, 3.0, 2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    snapshot = torch.ops.gptqmodel_qvq.norm_rank_telemetry_snapshot
    before = tuple(int(value) for value in snapshot())
    op(sequences, codebooks, 6, 16, None, None, _pruning_code(mode="auto"))
    after = tuple(int(value) for value in snapshot())

    dispatches = after[0] - before[0]
    evaluated = after[2] - before[2]
    possible = after[3] - before[3]
    assert dispatches == 1
    assert possible == 9 * 2 * 127 * (1 << 16)
    assert 0 < evaluated < possible
    assert possible - evaluated > 0


@pytest.mark.parametrize(
    "mode,fallback,env,expected",
    (
        # `auto` keeps honoring the deprecated A/B escape hatch in both directions.
        ("auto", "baseline", None, "dispatched"),
        ("auto", "baseline", "1", "suppressed"),
        # Explicit configuration is authoritative: `off`/`required` ignore it.
        ("off", "baseline", None, "suppressed"),
        ("off", "baseline", "1", "suppressed"),
        ("required", "baseline", None, "dispatched"),
        ("required", "baseline", "1", "dispatched"),
        # Under `auto` the variable makes the call unable to prune, which
        # `fallback="error"` reports rather than silently falling back.
        ("auto", "error", None, "dispatched"),
        ("auto", "error", "1", "raised"),
    ),
)
def test_qvq_pruning_config_precedence_over_the_legacy_env_variable(mode, fallback, env, expected):
    """`GPTQMODEL_QVQ_DISABLE_OCTET_GRID` is a deprecated A/B escape honored
    only under `mode="auto"`. Each combination runs in its own subprocess so
    the eight cases cannot contaminate one another's environment; the
    same-process mutation contract is covered separately below."""

    import os
    import subprocess
    import sys

    script = r"""
import torch
from gptqmodel.quantization.config import ViterbiPruningConfig
from gptqmodel.quantization.qvq_pruning import viterbi_pruning_dispatch_code
from gptqmodel.quantization.qvq_codecs.pgc16 import pgc16_codebook_v2_bank
from gptqmodel.utils.qvq_cuda import (
    _qvq_cuda_viterbi_v2_segment_grid_trusted_op,
    qvq_cuda_viterbi_v2_segment_banked,
)
op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
generator = torch.Generator(device="cuda").manual_seed(20260909)
sequences = torch.randn((7, 128, 2), generator=generator, device="cuda")
codebooks = torch.stack(
    (pgc16_codebook_v2_bank(0, bits=3.0, dtype=torch.float32),
     pgc16_codebook_v2_bank(1, bits=3.0, dtype=torch.float32))
).to(device="cuda", dtype=torch.float16)
code = viterbi_pruning_dispatch_code(ViterbiPruningConfig(mode=MODE, fallback=FALLBACK))
before = int(torch.ops.gptqmodel_qvq.norm_rank_grid_dispatch_count())
if EXPECTED == "raised":
    try:
        op(sequences, codebooks, 6, 16, None, None, code)
    except RuntimeError as error:
        assert "GPTQMODEL_QVQ_DISABLE_OCTET_GRID" in str(error), str(error)
    else:
        raise AssertionError("expected a refusal")
    assert int(torch.ops.gptqmodel_qvq.norm_rank_grid_dispatch_count()) == before
else:
    actual = op(sequences, codebooks, 6, 16, None, None, code)
    delta = int(torch.ops.gptqmodel_qvq.norm_rank_grid_dispatch_count()) - before
    assert delta == (1 if EXPECTED == "dispatched" else 0), delta
    expected = qvq_cuda_viterbi_v2_segment_banked(sequences, codebooks, 3.0, 16, None, None)
    for index in range(3):
        assert torch.equal(expected[index], actual[index]), index
print("PRECEDENCE-OK")
"""
    env_vars = dict(os.environ)
    if env is None:
        env_vars.pop("GPTQMODEL_QVQ_DISABLE_OCTET_GRID", None)
    else:
        env_vars["GPTQMODEL_QVQ_DISABLE_OCTET_GRID"] = env
    script = f"MODE = {mode!r}\nFALLBACK = {fallback!r}\nEXPECTED = {expected!r}\n" + script
    result = subprocess.run(
        [sys.executable, "-c", script], env=env_vars, capture_output=True, text=True, timeout=900, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "PRECEDENCE-OK" in result.stdout


def test_qvq_pruning_legacy_env_mutation_is_observed_same_process():
    """B2: `GPTQMODEL_QVQ_DISABLE_OCTET_GRID` must be re-read on every
    dispatch. Mutating it between calls in one process deterministically
    toggles the `auto` fast path, and clearing it restores dispatch — no
    stale cached process-global policy."""

    import os

    sequences, codebooks = _norm_rank_case(20260913, 7, 3.0, 2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    auto = _pruning_code(mode="auto")
    saved = os.environ.pop("GPTQMODEL_QVQ_DISABLE_OCTET_GRID", None)
    try:
        before = _norm_rank_dispatch_count()
        first = op(sequences, codebooks, 6, 16, None, None, auto)
        assert _norm_rank_dispatch_count() == before + 1

        os.environ["GPTQMODEL_QVQ_DISABLE_OCTET_GRID"] = "1"
        before = _norm_rank_dispatch_count()
        disabled = op(sequences, codebooks, 6, 16, None, None, auto)
        assert _norm_rank_dispatch_count() == before

        # `auto` + `fallback="error"` must also observe the fresh value.
        with pytest.raises(RuntimeError, match="GPTQMODEL_QVQ_DISABLE_OCTET_GRID"):
            op(sequences, codebooks, 6, 16, None, None, _pruning_code(mode="auto", fallback="error"))

        del os.environ["GPTQMODEL_QVQ_DISABLE_OCTET_GRID"]
        before = _norm_rank_dispatch_count()
        restored = op(sequences, codebooks, 6, 16, None, None, auto)
        assert _norm_rank_dispatch_count() == before + 1

        assert all(torch.equal(a, b) for a, b in zip(first, disabled))
        assert all(torch.equal(a, b) for a, b in zip(first, restored))
    finally:
        if saved is None:
            os.environ.pop("GPTQMODEL_QVQ_DISABLE_OCTET_GRID", None)
        else:
            os.environ["GPTQMODEL_QVQ_DISABLE_OCTET_GRID"] = saved


def test_qvq_pruning_required_ignores_env_mutation_same_process():
    """`required` ignores the deprecated variable even when it is set mid
    process: dispatch continues and results stay bit-identical."""

    import os

    # W3 batch 6: the W2.5 small-batch cooperative kernel is not band
    # eligible, so use the W3 grid cell that `required` can always serve.
    sequences, codebooks = _norm_rank_case(20260914, 6, 3.0, 2)
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    saved = os.environ.pop("GPTQMODEL_QVQ_DISABLE_OCTET_GRID", None)
    try:
        reference = op(sequences, codebooks, 6, 16, None, None, _pruning_code(mode="required"))
        os.environ["GPTQMODEL_QVQ_DISABLE_OCTET_GRID"] = "1"
        before = _norm_rank_dispatch_count()
        actual = op(sequences, codebooks, 6, 16, None, None, _pruning_code(mode="required"))
        assert _norm_rank_dispatch_count() == before + 1
        assert all(torch.equal(a, b) for a, b in zip(reference, actual))
    finally:
        if saved is None:
            os.environ.pop("GPTQMODEL_QVQ_DISABLE_OCTET_GRID", None)
        else:
            os.environ["GPTQMODEL_QVQ_DISABLE_OCTET_GRID"] = saved
