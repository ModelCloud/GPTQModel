# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Dense-reference, edge-case, stream, and free-threaded tests for QVQ CUDA."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gptqmodel.looper.qvq_output_alignment import _FixedTrellisAlignmentLinear
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear, QVQReferenceLinear
from gptqmodel.quantization.qvq import (
    QVQQuantizationTelemetry,
    _batched_v2_banked_viterbi_quantize,
    _canonical_qvq_codebook,
    _canonical_qvq_v4_banks,
    batched_viterbi_quantize,
    block_ldlq_inner,
    block_ldlq_inner_banked,
    block_ldlq_inner_banked_candidates,
    optimize_qvq_output_channel_scales,
    pack_qvq_bank_ids,
    pack_qvq_binary_bank_ids,
    pack_trellis_states,
    quantize_qvq_linear,
    qvq_proxy_loss,
    reconstruct_qvq_inner_weight,
    unpack_trellis_states,
    yaqa_proxy_loss,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    canonical_pgc16_levels,
    pgc16_codebook,
    pgc16_codebook_v2_bank,
    pgc16_codebook_v4,
)
from gptqmodel.quantization.qvq_rates import qvq_transition_bits, qvq_words_per_tile
from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b
from gptqmodel.quantization.rotation.hadamard_utils import (
    matmul_hadU,
    matmul_hadU_stable,
)
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_cuda import (
    QVQ_CUDA_BITS,
    _qvq_cuda_viterbi_v2_segment_g_op,
    qvq_cuda_gemv,
    qvq_cuda_hadamard,
    qvq_cuda_supported,
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

    assert not torch.isfinite(historical).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=16.0)


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

    assert observed == [(torch.float16, True), (torch.bfloat16, False)]
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
