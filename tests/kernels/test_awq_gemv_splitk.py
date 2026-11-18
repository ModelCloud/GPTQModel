# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ GEMV kernel coverage for issue #4054 items 1 and 2.

Both kernels are checked against an independent float64 dequantize-then-matmul
oracle built from the packed codes, scales, and zero offsets. Neither test uses
the kernel under test as its own reference.
"""

from pathlib import Path

import pytest
import torch

from gptqmodel import CalibrationConfig
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import pack_intweight
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq_columns
from gptqmodel.utils.awq import (
    awq_fast_gemm_forward_prefill,
    awq_gemmv2_forward,
    awq_runtime_available,
)
from gptqmodel.utils.gemv import calculate_zeros_width


GROUP_SIZE = 128
OUT_FEATURES = 128
REL_TOL = 2e-3
QWEN25_DENSE_PATH = Path("/monster/data/model/Qwen2.5-0.5B-Instruct")

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AWQ GEMV kernels require CUDA"
)


def _require_runtime():
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA JIT extension entrypoint unavailable")


def _rel_error(actual: torch.Tensor, oracle: torch.Tensor) -> float:
    actual64 = actual.detach().float().cpu().double()
    numerator = (actual64 - oracle).norm()
    denominator = oracle.norm()
    if denominator == 0:
        return numerator.item()
    return (numerator / denominator).item()


def _build_gemv_fast_buffers(in_features: int, out_features: int, seed: int = 0):
    """Build GEMV_FAST packed buffers from explicit int4 codes."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_groups = in_features // GROUP_SIZE
    zeros_width = calculate_zeros_width(in_features, GROUP_SIZE, 8)

    codes = torch.randint(0, 16, (out_features, in_features), generator=generator, dtype=torch.int32)
    scales = (torch.rand(num_groups, out_features, generator=generator, dtype=torch.float32) * 0.5 + 0.75).half()
    zeros = torch.randint(0, 16, (num_groups, out_features), generator=generator, dtype=torch.int32)
    offsets = (-scales.float() * zeros.float()).half()

    qscales = torch.zeros((out_features, zeros_width * 8), dtype=torch.float16)
    qscales[:, :num_groups] = scales.T
    qzeros = torch.zeros((out_features, zeros_width * 8), dtype=torch.float16)
    qzeros[:, :num_groups] = offsets.T

    qweight = pack_intweight(codes.contiguous(), interleave=4, kstride=64)
    return codes, scales, offsets, qweight, qscales.T.contiguous(), qzeros.T.contiguous()


def _gemv_fast_oracle(x: torch.Tensor, codes, scales, offsets) -> torch.Tensor:
    out_features, in_features = codes.shape
    group_index = torch.arange(in_features) // GROUP_SIZE
    weight = codes.double() * scales.double()[group_index].T + offsets.double()[group_index].T
    return x.detach().float().cpu().double() @ weight.T


def _build_gemv_buffers(in_features: int, out_features: int, seed: int = 1):
    """Build GEMV (gemmv2) packed buffers from explicit int4 codes."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_groups = in_features // GROUP_SIZE
    zeros_width = calculate_zeros_width(in_features, GROUP_SIZE, 8)

    codes = torch.randint(0, 16, (out_features, in_features), generator=generator, dtype=torch.int32)
    scales = (torch.rand(out_features, num_groups, generator=generator, dtype=torch.float32) * 0.5 + 0.75).half()
    zeros = torch.randint(0, 16, (out_features, num_groups), generator=generator, dtype=torch.int32)

    qweight = pack_awq_columns(codes, 4)
    qscales = torch.zeros((out_features, zeros_width * 8), dtype=torch.float16)
    qscales[:, :num_groups] = scales
    padded_zeros = torch.zeros((out_features, zeros_width * 8), dtype=torch.int32)
    padded_zeros[:, :num_groups] = zeros
    qzeros = pack_awq_columns(padded_zeros, 4)
    return codes, scales, zeros, qweight, qscales, qzeros


def _gemv_oracle(x: torch.Tensor, codes, scales, zeros) -> torch.Tensor:
    in_features = codes.shape[1]
    group_index = torch.arange(in_features) // GROUP_SIZE
    scale = scales.double()[:, group_index]
    weight = codes.double() * scale - zeros.double()[:, group_index] * scale
    return x.detach().float().cpu().double() @ weight.T


def _nan_tail_input(rows: int, in_features: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (view, source) where the view's storage ends in NaN sentinels.

    The view is contiguous and keeps the kernel's ``in_features`` row stride, so
    any read past the final row lands in the NaN tail of the same storage.
    """
    torch.manual_seed(seed)
    source = torch.randn(rows, in_features, dtype=torch.float16, device="cuda")
    flat = torch.empty(rows * in_features + 4096, dtype=torch.float16, device="cuda")
    flat.fill_(float("nan"))
    view = flat[: rows * in_features].view(rows, in_features)
    view.copy_(source)
    return view, source


@CUDA_REQUIRED
@pytest.mark.parametrize("in_features", [896, 1024, 1152, 1408, 4864])
@pytest.mark.parametrize("rows", [1, 7, 8, 16, 32, 33, 100])
def test_gemv_fast_prefill_computes_every_k_tile(in_features, rows):
    """Issue #4054 item 1: an odd K/128 tile count must still cover all tiles.

    ``gemm_forward_cuda_prefill`` uses SPLITK=2 with CTA_K=128 for M <= 32. The
    old split offsets skipped the tail tiles whenever K/128 was odd, so the
    output lost whole K ranges (K=896, 1152, 1408) while K=1024 and K=4864
    happened to divide evenly.
    """
    _require_runtime()
    codes, scales, offsets, qweight, scale_buf, zero_buf = _build_gemv_fast_buffers(
        in_features, OUT_FEATURES
    )
    qweight = qweight.cuda()
    scale_buf = scale_buf.cuda()
    zero_buf = zero_buf.cuda()

    torch.manual_seed(0)
    x = torch.randn(rows, in_features, dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        out = awq_fast_gemm_forward_prefill(x, qweight, scale_buf, zero_buf)

    oracle = _gemv_fast_oracle(x, codes, scales, offsets)
    assert out.shape == (rows, OUT_FEATURES)
    assert torch.isfinite(out).all()
    error = _rel_error(out, oracle)
    assert error <= REL_TOL, (
        f"K={in_features} M={rows} relative error {error:.3e} exceeds {REL_TOL:.0e}; "
        "the prefill split-K path dropped K tiles"
    )


@CUDA_REQUIRED
@pytest.mark.parametrize("in_features", [896, 1152, 1408])
@pytest.mark.parametrize("rows", [9, 16, 64, 128])
def test_gemv_fast_prefill_ignores_nonfinite_memory_after_input(in_features, rows):
    """The fixed split must not touch the activation buffer past the last row."""
    _require_runtime()
    codes, scales, offsets, qweight, scale_buf, zero_buf = _build_gemv_fast_buffers(
        in_features, OUT_FEATURES
    )
    qweight = qweight.cuda()
    scale_buf = scale_buf.cuda()
    zero_buf = zero_buf.cuda()

    view, source = _nan_tail_input(rows, in_features)
    with torch.inference_mode():
        out = awq_fast_gemm_forward_prefill(view, qweight, scale_buf, zero_buf)

    assert torch.isfinite(out).all(), "prefill read non-finite memory past the activation row"
    assert _rel_error(out, _gemv_fast_oracle(source, codes, scales, offsets)) <= REL_TOL


@CUDA_REQUIRED
@pytest.mark.parametrize("in_features", [896, 1024, 1152, 1408, 4864])
@pytest.mark.parametrize("rows", [9, 16, 33, 128])
def test_gemmv2_matches_fp64_oracle_across_k_tiles(in_features, rows):
    """Issue #4054 item 2: GEMMV2 (M > 8) must read only in-range k-tiles."""
    _require_runtime()
    codes, scales, zeros, qweight, scale_buf, zero_buf = _build_gemv_buffers(
        in_features, OUT_FEATURES
    )
    qweight = qweight.cuda()
    scale_buf = scale_buf.cuda()
    zero_buf = zero_buf.cuda()

    torch.manual_seed(0)
    x = torch.randn(rows, in_features, dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        out = awq_gemmv2_forward(x, qweight, scale_buf, zero_buf, GROUP_SIZE, 8)

    assert out.shape == (rows, OUT_FEATURES)
    assert torch.isfinite(out).all()
    error = _rel_error(out, _gemv_oracle(x, codes, scales, zeros))
    assert error <= REL_TOL, f"K={in_features} M={rows} relative error {error:.3e}"


@CUDA_REQUIRED
@pytest.mark.parametrize("in_features", [896, 1152, 1408, 4864])
@pytest.mark.parametrize(
    "rows,split_k_iters",
    [(9, 8), (16, 8), (33, 8), (64, 8), (128, 8), (9, 5), (9, 3), (9, 2), (9, 1)],
)
def test_gemmv2_never_reads_past_the_activation_row(in_features, rows, split_k_iters):
    """Issue #4054 item 2: place NaN immediately after the input storage.

    ``gemmv2`` compared a k-tile index against the column count when it trimmed
    the last split, so splits whose last tile was out of range read the bytes
    after the row. For the final row that memory is past the tensor, and a
    ``0 * NaN`` from the zero-padded scale poisoned the output row.
    """
    _require_runtime()
    codes, scales, zeros, qweight, scale_buf, zero_buf = _build_gemv_buffers(
        in_features, OUT_FEATURES
    )
    qweight = qweight.cuda()
    scale_buf = scale_buf.cuda()
    zero_buf = zero_buf.cuda()

    view, source = _nan_tail_input(rows, in_features)
    with torch.inference_mode():
        out = awq_gemmv2_forward(view, qweight, scale_buf, zero_buf, GROUP_SIZE, split_k_iters)

    assert torch.isfinite(out).all(), (
        f"K={in_features} M={rows} split_k={split_k_iters} read non-finite memory "
        "past the activation row"
    )
    assert _rel_error(out, _gemv_oracle(source, codes, scales, zeros)) <= REL_TOL


@pytest.mark.slow
@pytest.mark.model
@CUDA_REQUIRED
@pytest.mark.skipif(not QWEN25_DENSE_PATH.exists(), reason="Qwen2.5-0.5B-Instruct not available")
def test_real_qwen25_awq_gemv_greedy_decode_stays_finite(tmp_path):
    """Issue #4054 item 2: a real 896/4864 model must decode without NaN rows.

    The checkpoint is quantized to ``FORMAT.GEMV`` so ``AwqGEMVLinear`` (and its
    ``gemmv2`` M > 8 path) runs the same hidden width as Qwen2.5-0.5B. A prompt
    longer than 8 tokens reaches ``gemmv2`` during prefill, and greedy decoding
    checks every returned score tensor for non-finite values.
    """
    _require_runtime()
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
    from gptqmodel.quantization import FORMAT, METHOD

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=GROUP_SIZE,
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMV,
    )
    model = GPTQModel.load(
        str(QWEN25_DENSE_PATH),
        quantize_config=quantize_config,
        backend=BACKEND.AWQ_GEMV,
        device="cuda",
        dtype=torch.float16,
    )
    torch.manual_seed(0)
    calibration = [
        {
            "input_ids": torch.randint(3, 1000, (1, 32)),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(4)
    ]
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=1))
    model.save(str(tmp_path))
    del model

    model = GPTQModel.load(str(tmp_path), backend=BACKEND.AWQ_GEMV, device="cuda")
    assert any(isinstance(module, AwqGEMVLinear) for _, module in model.named_modules())

    torch.manual_seed(0)
    input_ids = torch.randint(3, 1000, (1, 12), device="cuda")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    assert generated.scores, "greedy decode returned no scores"
    assert all(bool(torch.isfinite(score).all()) for score in generated.scores), (
        "AWQ GEMV greedy decode produced non-finite logits"
    )


def _wide_output_sentinel_input(rows: int, in_features: int, tail: int = 8):
    """Activations whose only nonzero rows are the final ``tail`` rows.

    Every other row is exactly zero, so a missed store is the only way the tail
    of the output can disagree with the oracle. K stays small so the 2^31
    output-element shape fits in a single allocation without a wide input.
    """
    torch.manual_seed(0)
    x = torch.zeros(rows, in_features, dtype=torch.float16, device="cuda")
    x[-tail:] = torch.randn(tail, in_features, dtype=torch.float16, device="cuda")
    return x


def _wide_output_geometry(tail: int = 8):
    """Return (rows, out_features, in_features, tail) just above 2^31 elements.

    ``rows * out_features`` is 2^31 + 4096 with K = 256, so both the output
    store and the A-side row offset cross the 32-bit boundary while the
    activation itself stays 128 MiB.
    """
    in_features = 256
    out_features = 8192
    rows = 2**31 // out_features + (4096 + out_features - 1) // out_features
    assert rows * out_features > 2**31
    return rows, out_features, in_features, tail


@CUDA_REQUIRED
@pytest.mark.parametrize("split_k_iters", [1, 2])
def test_gemmv2_stores_every_row_above_int32_element_count(split_k_iters):
    """Issue #5091: GEMMV2 (M > 8) must store rows past 2^31 output elements.

    ``row_offset * OC`` was a 32-bit product, so once ``rows * out_features``
    passed 2^31 the store wrapped to a negative element offset: the sm_80+ path
    silently dropped the tail rows and the older arch build faulted with
    ``cudaErrorIllegalAddress``. Only the sampled tail rows need an oracle, so
    the fixture stays inside one output allocation.
    """
    _require_runtime()
    rows, out_features, in_features, tail = _wide_output_geometry()
    codes, scales, zeros, qweight, scale_buf, zero_buf = _build_gemv_buffers(
        in_features, out_features
    )
    qweight = qweight.cuda()
    scale_buf = scale_buf.cuda()
    zero_buf = zero_buf.cuda()

    x = _wide_output_sentinel_input(rows, in_features, tail)
    with torch.inference_mode():
        out = awq_gemmv2_forward(x, qweight, scale_buf, zero_buf, GROUP_SIZE, split_k_iters)
        torch.cuda.synchronize()

    assert out.shape == (rows, out_features)
    assert torch.isfinite(out[-tail:]).all()
    error = _rel_error(out[-tail:], _gemv_oracle(x[-tail:], codes, scales, zeros))
    assert error <= REL_TOL, (
        f"GEMMV2 split_k={split_k_iters} relative error {error:.3e} above 2^31 output "
        "elements; the tail rows were not stored"
    )


@CUDA_REQUIRED
def test_gemv_fast_prefill_stores_every_row_above_int32_element_count():
    """Issue #5091: GEMV_FAST prefill must store rows past 2^31 elements.

    ``int index = write_row * N + ...`` wrapped in the T2 prefill kernel, so a
    long-context prefill aborted the CUDA context with an illegal memory
    access. The tail rows are the only nonzero inputs, so a dropped store is
    visible without materializing a wide float32 oracle.
    """
    _require_runtime()
    rows, out_features, in_features, tail = _wide_output_geometry()
    codes, scales, offsets, qweight, scale_buf, zero_buf = _build_gemv_fast_buffers(
        in_features, out_features
    )
    qweight = qweight.cuda()
    scale_buf = scale_buf.cuda()
    zero_buf = zero_buf.cuda()

    x = _wide_output_sentinel_input(rows, in_features, tail)
    with torch.inference_mode():
        out = awq_fast_gemm_forward_prefill(x, qweight, scale_buf, zero_buf)
        torch.cuda.synchronize()

    assert out.shape == (rows, out_features)
    assert torch.isfinite(out[-tail:]).all()
    error = _rel_error(out[-tail:], _gemv_fast_oracle(x[-tail:], codes, scales, offsets))
    assert error <= REL_TOL, (
        f"GEMV_FAST prefill relative error {error:.3e} above 2^31 output elements; "
        "the tail rows were not stored"
    )


@CUDA_REQUIRED
def test_gemmv2_reads_activations_above_int32_row_stride_product():
    """Issue #5091: GEMMV2 must index A rows once rows * in_features > 2^31.

    ``A_ptr`` multiplied the row by ``IC`` in 32 bits, so a long prefill read
    the wrong activation row for every tile whose ``row * IC`` crossed 2^31.
    The output stays small here (N = 128) and only the final rows are nonzero,
    so a wrapped A offset changes the sampled rows without a wide output.
    """
    _require_runtime()
    in_features = 8192
    out_features = 128
    tail = 8
    rows = 2**31 // in_features + (4096 + in_features - 1) // in_features + 1
    assert rows * in_features > 2**31

    codes, scales, zeros, qweight, scale_buf, zero_buf = _build_gemv_buffers(
        in_features, out_features
    )
    qweight = qweight.cuda()
    scale_buf = scale_buf.cuda()
    zero_buf = zero_buf.cuda()

    x = _wide_output_sentinel_input(rows, in_features, tail)
    with torch.inference_mode():
        out = awq_gemmv2_forward(x, qweight, scale_buf, zero_buf, GROUP_SIZE, 1)
        torch.cuda.synchronize()

    assert out.shape == (rows, out_features)
    assert torch.isfinite(out[-tail:]).all()
    error = _rel_error(out[-tail:], _gemv_oracle(x[-tail:], codes, scales, zeros))
    assert error <= REL_TOL, (
        f"GEMMV2 relative error {error:.3e} once rows * in_features passed 2^31; "
        "the activation row offset wrapped"
    )
