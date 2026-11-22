# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Triton activation kernels must address rows past the 2^31 element boundary.

``row * K`` was computed in 32-bit, so once ``rows * K`` exceeded 2^31 the
address wrapped and rows past the boundary read or wrote the wrong memory.
"""

import pytest
import torch


# Every case here exercises a Triton kernel module that imports Triton.
pytest.importorskip("triton")

from gptqmodel.nn_modules.qlinear.qqq_triton import dynamic_quant


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


@requires_cuda
@pytest.mark.cuda
def test_qqq_dynamic_quant_addresses_rows_past_two_to_the_31_elements():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.total_memory < 12 * 1024**3:
        pytest.skip("the overflow regression needs at least 12 GiB of device memory")

    width = 2048
    rows = (2**31 // width) + 1
    x = torch.randn(rows, width, device="cuda", dtype=torch.float16)

    codes, scales = dynamic_quant(x)
    torch.cuda.synchronize()

    assert codes.shape == x.shape
    assert scales.shape == (rows, 1)

    # The last row sits past the 32-bit element boundary. An independent Torch
    # computation of the documented QQQ quantization pins its value.
    for row in (0, rows - 1):
        values = x[row].float()
        scale = values.abs().max() * (1.0 / 127.0)
        expected = torch.clamp(torch.round(values / scale), -128, 127).to(torch.int8)
        assert torch.equal(codes[row], expected)
        assert torch.equal(scales[row, 0], scale)


def _synthetic_4bit_weight(out_features: int, in_features: int, *, blocksize: int = 128):
    """Deterministic packed NF4-style codes over the flat ``(N, K)`` stream.

    The fixture is built directly from the documented layout (even element in
    the high nibble, odd element in the low nibble) so the test does not depend
    on bitsandbytes or on the kernel's own byte packing.
    """

    from types import SimpleNamespace

    elements = out_features * in_features
    indices = torch.arange(elements, device="cuda", dtype=torch.int64)
    codes = ((indices * 7 + 3) % 16).to(torch.uint8)
    packed = torch.zeros((elements + 1) // 2, device="cuda", dtype=torch.uint8)
    packed[:] = codes[0::2] << 4
    packed[: codes[1::2].numel()] |= codes[1::2]
    codebook = torch.arange(16, device="cuda", dtype=torch.float32) - 8.0
    absmax = torch.ones((elements + blocksize - 1) // blocksize, device="cuda")
    state = SimpleNamespace(
        shape=(out_features, in_features),
        dtype=torch.float16,
        absmax=absmax,
        code=codebook,
        blocksize=blocksize,
    )
    return packed, state


def _decode_oracle(packed: torch.Tensor, state) -> torch.Tensor:
    """Decode the packed stream with Torch shifts only (no kernel under test)."""

    out_features, in_features = state.shape
    elements = out_features * in_features
    indices = torch.arange(elements, device=packed.device, dtype=torch.int64)
    byte = packed[indices // 2].long()
    nibble = torch.where(indices % 2 == 0, (byte >> 4) & 0xF, byte & 0xF)
    scale = state.absmax.to(torch.float64).reshape(-1)[indices // int(state.blocksize)]
    return (state.code.to(torch.float64)[nibble] * scale).reshape(out_features, in_features)


def _assert_sampled_rows_match(x, out, dense, dtype):
    rows = torch.tensor([0, 1, x.shape[0] // 2, x.shape[0] - 2, x.shape[0] - 1], device="cuda")
    expected = (x[rows].to(torch.float64) @ dense.T).to(dtype)
    torch.testing.assert_close(out[rows], expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.cuda
def test_bitsandbytes_fused_gemm_addresses_rows_past_two_to_the_31_output_elements():
    """``rows * out_features`` beyond 2^31 must not wrap the output offset."""

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.total_memory < 12 * 1024**3:
        pytest.skip("the overflow regression needs at least 12 GiB of device memory")

    from gptqmodel.nn_modules.qlinear import bitsandbytes_triton as bt

    in_features, out_features = 896, 4864
    rows = 2**31 // out_features + 8
    assert rows * out_features > 2**31  # the launch really crosses the boundary
    assert rows * in_features < 2**31  # only the output offset can wrap here

    packed, state = _synthetic_4bit_weight(out_features, in_features)
    torch.manual_seed(5)
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)

    out = bt.matmul_4bit(x, packed, state, out_dtype=torch.float16)
    torch.cuda.synchronize()

    assert out.shape == (rows, out_features)
    _assert_sampled_rows_match(x, out, _decode_oracle(packed, state), torch.float16)

    # The dispatch gate must reach the fused path at this row count: AUTO on a
    # tuned ROCm family, or the explicit force used for A/B runs elsewhere.
    device = torch.device("cuda")
    if bt._family(device) in bt.tuned_families():
        from gptqmodel.utils.kernel import KernelConfig

        forced = KernelConfig(bitsandbytes_triton=True)
        assert bt.should_use(device, torch.float16, rows, kernel=forced) is True
        if bt.IS_ROCM:
            assert bt.should_use(device, torch.float16, rows) is True


@requires_cuda
@pytest.mark.cuda
def test_bitsandbytes_fused_gemm_addresses_rows_past_two_to_the_31_input_elements():
    """``rows * in_features`` beyond 2^31 must not wrap the activation offset."""

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.total_memory < 8 * 1024**3:
        pytest.skip("the overflow regression needs at least 8 GiB of device memory")

    from gptqmodel.nn_modules.qlinear import bitsandbytes_triton as bt

    in_features, out_features = 4096, 32
    rows = 2**31 // in_features + 8
    assert rows * in_features > 2**31  # the launch really crosses the boundary
    assert rows * out_features < 2**31  # only the activation offset can wrap here

    packed, state = _synthetic_4bit_weight(out_features, in_features)
    torch.manual_seed(6)
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)

    out = bt.matmul_4bit(x, packed, state, out_dtype=torch.float16)
    torch.cuda.synchronize()

    assert out.shape == (rows, out_features)
    _assert_sampled_rows_match(x, out, _decode_oracle(packed, state), torch.float16)
