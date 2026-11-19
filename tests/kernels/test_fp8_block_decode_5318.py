# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Block-scaled FP8 decode must read the payload once, not twice, per forward.

``TorchFP8Linear._forward_lean_scaled_matmul`` built a block-scaled weight with
two full-weight passes: ``weight.to(activation_dtype)`` (read 1 B, write 2 B per
element) and then an in-place block scale (read 2 B, write 2 B). A 1 B/element
format therefore paid ~9 B of traffic per weight element before the GEMM. The
row/tensor epilogue of #5173 cannot cover this layout, because a block scale
varies along both axes of the reduction.

The regression pins three things:

1. the forward takes the single-pass (or fused) path, and the path it took is
   observable rather than inferred;
2. its output matches an independent FP64 oracle and the pre-fix portable
   decode, so the fix is a traffic change and not an arithmetic change;
3. the forward keeps no decoded weight after it returns, so the single-pass
   temporary cannot become a persistent dense cache.

Detection evidence: on the pre-fix tree the path assertion fails because no
block-decode entry point exists and ``_forward_dequant_matmul`` runs the
two-pass cast/scale; the oracle and storage assertions pass on both trees.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
import torch.nn as nn

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module
from gptqmodel.utils import fp8_block_triton
from tests.fp8_test_utils import pack_fp8_module


pytestmark = [pytest.mark.gpu]

BLOCK = (128, 128)


_TRITON_DECODE_AVAILABLE = fp8_block_triton.triton_fp8_available(torch.device("cuda"))
_DECODE_SKIP = pytest.mark.skipif(
    not _TRITON_DECODE_AVAILABLE,
    reason="block-scaled FP8 Triton decode is not available on this device",
)


def _block_kernel(*, in_features: int = 512, out_features: int = 512, bias: bool = False):
    torch.manual_seed(11)
    linear = nn.Linear(in_features, out_features, bias=bias).eval()
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.02)
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        format="float8_e4m3fn",
        weight_scale_method="block",
        weight_block_size=BLOCK,
        register_buffers=False,
    )
    pack_fp8_module(kernel, linear)
    return kernel.to("cuda").eval()


def _module_fp8_dtype(kernel) -> torch.dtype:
    return kernel.weight.dtype


def _oracle(kernel, x: torch.Tensor) -> torch.Tensor:
    """Independent FP64 decode, never calling the code under test."""

    codes = kernel.weight.to(torch.float64)
    scale = kernel.weight_scale_inv.to(torch.float64)
    block_rows, block_cols = kernel.weight_block_size
    rows, cols = codes.shape
    blocks = codes.reshape(rows // block_rows, block_rows, cols // block_cols, block_cols)
    if kernel.weight_scale_semantics == "multiplier":
        decoded = blocks * scale[:, None, :, None]
    else:
        decoded = blocks / scale[:, None, :, None]
    output = x.to(torch.float64) @ decoded.reshape(rows, cols).t()
    if kernel.bias is not None:
        output = output + kernel.bias.to(torch.float64)
    return output


class _FullWeightInPlaceScale(torch.overrides.TorchFunctionMode):
    """Record in-place arithmetic applied to a decoded-weight-sized tensor.

    The pre-fix block path materialised the whole weight in the activation
    dtype and then scaled it in place. That mutation is the second full-weight
    pass this regression is about, and it is observable without knowing which
    function the layer called: the module never scales a decoded weight in
    place on the fixed path.
    """

    def __init__(self, numel: int, fp8_dtype: torch.dtype):
        super().__init__()
        self.numel = numel
        self.fp8_dtype = fp8_dtype
        self.hits: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = getattr(func, "__name__", "")
        if name in {"mul_", "div_", "add_", "sub_"} and args:
            target = args[0]
            # A decoded weight is one element per packed code; the packed
            # payload itself is FP8 and is never the target of a scale op.
            if (
                isinstance(target, torch.Tensor)
                and target.numel() == self.numel
                and target.dtype != self.fp8_dtype
            ):
                self.hits.append(name)
        return func(*args, **kwargs)


@_DECODE_SKIP
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rows", [1, 3, 16, 64, 128])
def test_block_decode_matches_oracle_and_avoids_the_two_pass_cast(dtype, bias, rows):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")

    module = _block_kernel(bias=bias)
    x = torch.randn(rows, module.in_features, device="cuda", dtype=dtype)
    probe = _FullWeightInPlaceScale(module.weight.numel(), _module_fp8_dtype(module))

    with torch.inference_mode(), probe:
        output = module(x)

    # The fix reads the payload once; the pre-fix cast-then-scale mutates a
    # full decoded weight in place, which is the extra pass this pins.
    assert probe.hits == [], f"forward ran a full-weight in-place scale: {probe.hits}"
    assert output.shape == x.shape[:-1] + (module.out_features,)
    assert output.dtype == dtype

    expected = _oracle(module, x)
    relative = (output.to(torch.float64) - expected).abs().max() / expected.abs().max()
    # The activation dtype rounding dominates: half an FP16 ulp is ~5e-4 and
    # half a BF16 ulp is ~4e-3, which is the limit the numerical skill allows.
    limit = 2e-3 if dtype is torch.float16 else 8e-3
    assert float(relative) <= limit


@_DECODE_SKIP
def test_block_decode_and_portable_decode_agree():
    """The traffic fix must not change arithmetic on either block path."""

    module = _block_kernel(in_features=512, out_features=384, bias=True)
    x = torch.randn(9, module.in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        accelerated = module(x)
        # The same payload through the portable cast/scale path, by making the
        # Triton decode unavailable for one call only.
        original = fp8_module.triton_fp8_available
        fp8_module.triton_fp8_available = lambda *args, **kwargs: False
        try:
            module._fp8_block_plan_key = None
            reference = module(x)
        finally:
            fp8_module.triton_fp8_available = original
            module._fp8_block_plan_key = None

    # The fused product accumulates a different reduction order, so compare at
    # the activation dtype's rounding limit rather than bitwise.
    torch.testing.assert_close(accelerated, reference, rtol=4e-3, atol=4e-3)


@_DECODE_SKIP
def test_block_decode_keeps_no_decoded_weight_after_forward():
    """The single-pass temporary must not become a persistent dense cache."""

    module = _block_kernel(in_features=512, out_features=512, bias=False)
    x = torch.randn(1, module.in_features, device="cuda", dtype=torch.float16)
    references: list[weakref.ref] = []
    original = fp8_module.fp8_block_decode

    def tracked_decode(codes, scale, **kwargs):
        decoded = original(codes, scale, **kwargs)
        references.append(weakref.ref(decoded))
        return decoded

    # Run the single-pass decode directly: that is the branch that allocates a
    # decoded weight at all.
    fp8_module.fp8_block_decode = tracked_decode
    try:
        with torch.inference_mode():
            output = module._forward_block_decode(x)
    finally:
        fp8_module.fp8_block_decode = original

    assert references, "the single-pass decode was not exercised"
    assert output.shape == (1, module.out_features)
    del output
    gc.collect()
    assert all(ref() is None for ref in references), "the decoded weight outlived its forward"
    # Nothing decoded is retained on the module either.
    for name, value in vars(module).items():
        if isinstance(value, torch.Tensor) and value.dtype in (torch.float16, torch.bfloat16):
            assert value.shape != (module.out_features, module.in_features), (
                f"forward retained a dense decoded weight in `{name}`"
            )


@_DECODE_SKIP
def test_block_decode_declines_unsupported_payloads(monkeypatch) -> None:
    """Non-divisible, padded and non-Triton payloads keep the portable decode."""

    module = _block_kernel(in_features=512, out_features=512)
    x = torch.randn(4, module.in_features, device="cuda", dtype=torch.float16)
    x_flat = x.reshape(-1, module.in_features)

    with torch.inference_mode():
        assert module._block_decode_eligible(x_flat, fused=False)

        monkeypatch.setattr(fp8_module, "triton_fp8_available", lambda *a, **k: False)
        module._fp8_block_plan_key = None
        assert not module._block_decode_eligible(x_flat, fused=False)
        module._fp8_block_plan_key = None
        monkeypatch.undo()

        # A padded block-scale tensor is not the exact per-block layout.
        monkeypatch.setattr(module, "_block_scale_matches_weight", lambda: False)
        module._fp8_block_plan_key = None
        assert not module._block_decode_eligible(x_flat, fused=False)
        module._fp8_block_plan_key = None
        monkeypatch.undo()

        # Non-contiguous activations have no kernel launch contract here.
        strided = torch.randn(module.in_features, 4, device="cuda", dtype=torch.float16).t()
        assert strided.shape == x_flat.shape and not strided.is_contiguous()
        assert not module._block_decode_eligible(strided, fused=False)

        # A row/tensor payload never enters the block decode.
        module.weight_scale_method = "row"
        module._fp8_block_plan_key = None
        assert not module._block_decode_eligible(x_flat, fused=False)
        module.weight_scale_method = "block"
        module._fp8_block_plan_key = None
        assert module._block_decode_eligible(x_flat, fused=False)

        # A checkpoint whose block scale is stored narrower than FP32 still
        # decodes; the forward widens that one small tensor.
        module.weight_scale_inv = module.weight_scale_inv.half()
        module._fp8_block_plan_key = None
        assert module._block_decode_eligible(x_flat, fused=False)


@_DECODE_SKIP
def test_block_decode_tiles_are_the_measured_ones() -> None:
    """Pin the launch tiles the sweep selected, so a drift shows up as a diff."""

    assert fp8_block_triton._gemv_tiles(1, 1024) == (16, 16, 2, 4)
    assert fp8_block_triton._gemv_tiles(16, 4096) == (16, 32, 4, 4)
    assert fp8_block_triton._gemv_tiles(64, 14336) == (64, 64, 4, 4)
    assert fp8_block_triton._decode_tile(128, 128) == (32, 128)
