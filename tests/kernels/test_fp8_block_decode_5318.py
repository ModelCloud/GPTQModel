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
from tests.fp8_test_utils import install_fp8_payload, pack_fp8_module


pytestmark = [pytest.mark.gpu]

BLOCK = (128, 128)


_TRITON_CUDA_AVAILABLE = fp8_block_triton.triton_fp8_available(torch.device("cuda"))
_TRITON_DECODE_AVAILABLE = _TRITON_CUDA_AVAILABLE and fp8_block_triton.triton_fp8_supports_format(
    torch.device("cuda"), "float8_e4m3fn"
)
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


def _all_ones_block_kernel():
    """The #5400 payload: one 128x128 block, every code and the scale at 1."""

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=False,
        format="float8_e4m3fn",
        weight_scale_method="block",
        weight_block_size=BLOCK,
        register_buffers=False,
    )
    kernel.weight = torch.ones((128, 128), dtype=torch.float8_e4m3fn, device="cuda")
    kernel.weight_scale_inv = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    kernel.bias = None
    kernel._scaled_mm_hard_disabled = False
    kernel._native_dequant_disabled = False
    kernel._invalidate_compiled()
    return kernel.eval()


@pytest.mark.skipif(not _TRITON_CUDA_AVAILABLE, reason="requires Triton on NVIDIA CUDA")
def test_block_dispatch_skips_an_operand_the_device_cannot_lower(monkeypatch):
    """#5400: a target without ``fp8e4nv`` must keep the portable block decode.

    Triton's CUDA backend only adds ``fp8e4nv`` -- the type behind
    ``torch.float8_e4m3fn`` -- to a target's supported operands at SM 8.9 and
    later. The dispatcher used to gate on "CUDA and not ROCm" alone, so an
    Ampere target compiled a kernel that raised ``ValueError: type fp8e4nv not
    supported in this architecture`` inside the layer's forward.

    The compute capability is a hardware fact this host cannot change, so the
    test injects it at the Torch boundary. Triton derives its target arch from
    the same probe, so on the pre-fix tree the injected capability reproduces
    the reported compile error end to end instead of only a wrong dispatch.
    """

    module = _all_ones_block_kernel()
    x = torch.zeros((1, 128), dtype=torch.float16, device="cuda")
    x[0, 0] = 0.25
    expected = torch.zeros((1, 128), dtype=torch.float32, device="cuda")
    expected[0, :] = 0.25

    calls: list[str] = []
    real_gemv = fp8_module.fp8_block_gemv
    real_decode = fp8_module.fp8_block_decode
    monkeypatch.setattr(
        fp8_module,
        "fp8_block_gemv",
        lambda *a, **k: calls.append("fused") or real_gemv(*a, **k),
    )
    monkeypatch.setattr(
        fp8_module,
        "fp8_block_decode",
        lambda *a, **k: calls.append("decode") or real_decode(*a, **k),
    )

    real_capability = torch.cuda.get_device_capability
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    try:
        with torch.inference_mode():
            output = module(x)
    finally:
        module._fp8_block_plan_key = None

    assert calls == [], (
        "#5400: the dispatcher selected a Triton FP8 operand SM 8.0 cannot lower: "
        f"{calls}"
    )
    torch.testing.assert_close(output.float(), expected, rtol=2e-3, atol=2e-3)

    if not _TRITON_DECODE_AVAILABLE:
        # This host's own target cannot run the accelerated control either.
        return

    # Control: at the host's real capability the same payload still uses the
    # fused product, so the gate discriminates instead of always falling back.
    monkeypatch.setattr(torch.cuda, "get_device_capability", real_capability)
    assert fp8_block_triton.triton_fp8_supports_format(
        module.weight.device, "float8_e4m3fn"
    )
    module._fp8_block_plan_key = None
    calls.clear()
    with torch.inference_mode():
        accelerated = module(x)
    assert calls == ["fused"]
    torch.testing.assert_close(accelerated.float(), expected, rtol=2e-3, atol=2e-3)


@_DECODE_SKIP
def test_block_decode_tiles_are_the_measured_ones() -> None:
    """Pin the launch tiles the sweep selected, so a drift shows up as a diff."""

    assert fp8_block_triton._gemv_tiles(1, 1024) == (16, 16, 2, 4)
    assert fp8_block_triton._gemv_tiles(16, 4096) == (16, 32, 4, 4)
    assert fp8_block_triton._gemv_tiles(64, 14336) == (64, 64, 4, 4)
    assert fp8_block_triton._decode_tile(128, 128) == (32, 128)


def _range_extreme_kernel(
    *, scale: float, semantics: str, in_features: int = 128, out_features: int = 128
):
    """A 128-wide block payload whose decoded weight is finite but huge.

    Every stored code is 1, so the decoded weight is ``1 / scale`` for the
    inverse convention and ``1 * scale`` for the multiplier one. A scale of
    ``2**-16`` (or ``2**16``) decodes to 65536: finite in FP32 and BF16, but
    above the FP16 maximum of 65504.
    """

    linear = nn.Linear(in_features, out_features, bias=False).eval()
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        format="float8_e5m2",
        weight_scale_method="block",
        weight_block_size=BLOCK,
        register_buffers=False,
    )
    codes = torch.ones((out_features, in_features), dtype=torch.float8_e5m2)
    scale_inv = torch.full(
        (out_features // BLOCK[0], in_features // BLOCK[1]),
        scale,
        dtype=torch.float32,
    )
    install_fp8_payload(kernel, linear, codes, scale_inv)
    kernel.weight_scale_semantics = semantics
    return kernel.to("cuda").eval()


@_DECODE_SKIP
@pytest.mark.parametrize(
    "dtype,decode_expected",
    [
        # FP16 cannot hold the decoded weight, so the narrowing decode must
        # decline and the FP32 fallback must answer.
        (torch.float16, False),
        # BF16 and FP32 keep the FP32 exponent range and stay on the decode.
        (torch.bfloat16, True),
        (torch.float32, True),
    ],
)
@pytest.mark.parametrize(
    "semantics,scale",
    [("inverse", 2.0**-16), ("multiplier", 2.0**16)],
)
def test_block_decode_keeps_an_out_of_fp16_range_payload_finite(
    dtype, decode_expected, semantics, scale
):
    """A finite decoded weight above FP16 range must not be narrowed to inf (#5405).

    The single-pass decode writes the decoded weight in the activation dtype
    before the FP32-reduced GEMM. At 65536 an FP16 store is ``inf`` and the
    zero codes of the weight row make the product ``0 * inf = NaN``, even
    though ``0.25 * 65536 = 16384`` fits FP16. The range gate must send only
    the overflowing FP16 payload to the exact FP32 fallback.
    """

    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")

    module = _range_extreme_kernel(scale=scale, semantics=semantics)
    x = torch.zeros(1, module.in_features, dtype=dtype, device="cuda")
    x[0, 0] = 0.25
    x_flat = x.reshape(-1, module.in_features)

    with torch.inference_mode():
        output = module(x)

    assert torch.isfinite(output).all(), f"non-finite output for {dtype}: {output}"
    # 0.25 * 65536 is exact in every dtype here, so the oracle is exact too.
    reference = _oracle(module, x)
    torch.testing.assert_close(output.to(torch.float64), reference, rtol=0.0, atol=0.0)
    # Only the payload that would narrow to inf leaves the single-pass decode.
    with torch.inference_mode():
        assert module._block_decode_eligible(x_flat, fused=False) == decode_expected


@_DECODE_SKIP
def test_block_decode_range_gate_keeps_the_in_range_fast_path() -> None:
    """The #5405 range gate must not move an in-range payload off the fast path."""

    module = _range_extreme_kernel(scale=1.0, semantics="inverse")
    x = torch.zeros(1, module.in_features, dtype=torch.float16, device="cuda")
    x[0, 0] = 0.25
    x_flat = x.reshape(-1, module.in_features)
    calls = {"gemv": 0, "decode": 0}
    original_gemv = fp8_module.fp8_block_gemv
    original_decode = fp8_module.fp8_block_decode

    def counted_gemv(*args, **kwargs):
        calls["gemv"] += 1
        return original_gemv(*args, **kwargs)

    def counted_decode(*args, **kwargs):
        calls["decode"] += 1
        return original_decode(*args, **kwargs)

    fp8_module.fp8_block_gemv = counted_gemv
    fp8_module.fp8_block_decode = counted_decode
    try:
        with torch.inference_mode():
            assert module._block_decode_eligible(x_flat, fused=False)
            # One row of a divisible 128-wide block still takes the fused
            # W8A16 product, which is the fastest in-range route.
            assert module._block_decode_eligible(x_flat, fused=True)
            output = module(x)
    finally:
        fp8_module.fp8_block_gemv = original_gemv
        fp8_module.fp8_block_decode = original_decode

    assert calls == {"gemv": 1, "decode": 0}, calls
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, torch.full_like(output, 0.25), rtol=0.0, atol=0.0
    )


@_DECODE_SKIP
def test_fused_block_gemv_keeps_the_bias_gradient() -> None:
    """#5402: a trainable bias must not ride the opaque fused epilogue.

    The fused W8A16 product adds the bias inside the Triton kernel, so its
    output has no autograd wrapper. A bias-only training step therefore needs
    the differentiable single-pass decode, while a frozen bias (and any
    no-grad call) keeps the fused product.
    """

    module = _block_kernel(bias=True)
    x = torch.randn(4, module.in_features, device="cuda", dtype=torch.float16)
    x_flat = x.reshape(-1, module.in_features)
    module.weight.requires_grad_(False)
    module.weight_scale_inv.requires_grad_(False)
    module.bias.requires_grad_(True)

    # The root cause: the fused gate looked at the activation, weight and
    # scale, but not at the bias.
    assert not module._block_decode_eligible(x_flat, fused=True)
    assert module._block_decode_eligible(x_flat, fused=False)

    output = module(x)
    assert output.requires_grad
    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    expected = upstream.float().sum(dim=0)
    torch.testing.assert_close(module.bias.grad, expected)

    # A frozen bias is a constant, so the fused product stays available.
    module.bias.requires_grad_(False)
    assert module._block_decode_eligible(x_flat, fused=True)

    # And a trainable bias under no_grad is inference, not training.
    module.bias.requires_grad_(True)
    with torch.no_grad():
        assert module._block_decode_eligible(x_flat, fused=True)
        assert torch.isfinite(module(x)).all()
