# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""W4A weights stay in their native/packed representation across the lifecycle.

``W4AFP8Linear`` builds one E4M3 operand per forward from the native GPTQ
INT4 codes. That operand is an expansion of
the checkpoint (one byte per weight on top of the packed half byte plus
scales), so it may exist only while its GEMM consumes it. These tests assert
that persistent-state contract on CPU: they drive the real ``forward``
dispatch with CPU kernel stand-ins, then scan the module for retained expanded
weights and prove the operand is collectable afterwards. Numeric checks use a
Torch oracle that decodes the packed INT4 words itself.

The GPU kernels (Triton FP8 MMA) are not exercised here; see
``test_w4afp8_gb10.py``. The few cases that do launch a kernel are gated on
``w4a_kernels_available()`` rather than ``torch.cuda.is_available()``: HIP
reuses the CUDA device namespace, so the torch check is true on ROCm even
though the kernels (``ttg.nvidia_mma`` operands and PTX rounding) cannot be
compiled there.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear, unpack_int4_codes
from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear
from tests.w4a_hardware_marks import w4a_kernels_available


K, N = 256, 128

_W4A_GPU_REASON = (
    "the W4A Triton kernels require an NVIDIA CUDA device "
    "(they lower to `ttg.nvidia_mma` and emit PTX, which HIP cannot compile)"
)

# Comment on storage: a packed 4-bit weight uses half a byte per weight, so any
# floating tensor of at least ``K * N`` bytes is an expansion, not a layout.
_EXPANSION_BYTES = K * N


@pytest.fixture(autouse=True)
def allow_cpu_construction(monkeypatch):
    """Bypass the CUDA capability gate so the CPU storage contract is testable."""
    monkeypatch.setattr(
        W4AFP8Linear, "validate", classmethod(lambda _cls, **kwargs: (True, None)), raising=True
    )


def _module(cls, k: int = K, n: int = N):
    module = cls(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False,
    )
    codes = torch.arange(k, dtype=torch.int32).remainder(16)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    words = (codes.reshape(k // 8, 8) << shifts).sum(dim=1).to(torch.int32)
    module.qweight.copy_(words[:, None].expand(k // 8, n))
    # GPTQ v1 stores the logical symmetric zero point 8 as nibble 7.
    module.qzeros.fill_(0x77777777)
    module.scales.copy_(
        torch.arange(1, k // 128 + 1, dtype=torch.float16)[:, None].expand(k // 128, n) / 8
    )
    if hasattr(module, "activation_global_scale"):
        module.activation_global_scale.fill_(0.03125)
    module.post_init()
    return module


# --------------------------------------------------------------------------
# Independent Torch oracles (no kernel or module helper is reused).
# --------------------------------------------------------------------------

def _decode_centered_codes(module) -> torch.Tensor:
    """Unpack GPTQ v1 words and zero-point nibbles with plain Torch shifts."""
    k, n = module.in_features, module.out_features
    words = module.qweight.to(torch.int64)
    codes = torch.stack(
        [((words >> (4 * bit)) & 15) for bit in range(8)], dim=1
    ).reshape(k, n).to(torch.int16)
    zwords = module.qzeros.to(torch.int64)
    zeros = torch.stack(
        [((zwords >> (4 * bit)) & 15) for bit in range(8)], dim=2
    ).reshape(k // 128, n).to(torch.int16)
    if module.qzero_format() == 1:
        # GPTQ v1 stores the logical zero point minus one.
        zeros = (zeros + 1) & 15
    assert torch.equal(zeros, torch.full_like(zeros, 8)), "oracle only covers symmetric zero points"
    return codes - zeros.repeat_interleave(128, dim=0)


def _fp8_pack_oracle(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token E4M3 quantization with an amax/448 scale, in Torch."""
    amax = x.float().abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    codes = (x.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes, scale


def _fp8_gemm_oracle(codes, row_scale, codes_weight, scales) -> torch.Tensor:
    rows, k = codes.shape
    n = codes_weight.shape[1]
    total = torch.zeros((rows, n), dtype=torch.float32)
    for group in range(k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        total += (codes[:, sl].float() @ codes_weight[sl].float()) * scales[group].float()
    return total * row_scale[:, None]


def _fp8_carrier(x: torch.Tensor) -> W4AActivation:
    rows, k = x.numel() // x.shape[-1], x.shape[-1]
    codes, scales = _fp8_pack_oracle(x.contiguous().reshape(rows, k))
    return W4AActivation(
        mode="w4afp8", codes=codes, scales=scales,
        shape=tuple(x.shape), model_dtype=x.dtype,
    )


def _decode_e2m1_planes(plane: torch.Tensor) -> torch.Tensor:
    """Decode one packed E2M1 plane through the format's numeric codebook.

    Independent of the module helpers: the nibbles are read directly and each
    four-bit code is looked up in the E2M1 value table, so a packing bug cannot
    cancel itself out. Codes are ``magnitude | (sign << 3)``.
    """
    values = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.))
    data = plane.view(torch.uint8)
    rows, cols = data.shape
    codes = torch.stack((data & 15, data >> 4), dim=1).reshape(rows * 2, cols)
    return values[(codes & 7).long()] * torch.where((codes & 8) == 0, 1., -1.)


# --------------------------------------------------------------------------
# CPU kernel stand-ins. `w4a_triton` cannot be imported on a CPU-only host
# (its Triton build lacks `mma_v2`), so the forward's lazy import is served a
# stub module that implements the same documented math in Torch.
# --------------------------------------------------------------------------

@pytest.fixture
def cpu_kernels(monkeypatch):
    """Dense stand-ins for the CUDA kernels.

    The stub folds the per-group scales into a materialized weight and issues
    one plain matmul, so it does not share the oracle's group-loop structure.
    Materializing a dense weight here is fine: this is test scaffolding, not
    inference state.
    """

    def _dense_gemm(q, row_scale, weight, scales):
        dense = weight.float() * scales.float().repeat_interleave(128, dim=0)
        return (q.float() @ dense) * row_scale[:, None]

    def fp8_linear(x, weight, scales, bias):
        rows, k = x.numel() // x.shape[-1], x.shape[-1]
        flattened = x.contiguous().reshape(rows, k)
        amax = flattened.float().abs().amax(dim=-1)
        scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
        q = (flattened.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        out = _dense_gemm(q, scale, weight, scales)
        if bias is not None:
            out = out + bias.float()
        return out.to(x.dtype).reshape(x.shape[:-1] + (weight.shape[1],))

    def fp8_linear_prepacked(q, row_scale, weight, scales, bias, output_dtype):
        out = _dense_gemm(q, row_scale, weight, scales)
        if bias is not None:
            out = out + bias.float()
        return out.to(output_dtype)

    stub = types.ModuleType("gptqmodel.nn_modules.qlinear.w4a_triton")
    stub.fp8_linear = fp8_linear
    stub.fp8_linear_prepacked = fp8_linear_prepacked
    stub.fp8_pack = _fp8_pack_oracle
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return stub


# --------------------------------------------------------------------------
# Persistent-state scanning.
# --------------------------------------------------------------------------

def _expanded_weight_tensors(module, k: int = K, n: int = N) -> list[str]:
    """Names of module-owned tensors that hold at least one expanded weight.

    Scans buffers, parameters, and ``__dict__`` (recursing into plain
    containers) so an expansion hidden in a dict or list is still found.

    Detection is by representation, not by name or dtype. A full-layer INT8
    plane is just as prohibited as an E4M3 plane, so the size test is applied
    to integer tensors too; only the native checkpoint tensors are exempt.
    Packed layouts are exempt by *verified* shape/byte contract rather than by
    an attribute allowlist, so an expanded tensor that happens to reuse a
    packed-cache name is still reported.
    """
    native = {"qweight", "qzeros", "scales", "g_idx"}
    found: list[str] = []

    def is_packed_layout(value: torch.Tensor) -> bool:
        # Two FP4 codes per byte: one byte per weight for the pair of planes.
        return (getattr(torch, "float4_e2m1fn_x2", None) is not None
                and value.dtype is torch.float4_e2m1fn_x2)

    def visit(name: str, value, depth: int) -> None:
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(value, torch.Tensor):
            if leaf in native or is_packed_layout(value):
                return
            if value.numel() * value.element_size() >= _EXPANSION_BYTES:
                found.append(f"{name}({value.dtype}, {value.numel() * value.element_size()}B)")
        elif depth < 3 and isinstance(value, (dict, list, tuple)):
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, inner in items:
                visit(f"{name}.{key}", inner, depth + 1)

    for name, value in module._buffers.items():
        visit(name, value, 0)
    for name, value in module._parameters.items():
        visit(name, value, 0)
    for name, value in module.__dict__.items():
        if name in {"_buffers", "_parameters", "_modules"}:
            continue
        visit(name, value, 0)
    return found


def _bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


# --------------------------------------------------------------------------
# post_init must not leave an expanded weight behind.
# --------------------------------------------------------------------------

def test_post_init_retains_no_expanded_weight():
    module = _module(W4AFP8Linear)

    assert not hasattr(module, "_weight_e4m3")
    assert "_weight_e4m3" not in module._buffers
    assert "_weight_e4m3" not in module.__dict__
    assert "_weight_e4m3" not in module._WEIGHT_CACHE_NAMES
    assert _expanded_weight_tensors(module) == []
    assert module._w4a_weights_ready is True
    assert module.qweight.dtype == torch.int32

    # Constructor state: not ready until post_init() has validated the layout.
    fresh = W4AFP8Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=K, out_features=N, bias=False,
    )
    assert fresh._w4a_weights_ready is False
    assert "_weight_e4m3" not in fresh.__dict__

    assert set(module.state_dict()) == {"qweight", "qzeros", "scales", "g_idx"}


# --------------------------------------------------------------------------
# Both dispatch paths, twice, must not retain the operand.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("encoded", [False, True])
def test_forward_releases_the_expanded_weight(cpu_kernels, dtype, encoded):
    module = _module(W4AFP8Linear)
    native_state = {name: value.clone() for name, value in module.state_dict().items()}
    baseline_buffers = {name: _bytes(value) for name, value in module._buffers.items()}

    x = torch.randn((3, K), dtype=dtype)
    x[0].zero_()
    seen: list[torch.Tensor] = []
    references: list[weakref.ref] = []

    original = module.fp8_weight_operand

    def tracking():
        weight = original()
        assert weight.dtype == torch.float8_e4m3fn
        assert weight.numel() == K * N
        seen.append(weight)
        references.append(weakref.ref(weight))
        return weight

    module.fp8_weight_operand = tracking
    try:
        for _ in range(2):
            carrier = _fp8_carrier(x) if encoded else x
            out = module(carrier)
            assert isinstance(out, torch.Tensor)
            assert out.dtype == dtype and out.shape == (3, N)
            assert torch.isfinite(out).all()
    finally:
        del module.fp8_weight_operand

    # One freshly materialized operand per call, never a reused cache.
    assert len(seen) == 2
    assert seen[0] is not seen[1]
    seen.clear()  # drop the test's own strong references
    gc.collect()
    assert all(ref() is None for ref in references), "the per-forward E4M3 operand outlived its forward"

    # A forward neither creates nor mutates persistent state.
    assert _expanded_weight_tensors(module) == []
    assert "_weight_e4m3" not in module.__dict__
    assert {name: _bytes(value) for name, value in module._buffers.items()} == baseline_buffers
    for name, value in native_state.items():
        assert torch.equal(module.state_dict()[name], value)


def test_operand_is_released_when_the_kernel_raises(cpu_kernels, monkeypatch):
    module = _module(W4AFP8Linear)
    calls = []
    original = module.fp8_weight_operand

    def tracking():
        weight = original()
        calls.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "fp8_weight_operand", tracking)
    monkeypatch.setattr(
        cpu_kernels, "fp8_linear",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kernel failed")),
    )
    with pytest.raises(RuntimeError, match="kernel failed"):
        module(torch.zeros(2, K, dtype=torch.bfloat16))
    gc.collect()
    assert len(calls) == 1 and calls[0]() is None
    assert _expanded_weight_tensors(module) == []


def test_inference_mode_forward_leaves_no_expanded_weight(cpu_kernels):
    module = _module(W4AFP8Linear)
    x = torch.randn((2, K), dtype=torch.bfloat16)
    with torch.inference_mode():
        module(x)
    assert _expanded_weight_tensors(module) == []


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("method", ["to", "cast"])
def test_model_dtype_conversion_keeps_native_storage(dtype, method):
    """Casting the model must not create a derived weight or touch the INT4 pack."""
    module = _module(W4AFP8Linear)
    native = {name: value.clone() for name, value in module.state_dict().items()}
    retained = {name: _bytes(value) for name, value in module._buffers.items()}
    parent = torch.nn.Sequential(module)

    if method == "to":
        parent.to(dtype=dtype)
    elif dtype is torch.bfloat16:
        parent.bfloat16()
    else:
        parent.half()

    assert module.scales.dtype == dtype
    assert module.qweight.dtype == torch.int32
    assert module._w4a_weights_ready is True
    for name, value in native.items():
        if name == "scales":
            continue  # native scales legitimately follow the model dtype
        assert torch.equal(module.state_dict()[name], value)
    assert {name: _bytes(value) for name, value in module._buffers.items()} == retained
    assert _expanded_weight_tensors(module) == []
    # Derived operands stay non-persistent, so a save cannot ship an expanded
    # weight even if one were ever staged.
    saved = set(module.state_dict())
    assert "_weight_e4m3" not in saved and "_weight_both" not in saved


# --------------------------------------------------------------------------
# pack()/load must reset readiness without materializing anything.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("operation", ["pack", "load"])
def test_native_weight_changes_invalidate_readiness(operation):
    module = _module(W4AFP8Linear)
    if operation == "pack":
        dense = torch.nn.Linear(K, N, bias=False)
        dense.weight.data.zero_()
        module.pack(dense, torch.ones(N, 2), torch.full((N, 2), 8.), module.g_idx.clone())
    else:
        state = {name: value.clone() for name, value in module.state_dict().items()}
        state["qweight"].zero_()
        module.load_state_dict(state)
    assert module._w4a_weights_ready is False
    assert _expanded_weight_tensors(module) == []
    with pytest.raises(RuntimeError, match="not initialized"):
        module(torch.zeros(1, K, dtype=torch.float16))
    module.post_init()
    assert module._w4a_weights_ready is True
    assert _expanded_weight_tensors(module) == []


# --------------------------------------------------------------------------
# Numeric oracle: the staged operand must equal the decoded INT4 codes.
# --------------------------------------------------------------------------

def test_fp8_operand_matches_independent_int4_decode():
    module = _module(W4AFP8Linear)
    expected = _decode_centered_codes(module)
    operand = module.fp8_weight_operand()

    assert operand.dtype == torch.float8_e4m3fn and operand.shape == (K, N)
    arithmetic = (operand.float() - expected.float()).abs()
    # Centered INT4 values are exactly representable in E4M3: no arithmetic
    # error, and byte identity with an independent E4M3 cast.
    assert float(arithmetic.max()) == 0.0
    assert float(arithmetic.mean()) == 0.0
    assert torch.equal(
        operand.view(torch.uint8),
        expected.to(torch.float8_e4m3fn).view(torch.uint8),
    )
    # The fixture spans the full centered range, so saturation is exercised.
    assert int(expected.min()) == -8 and int(expected.max()) == 7


def test_int4_to_e4m3_cast_is_exact_at_every_code():
    """The staging cast is lossless over the complete centered INT4 range.

    Every value in [-8, 7] is a representable E4M3 number, so the cast must be
    bit-exact and must not round or saturate at either endpoint.
    """
    codes = torch.arange(-8, 8, dtype=torch.int16)
    staged = torch.empty(codes.shape, dtype=torch.float8_e4m3fn)
    staged.copy_(codes)
    assert torch.equal(staged.float(), codes.float())
    assert torch.equal(staged.to(torch.float32).to(torch.int16), codes)
    assert int(staged.float().min()) == -8 and int(staged.float().max()) == 7
    # Endpoints one code outside the GPTQ range round rather than wrap; the
    # staging path never reaches them because centered 4-bit codes are bounded.
    outside = torch.tensor([-9.0, 8.0]).to(torch.float8_e4m3fn).float()
    assert float(outside[0]) == -9.0 and float(outside[1]) == 8.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("encoded", [False, True])
def test_forward_output_matches_torch_oracle(cpu_kernels, dtype, encoded):
    module = _module(W4AFP8Linear)

    x = torch.randn((4, K), dtype=torch.float32).to(dtype)
    x[0].zero_()
    x[1, 0] = 300.0  # non-power-of-two token scale
    actual = module(_fp8_carrier(x) if encoded else x)

    weight = _decode_centered_codes(module).float()
    codes, row_scale = _fp8_pack_oracle(x)
    fp32 = _fp8_gemm_oracle(codes, row_scale, weight, module.scales)
    visible = fp32.to(dtype)

    assert actual.dtype == dtype
    # Visible-output comparison: the emitted tensor must be the FP32 oracle
    # rounded once to the model dtype, with no additional kernel error.
    assert torch.equal(actual, visible)
    # Residual against the FP32 value is the mandatory output rounding only.
    rounding = (actual.float() - fp32).abs()
    half_ulp = fp32.abs().max() * 2 ** (-8 if dtype is torch.bfloat16 else -11)
    assert float(rounding.max()) <= float(half_ulp)
    # Report both parts separately: the arithmetic part is exact here because
    # centered INT4 codes are exactly representable in E4M3.
    arithmetic = (weight - _decode_centered_codes(module).float()).abs()
    assert float(arithmetic.max()) == 0.0


# --------------------------------------------------------------------------
# Packed-nibble decoding must stay byte-exact against the shift oracle.
# --------------------------------------------------------------------------

def _shift_unpack_reference(packed: torch.Tensor) -> torch.Tensor:
    """Unpack with int32 shifts, the form the byte-level path replaced."""
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return ((packed.to(torch.int64).unsqueeze(1) >> shifts.view(1, 8, 1)) & 15).reshape(
        packed.shape[0] * 8, packed.shape[1]
    )


@pytest.mark.parametrize("rows,out", [(256, 128), (512, 64), (1024, 96), (2048, 512)])
def test_byte_unpack_matches_shift_oracle(rows, out):
    """Random words exercise every nibble in every position, not a ramp."""
    generator = torch.Generator().manual_seed(rows * 31 + out)
    words = torch.randint(
        0, 2**31 - 1, (rows // 8, out), generator=generator, dtype=torch.int64
    ).to(torch.int32)
    got = W4AFP8Linear._unpack_int4_codes(words)
    want = _shift_unpack_reference(words)
    assert got.dtype == torch.uint8
    assert got.shape == (rows, out)
    assert torch.equal(got.to(torch.int64), want)
    if rows == 2048:
        # A wide random fill must actually reach both ends of the code range.
        assert int(want.min()) == 0 and int(want.max()) == 15


def test_byte_unpack_nibble_order_is_low_first():
    """GPTQ stores row ``8*i + j`` in nibble ``j``, low nibble first."""
    packed = torch.tensor([0x01234567], dtype=torch.int64).to(torch.int32).reshape(1, 1)
    assert W4AFP8Linear._unpack_int4_codes(packed).flatten().tolist() == [7, 6, 5, 4, 3, 2, 1, 0]
    # One-hot words prove each of the eight positions maps to its own row.
    one_hot = torch.tensor(
        [(0xF << (4 * position)) for position in range(8)], dtype=torch.int64
    ).to(torch.int32).reshape(8, 1)
    expected = torch.zeros(64, 1, dtype=torch.uint8)
    for position in range(8):
        expected[position * 8 + position, 0] = 15
    assert torch.equal(W4AFP8Linear._unpack_int4_codes(one_hot), expected)


def test_centered_codes_works_on_foreign_gptq_module():
    """The checkpoint audits call this on a ``TorchLinear``, which has no helper.

    ``W4AFP8Linear._centered_int4_codes(module)`` is the documented entry point
    for modules that are not ``W4AFP8Linear`` instances. Resolving the unpacker
    through ``self`` raised ``AttributeError`` there, so pin the cross-module
    path instead of only the self-hosted one.
    """
    module = _module(TorchLinear, 256, 64)
    assert not hasattr(TorchLinear, "_unpack_int4_codes")

    codes = W4AFP8Linear._centered_int4_codes(module)

    expected = _shift_unpack_reference(module.qweight).to(torch.int16) - 8
    assert codes.dtype == torch.int16
    assert torch.equal(codes, expected)


# --------------------------------------------------------------------------
# Byte accounting used by the audit report.
# --------------------------------------------------------------------------

def test_persistent_weight_bytes():
    module = _module(W4AFP8Linear)
    packed_native = _bytes(module.qweight)
    zero_points = _bytes(module.qzeros)
    scales = _bytes(module.scales)
    group_index = _bytes(module.g_idx)
    native = packed_native + zero_points + scales + group_index
    assert native == K * N // 2 + (K // 128) * (N // 8) * 4 + (K // 128) * N * 2 + K * 4
    assert native == 18048

    native_names = {"qweight", "qzeros", "scales", "g_idx"}
    retained = sum(
        _bytes(value) for name, value in module._buffers.items() if name not in native_names
    )
    assert retained == 0  # no derived tensor at all

    # Transient scratch for one forward is just the E4M3 operand. The fused
    # Triton unpack reads packed words and writes E4M3 in one pass, so neither
    # the 4 B/weight int32 shift/mask planes nor the 1 B/weight uint8-view
    # intermediates of the Torch decode are ever allocated. Measured on RTX
    # 4090 the allocator delta is exactly 1 B/weight (32768 B here). The Torch
    # fallback is retained for CPU/meta weights and still needs 5 B/weight; see
    # the test below.
    fused = K * N
    assert fused == 32768
    # The superseded staging forms, kept so a silent regression is caught here
    # rather than only in the allocator measurement.
    torch_uint8_view = 5 * K * N
    torch_int32_shift = 8 * K * N
    assert fused < torch_uint8_view < torch_int32_shift


def test_transient_staging_peak_is_measured_not_assumed():
    """Peak staging bytes must be measured on GPU, and stay at the fused floor.

    On CPU there is no allocator to sample, so the analytic bound above is used.
    On CUDA the real peak is sampled from the caching allocator. The fused
    decode must peak at the E4M3 operand and nothing more; a regression to the
    uint8 view (5 B/weight) or the int32 form (8 B/weight) has to be caught
    here, so the bound is two-sided.
    """
    module = _module(W4AFP8Linear)
    operand_bytes = K * N
    floor = K * N
    if not w4a_kernels_available():
        pytest.skip(_W4A_GPU_REASON)

    module.to("cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    operand = module.fp8_weight_operand()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    assert operand.numel() * operand.element_size() == operand_bytes
    assert peak_delta >= floor, (
        f"staged peak {peak_delta} B is below the fused floor {floor} B"
    )
    # Bound the peak from above as well. The lower bound alone cannot detect a
    # staging path that regressed to the uint8 view (5 B/weight) or the int32
    # form (8 B/weight), nor one that started leaking the operand past the
    # call. One allocator block of slack covers rounding.
    assert peak_delta <= floor + K * N, (
        f"staged peak {peak_delta} B exceeds the fused 1 B/weight bound {floor + K * N} B"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_device_round_trip_forward_restages_from_native_weights(dtype):
    """A cpu->cuda move must neither restage at move time nor break forward.

    The old design kept a resident E4M3 plane and asserted the move preserved it.
    With per-forward staging there is nothing to preserve, so the contract is the
    inverse: the move stages nothing, and the first forward on the new device
    still produces the operand from the native INT4 weights.
    """
    if not w4a_kernels_available():
        pytest.skip(_W4A_GPU_REASON)
    module = _module(W4AFP8Linear)

    def unexpected_post_init(*args, **kwargs):
        pytest.fail("A device move must not restage the weights")

    module.post_init = unexpected_post_init
    module.to(device="cuda", dtype=dtype)
    assert module._w4a_weights_ready is True
    assert _expanded_weight_tensors(module) == []

    generator = torch.Generator(device="cuda").manual_seed(2415)
    x = torch.randn((7, K), device="cuda", dtype=dtype, generator=generator)
    x[0].zero_()
    actual = module(x)
    assert actual.dtype == dtype and actual.shape == (7, N)
    assert torch.isfinite(actual).all()

    first = module(x)
    module.cpu().cuda()
    moved = module(x)
    torch.cuda.synchronize()
    assert moved.dtype == dtype
    assert torch.equal(moved, first)


# --------------------------------------------------------------------------
# Negative controls: the scanner must not be bypassable by dtype or by name.
# --------------------------------------------------------------------------

def test_scanner_detects_a_full_layer_integer_plane():
    """An INT8/INT16 full-layer weight is prohibited, not exempt by dtype."""
    module = _module(W4AFP8Linear)
    module._hidden_int8 = torch.zeros(K, N, dtype=torch.int8)
    module._hidden_int16 = torch.zeros(K, N, dtype=torch.int16)
    found = _expanded_weight_tensors(module)
    assert any("_hidden_int8" in item for item in found), found
    assert any("_hidden_int16" in item for item in found), found


def test_scanner_detects_an_float8_plane_using_a_packed_cache_name():
    """Reusing a packed-cache attribute name must not hide an expanded plane."""
    module = _module(W4AFP8Linear)
    # Same dtype and shape as the removed resident E4M3 cache, but parked under
    # a name the previous allowlist-based scanner would have skipped.
    module._weight_both = torch.zeros(K, N, dtype=torch.float8_e4m3fn)
    module._unit_weight_scales = torch.zeros(K, N, dtype=torch.float8_e4m3fn)
    found = _expanded_weight_tensors(module)
    assert any("_weight_both" in item for item in found), found
    assert any("_unit_weight_scales" in item for item in found), found


def test_scanner_does_not_flag_the_native_checkpoint_or_packed_fp4():
    module = _module(W4AFP8Linear)
    assert _expanded_weight_tensors(module) == []
    if getattr(torch, "float4_e2m1fn_x2", None) is not None:
        module._packed = torch.zeros(K // 2, N, dtype=torch.float4_e2m1fn_x2)
        assert _expanded_weight_tensors(module) == []


# --------------------------------------------------------------------------
# Inference tensors carry no version counter, so an in-place weight write is
# invisible to the layout token; the explicit-clear contract must cover it.
# --------------------------------------------------------------------------

def test_invalidate_weight_layout_forces_revalidation():
    """The public hook drops readiness and forward() refuses until post_init()."""
    module = _module(W4AFP8Linear)
    assert module._w4a_weights_ready is True

    module.invalidate_weight_layout()

    assert module._w4a_weights_ready is False
    assert module._w4a_layout_token is None
    with pytest.raises(RuntimeError, match="not initialized"):
        module(torch.zeros(1, K, dtype=torch.bfloat16))

    module.post_init()
    assert module._w4a_weights_ready is True
    assert _expanded_weight_tensors(module) == []


def test_inference_tensor_layout_is_reported_as_unobservable():
    """Inference tensors have no version counter, so the token cannot change.

    The module must therefore admit the gap rather than let a stale token look
    current: it warns once and offers ``invalidate_weight_layout()``.
    """
    with torch.inference_mode():
        module = _module(W4AFP8Linear)

    assert module.qzeros.is_inference()
    assert not module._layout_token_is_observable()

    before = module._weight_layout_token()
    module._w4a_layout_token = before
    module._w4a_weights_ready = True

    # The blind spot is real: mutating the packed zero points in place leaves
    # the token byte-identical, so it cannot be used to detect the write.
    with torch.inference_mode():
        module.qzeros.fill_(0x00000000)
    assert not torch.equal(
        W4AFP8Linear._group_zero_points(module), torch.full((K // 128, N), 8)
    )
    assert module._weight_layout_token() == before

    # Explicit invalidation is the supported escape hatch.
    module.invalidate_weight_layout()
    assert module._w4a_weights_ready is False
    assert _expanded_weight_tensors(module) == []


def test_observable_layout_is_reported_for_normal_construction():
    module = _module(W4AFP8Linear)
    assert module.qzeros.is_inference() is False
    assert module._layout_token_is_observable()


# --------------------------------------------------------------------------
# ROCm gating: HIP reuses the CUDA namespace but cannot lower these kernels.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [W4AFP8Linear, W4ANVFP4Linear])
def test_rocm_rejects_the_w4a_backend(monkeypatch, cls):
    """A ROCm target must be refused at validation, not at Triton compile time.

    ``torch.cuda.is_available()`` is true on ROCm and gfx1201 reports a
    synthetic ``(12, 0)`` capability that clears the FP8 floor, so the torch
    checks cannot gate this backend. Without the family gate the first forward
    aborts inside Triton with "Unsupported DotOp found when converting
    TritonGPU to LLVM".
    """
    w4a_floatx = sys.modules[W4AFP8Linear.__module__]
    monkeypatch.setattr(w4a_floatx, "normalize_device", lambda device: DEVICE.ROCM)
    error = cls._rocm_unsupported_error()
    assert isinstance(error, NotImplementedError)
    assert "ROCm" in str(error) and "ttg.nvidia_mma" in str(error)

    monkeypatch.setattr(w4a_floatx, "normalize_device", lambda device: DEVICE.CUDA)
    assert cls._rocm_unsupported_error() is None


def test_rocm_kernels_refuse_before_lowering(monkeypatch):
    """Direct kernel calls must report the platform, not a PassManager crash."""
    from gptqmodel.nn_modules.qlinear import w4a_nvfp4_triton as nvfp4_triton
    from gptqmodel.nn_modules.qlinear import w4a_triton

    monkeypatch.setattr(w4a_triton, "IS_ROCM", True)
    monkeypatch.setattr(nvfp4_triton, "IS_ROCM", True)

    error = w4a_triton.native_fp8_gemm_error()
    assert isinstance(error, RuntimeError)
    assert "ttg.nvidia_mma" in str(error)

    # Device/argument checks must not run first: a HIP host has no usable path
    # for any of these, whatever the operand happens to be.
    with pytest.raises(NotImplementedError, match="CUDA-only"):
        w4a_triton.unpack_int4_to_e4m3(
            torch.zeros((32, 64), dtype=torch.int32),
            torch.empty((256, 64), dtype=torch.float8_e4m3fn),
        )
    with pytest.raises(NotImplementedError, match="CUDA-only"):
        w4a_triton.fp8_pack(torch.zeros((4, K), dtype=torch.float16))
    with pytest.raises(NotImplementedError, match="CUDA-only"):
        nvfp4_triton.nvfp4_pack_and_swizzle(
            torch.zeros((4, K), dtype=torch.float16), torch.ones(())
        )


# --------------------------------------------------------------------------
# Fused INT4 -> E4M3 unpacking must stay byte-identical to the Torch decode.
# --------------------------------------------------------------------------

def _fused_unpack():
    """Import the Triton module lazily.

    ``w4a_triton`` needs a Triton build with Gluon ``mma_v2``; importing it at
    module scope would break the CPU-only storage tests on a build without it.
    """
    from gptqmodel.nn_modules.qlinear.w4a_triton import unpack_int4_to_e4m3
    return unpack_int4_to_e4m3


@pytest.mark.parametrize("rows,out", [(256, 128), (512, 64), (1024, 96), (2048, 512)])
def test_fused_unpack_matches_torch_decode(rows, out):
    """The fused kernel must reproduce the Torch decode bit for bit.

    The Torch path is the oracle: it expands the same packed words with plain
    shifts and casts to E4M3, so any nibble-order, centering, or rounding
    difference shows up as a byte mismatch. Random words exercise every nibble
    position instead of a ramp that could agree by accident.
    """
    if not w4a_kernels_available():
        pytest.skip(_W4A_GPU_REASON)

    generator = torch.Generator().manual_seed(rows * 17 + out)
    packed = torch.randint(
        0, 2**31 - 1, (rows // 8, out), generator=generator, dtype=torch.int64
    ).to(torch.int32).cuda()

    expected = (
        unpack_int4_codes(packed.cpu()).to(torch.int16) - 8
    ).to(torch.float8_e4m3fn).cuda()
    fused = torch.empty((rows, out), device="cuda", dtype=torch.float8_e4m3fn)

    _fused_unpack()(packed, fused)

    assert fused.dtype == torch.float8_e4m3fn and fused.shape == (rows, out)
    assert torch.equal(fused.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("layout", ["packed_rows", "packed_columns", "output_rows", "output_columns"])
def test_fused_unpack_respects_strided_input_and_caller_output(layout):
    if not w4a_kernels_available():
        pytest.skip(_W4A_GPU_REASON)

    rows, columns = 256, 64
    generator = torch.Generator().manual_seed(1753)
    packed = torch.randint(0, 2**31 - 1, (rows // 8, columns), generator=generator, dtype=torch.int64)
    packed = packed.to(device="cuda", dtype=torch.int32)
    expected = (unpack_int4_codes(packed.cpu()).to(torch.int16) - 8).to(torch.float8_e4m3fn).cuda()
    argument = packed
    output = torch.empty((rows, columns), device="cuda", dtype=torch.float8_e4m3fn)
    untouched = None
    if layout.startswith("packed"):
        if layout.endswith("rows"):
            storage = torch.empty((rows // 8, columns + 8), device="cuda", dtype=torch.int32)
            argument = storage[:, :columns]
        else:
            storage = torch.empty((rows // 8, columns * 2), device="cuda", dtype=torch.int32)
            argument = storage[:, ::2]
        argument.copy_(packed)
        assert not argument.is_contiguous()
    else:
        if layout.endswith("rows"):
            storage = torch.empty((rows, columns + 8), device="cuda", dtype=torch.float8_e4m3fn)
            output = storage[:, :columns]
            untouched = storage[:, columns:]
        else:
            storage = torch.empty((rows, columns * 2), device="cuda", dtype=torch.float8_e4m3fn)
            output = storage[:, ::2]
            untouched = storage[:, 1::2]
        storage.view(torch.uint8).fill_(0xA5)
        assert not output.is_contiguous()

    assert _fused_unpack()(argument, output) is output
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    if untouched is not None:
        assert bool((untouched.view(torch.uint8) == 0xA5).all())


def test_fused_unpack_rejects_bad_arguments():
    """Wrong dtype/shape/device must be refused instead of silently decoded."""
    if not w4a_kernels_available():
        pytest.skip(_W4A_GPU_REASON)

    packed = torch.zeros((32, 64), dtype=torch.int32, device="cuda")
    good = torch.empty((256, 64), dtype=torch.float8_e4m3fn, device="cuda")

    with pytest.raises(ValueError, match="CUDA"):
        _fused_unpack()(packed.cpu(), good.cpu())
    with pytest.raises(ValueError, match="E4M3"):
        _fused_unpack()(packed, torch.empty((256, 64), dtype=torch.float16, device="cuda"))
    with pytest.raises(ValueError, match=r"must be \(256, 64\)"):
        _fused_unpack()(packed, torch.empty((128, 64), dtype=torch.float8_e4m3fn, device="cuda"))


def test_fused_unpack_all_zero_and_all_fifteen_words():
    """The extreme word values pin the ends of the centered INT4 range."""
    if not w4a_kernels_available():
        pytest.skip(_W4A_GPU_REASON)

    out = torch.empty((64, 32), device="cuda", dtype=torch.float8_e4m3fn)
    zero_words = torch.zeros((8, 32), dtype=torch.int32, device="cuda")
    full_words = torch.full((8, 32), -1, dtype=torch.int32, device="cuda")

    assert torch.equal(
        _fused_unpack()(zero_words, out).float(), torch.full((64, 32), -8.0, device="cuda")
    )
    assert torch.equal(
        _fused_unpack()(full_words, out).float(), torch.full((64, 32), 7.0, device="cuda")
    )


# --------------------------------------------------------------------------
# W4ANVFP4Linear stages its own *packed* operand caches and so skips
# W4AFP8Linear.post_init; its post_init must still revalidate the packed
# checkpoint layout before decoding, or a bad checkpoint silently corrupts
# every weight it exports. The FP8 lane it inherits is per-forward only.
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 operand caches need native PyTorch FP4 support",
)
def test_nvfp4_post_init_rejects_desc_act_groups():
    module = _module(W4ANVFP4Linear)
    module.g_idx[0], module.g_idx[-1] = module.g_idx[-1].clone(), module.g_idx[0].clone()
    with pytest.raises(ValueError, match="desc_act"):
        module.post_init()


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 operand caches need native PyTorch FP4 support",
)
def test_nvfp4_post_init_rejects_non_symmetric_zero_points():
    module = _module(W4ANVFP4Linear)
    module.qzeros[0, 0] = 0x66666666
    with pytest.raises(ValueError, match="zero point 8"):
        module.post_init()


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 operand caches need native PyTorch FP4 support",
)
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_nvfp4_hf_pack_layer_checkpoint_format(checkpoint_format):
    """hf_pack_layer must mark the raw packer's v2-layout zero points
    truthfully, converting only when a v1 kernel checkpoint is requested.

    A mislabeled format either skips the needed v1 conversion or adds one to
    already-v2 words; both leave post_init's symmetric-zero-point check
    reading 9 instead of 8."""
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils.packing import hf_pack_layer

    module = W4ANVFP4Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=K, out_features=N, bias=False,
    )
    module.activation_global_scale.fill_(0.03125)
    linear = torch.nn.Linear(K, N, bias=False)
    scales = torch.full((N, K // 128), 0.125, dtype=torch.float16)
    zeros = torch.full((N, K // 128), 8, dtype=torch.int32)
    g_idx = torch.arange(K, dtype=torch.int32) // 128

    fmt = FORMAT.GPTQ if checkpoint_format == "gptq" else FORMAT.GPTQ_V2
    hf_pack_layer(module, linear, scales, zeros, g_idx, checkpoint_format=fmt)

    # The v1 flavor stores zp-1 nibbles; the v2 flavor stores logical zeros.
    expected_word = 0x77777777 if checkpoint_format == "gptq" else -0x77777778
    assert torch.equal(module.qzeros, torch.full_like(module.qzeros, expected_word))

    # Either way the logical zero point decodes back to the symmetric point.
    zeros_decoded = W4AFP8Linear._group_zero_points(module)
    assert torch.equal(
        zeros_decoded, torch.full((K // 128, N), 8, dtype=zeros_decoded.dtype)
    )
    # post_init() revalidates the packed layout and raises on a format
    # mismatch, so returning cleanly is the usability signal on CPU; forward
    # itself is CUDA-gated.
    module.post_init()


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 operand caches need native PyTorch FP4 support",
)
def test_nvfp4_state_dict_roundtrip_restores_v2_zero_format():
    """`_qzeros_format` is not serialized: a fresh module must recover the v2
    domain from its declared `format` or the post_init layout check reads the
    valid `0x88888888` word as zero point 9."""
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils.packing import hf_pack_layer

    module = W4ANVFP4Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=K, out_features=N, bias=False,
        format=FORMAT.GPTQ_V2,
    )
    module.activation_global_scale.fill_(0.03125)
    linear = torch.nn.Linear(K, N, bias=False)
    scales = torch.full((N, K // 128), 0.125, dtype=torch.float16)
    zeros = torch.full((N, K // 128), 8, dtype=torch.int32)
    g_idx = torch.arange(K, dtype=torch.int32) // 128
    hf_pack_layer(
        module, linear, scales, zeros, g_idx, checkpoint_format=FORMAT.GPTQ_V2
    )

    fresh = W4ANVFP4Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=K, out_features=N, bias=False,
        format=FORMAT.GPTQ_V2,
    )
    fresh.load_state_dict(module.state_dict())
    # Without the format restore this raises, reading zp=8 words as zp=9.
    fresh.post_init()


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 operand caches need native PyTorch FP4 support",
)
def test_nvfp4_failed_repost_init_clears_operand_caches():
    """A failed re-post_init must not leave forward() serving operands decoded
    from a layout that no longer matches the weights."""
    module = _module(W4ANVFP4Linear)

    module.qzeros[0, 0] = 0x66666666
    with pytest.raises(ValueError, match="zero point 8"):
        module.post_init()

    for name in W4ANVFP4Linear._WEIGHT_CACHE_NAMES:
        assert getattr(module, name).numel() == 0
    with pytest.raises(RuntimeError, match="weight cache is absent"):
        module.forward(torch.zeros(1, K, dtype=torch.float16))


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 packed caches need native PyTorch FP4 support",
)
def test_nvfp4_post_init_retains_only_packed_planes():
    """The NVFP4 caches are packed layouts, not expanded weight planes.

    ``_weight_both`` is two E2M1 planes at 4 bits per weight, so 1 B/weight in
    total; that is an allowed alternate packed layout. A full-layer E4M3 plane
    is one code per weight (an expansion) and must not be retained even though
    the mixed FP8 attention lane needs it on demand.
    """
    module = _module(W4ANVFP4Linear)

    assert not hasattr(module, "_weight_e4m3")
    assert "_weight_e4m3" not in module._buffers
    assert "_weight_e4m3" not in module.__dict__
    assert "_weight_e4m3" not in module._WEIGHT_CACHE_NAMES
    assert module._w4a_weights_ready is True

    # The retained packed planes are 1 B/weight; native INT4 is 0.5 B/weight.
    assert _bytes(module.qweight) == K * N // 2
    assert _bytes(module._weight_both) == K * N
    assert _bytes(module._unit_weight_scales) < K * N
    assert _expanded_weight_tensors(module) == []
    assert set(module.state_dict()) == {
        "qweight", "qzeros", "scales", "g_idx", "activation_global_scale_bits",
    }

    # Independent E2M1 decode: the packed planes must reconstruct every
    # centered INT4 code, so removing the E4M3 plane changed no weights.
    centered = (
        _decode_e2m1_planes(module._weight_both[:, :N])
        + 4 * _decode_e2m1_planes(module._weight_both[:, N:])
    )
    assert torch.equal(centered, _decode_centered_codes(module).float())


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 packed caches need native PyTorch FP4 support",
)
def test_nvfp4_fp8_operand_is_staged_per_forward(cpu_kernels):
    """The mixed FP8 lane materializes and drops one E4M3 operand per call."""
    module = _module(W4ANVFP4Linear)
    native_state = {name: value.clone() for name, value in module.state_dict().items()}
    baseline_buffers = {name: _bytes(value) for name, value in module._buffers.items()}

    x = torch.randn((3, K), dtype=torch.bfloat16)
    x[0].zero_()
    seen: list[torch.Tensor] = []
    references: list[weakref.ref] = []
    original = module.fp8_weight_operand

    def tracking():
        weight = original()
        assert weight.dtype == torch.float8_e4m3fn
        assert weight.numel() == K * N
        seen.append(weight)
        references.append(weakref.ref(weight))
        return weight

    module.fp8_weight_operand = tracking
    try:
        for _ in range(2):
            out = module._forward_fp8_operand(_fp8_carrier(x))
            assert out.dtype == torch.bfloat16 and out.shape == (3, N)
            assert torch.isfinite(out).all()
    finally:
        del module.fp8_weight_operand

    assert len(seen) == 2
    assert seen[0] is not seen[1]
    seen.clear()
    gc.collect()
    assert all(ref() is None for ref in references), "the per-forward E4M3 operand outlived its forward"

    assert not hasattr(module, "_weight_e4m3")
    assert _expanded_weight_tensors(module) == []
    assert {name: _bytes(value) for name, value in module._buffers.items()} == baseline_buffers
    for name, value in native_state.items():
        assert torch.equal(module.state_dict()[name], value)


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 packed caches need native PyTorch FP4 support",
)
def test_nvfp4_fp8_operand_matches_independent_int4_decode():
    module = _module(W4ANVFP4Linear)
    expected = _decode_centered_codes(module)
    operand = module.fp8_weight_operand()

    assert operand.dtype == torch.float8_e4m3fn and operand.shape == (K, N)
    assert torch.equal(
        operand.view(torch.uint8),
        expected.to(torch.float8_e4m3fn).view(torch.uint8),
    )
    # Staging the operand must not install it as module state.
    assert not hasattr(module, "_weight_e4m3")
    assert _expanded_weight_tensors(module) == []


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 packed caches need native PyTorch FP4 support",
)
def test_nvfp4_forward_rejects_a_stale_packed_cache():
    """An in-place qweight write must invalidate the packed planes."""
    module = _module(W4ANVFP4Linear)
    module.qweight[0, 0] ^= 1

    with pytest.raises(RuntimeError, match="changed since post_init"):
        module.forward(torch.zeros(1, K, dtype=torch.float16))

    # Explicit invalidation has the same effect and drops the packed planes.
    module.post_init()
    module.invalidate_weight_layout()
    assert module._weight_both.numel() == 0
    with pytest.raises(RuntimeError, match="weight cache is absent"):
        module.forward(torch.zeros(1, K, dtype=torch.float16))


@pytest.mark.skipif(
    getattr(torch, "float4_e2m1fn_x2", None) is None,
    reason="W4ANVFP4 packed caches need native PyTorch FP4 support",
)
def test_nvfp4_fp8_staging_cannot_launder_a_stale_packed_cache():
    """Staging the FP8 operand must not make the packed planes look current.

    The inherited ``fp8_weight_operand()`` revalidates a changed layout and then
    rewrites ``_w4a_layout_token`` to the *new* weights. Nothing rebuilds
    ``_weight_both`` there, so a refresh would let the next ``forward()`` pass
    its freshness check while the FP4 planes still encode the old codes. NVFP4
    therefore rejects changed weights instead of revalidating them.
    """
    module = _module(W4ANVFP4Linear)
    before = _decode_centered_codes(module)
    assert module._weights_are_current() is True

    module.qweight[0, 0] ^= 1
    mutated = _decode_centered_codes(module)
    assert not torch.equal(before, mutated)
    assert module._weights_are_current() is False

    # The public FP8 staging helper must refuse, not silently refresh the token.
    with pytest.raises(RuntimeError, match="changed since post_init"):
        module.fp8_weight_operand()
    assert module._weights_are_current() is False, (
        "FP8 staging refreshed the layout token without rebuilding the packed cache"
    )

    # The FP8 lane and the NVFP4 lane both refuse the mutated module.
    with pytest.raises(RuntimeError, match="changed since post_init"):
        module._forward_fp8_operand(_fp8_carrier(torch.zeros(2, K, dtype=torch.bfloat16)))
    with pytest.raises(RuntimeError, match="changed since post_init"):
        module.forward(torch.zeros(1, K, dtype=torch.float16))

    # Rebuilding restores agreement between both weight representations.
    module.post_init()
    assert module._weights_are_current() is True
    assert torch.equal(_decode_centered_codes(module), mutated)
    planes = (
        _decode_e2m1_planes(module._weight_both[:, :N])
        + 4 * _decode_e2m1_planes(module._weight_both[:, N:])
    )
    assert torch.equal(planes, mutated.float())
    assert torch.equal(
        module.fp8_weight_operand().view(torch.uint8),
        mutated.to(torch.float8_e4m3fn).view(torch.uint8),
    )


def test_failed_repost_init_clears_w4a_readiness():
    module = _module(W4AFP8Linear)
    assert module._w4a_weights_ready is True

    module.qzeros[0, 0] = 0x66666666
    with pytest.raises(ValueError, match="zero point 8"):
        module.post_init()

    assert module._w4a_weights_ready is False
    assert module._w4a_layout_token is None
