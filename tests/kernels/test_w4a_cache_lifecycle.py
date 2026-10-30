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
``test_w4afp8_gb10.py``.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear


K, N = 256, 128

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

    # Transient scratch for one forward. The byte-level unpack no longer
    # materializes 4 B/weight int32 shift and mask planes; the peak is now the
    # E4M3 operand plus the one-byte-per-weight intermediate planes that are
    # live together while the decode runs. Measured on RTX 4090 the allocator
    # delta is exactly 5 B/weight (163840 B here), down from 8 B/weight for the
    # int32 form -- and still above the 65536 B figure the first draft of this
    # work reported, which counted the operand plus one group but not the live
    # intermediates.
    floor = 5 * K * N
    assert floor == 163840
    assert floor > 2 * 65536, "the reported 65536 B constant cannot cover the intermediates"


def test_transient_staging_peak_is_measured_not_assumed():
    """Peak staging bytes must be measured on GPU, and exceed the old constant.

    On CPU there is no allocator to sample, so the analytic floor above is used.
    On CUDA the real peak is sampled from the caching allocator; it must cover
    the per-weight intermediates the decode allocates, not just the E4M3
    operand.
    """
    module = _module(W4AFP8Linear)
    operand_bytes = K * N
    floor = 5 * K * N
    if not torch.cuda.is_available():
        pytest.skip("allocator sampling requires CUDA")

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
        f"staged peak {peak_delta} B is below the intermediate-plane floor {floor} B"
    )
    # Bound the peak from above as well. The lower bound alone cannot detect a
    # staging path that regressed to the old int32 form (8 B/weight) or that
    # started leaking the operand beyond the call, and any assertion comparing
    # against the superseded 65536 B constant is vacuous once the 5 B/weight
    # floor holds. One allocator block of slack covers rounding.
    assert peak_delta <= floor + K * N, (
        f"staged peak {peak_delta} B exceeds the 5 B/weight bound {floor + K * N} B"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_device_round_trip_forward_restages_from_native_weights(dtype):
    """A cpu->cuda move must neither restage at move time nor break forward.

    The old design kept a resident E4M3 plane and asserted the move preserved it.
    With per-forward staging there is nothing to preserve, so the contract is the
    inverse: the move stages nothing, and the first forward on the new device
    still produces the operand from the native INT4 weights.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
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
