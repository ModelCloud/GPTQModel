# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.

from __future__ import annotations

import weakref
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch_int8 import Int8PackedModule, TorchInt8Linear, _write_int8_buffers
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.model import get_state_dict_for_save
from tests.test_torch_weight_storage import build_case, state_tensors


@pytest.mark.parametrize("bits", [2, 4])
def test_lower_bit_int8_construction_is_rejected(bits):
    with pytest.raises(NotImplementedError, match="8-bit source weights"):
        TorchInt8Linear(bits=bits, group_size=32, sym=True, desc_act=False,
                        in_features=256, out_features=64)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("backend", [BACKEND.GPTQ_TORCH_INT8, BACKEND.TORCH_INT8])
def test_lower_bit_int8_selection_is_rejected(bits, backend):
    with pytest.raises(ValueError, match="8-bit source weights"):
        select_quant_linear(bits=bits, group_size=32, desc_act=False, sym=True,
                            device=DEVICE.CPU, backend=backend, format=FORMAT.GPTQ_V2,
                            quant_method=METHOD.GPTQ, pack_dtype=torch.int32)


@pytest.mark.parametrize("bits", [2, 4])
def test_requantization_writer_rejects_expansion_before_work(bits, monkeypatch):
    module = torch.nn.Module()
    module.bits = bits

    def forbidden(*args, **kwargs):
        pytest.fail("A rejected low-bit conversion must not requantize weights")

    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.torch_int8._requantize_to_int8", forbidden)
    with pytest.raises(NotImplementedError, match="8-bit source weights"):
        _write_int8_buffers(module, torch.zeros(32, 64), torch.float32)
    assert list(module.buffers()) == []


@pytest.mark.parametrize("entry", ["post_init", "transform_cpu", "forward"])
def test_low_bit_runtime_entry_points_cannot_bypass_guard(entry, monkeypatch):
    module = TorchInt8Linear(bits=8, group_size=32, sym=True, desc_act=False,
                            in_features=256, out_features=64).eval()
    module.bits = 4
    monkeypatch.setattr(module, "dequantize_weight",
                        lambda **kwargs: pytest.fail("Do not dequantize rejected source weights"))
    with pytest.raises(NotImplementedError, match="8-bit source weights"):
        if entry == "post_init":
            module.post_init()
        elif entry == "transform_cpu":
            module.transform_cpu(torch.float32)
        else:
            module(torch.ones(1, 256))
    assert not hasattr(module, "int8_weight_nk")


@pytest.mark.parametrize("group_size", [-1, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_int8_codes_output_and_storage_oracle(group_size, dtype, monkeypatch):
    source, weight = build_case(bits=8, group_size=group_size)
    channel_scale = (weight.abs().amax(0) / 127).clamp_min(1e-10)
    codes = (weight / channel_scale).round().clamp(-128, 127).to(torch.int8)
    candidate = TorchInt8Linear(bits=8, group_size=group_size, sym=False, desc_act=True,
                                in_features=source.in_features, out_features=source.out_features)
    candidate.load_state_dict(source.state_dict())
    references = []
    original = candidate._dequantize_gptq_source_weight

    def tracked(*args, **kwargs):
        dense = original(*args, **kwargs)
        references.append(weakref.ref(dense))
        return dense

    monkeypatch.setattr(candidate, "_dequantize_gptq_source_weight", tracked)
    candidate.post_init()
    candidate.eval()
    assert all(reference() is None for reference in references)
    torch.testing.assert_close(candidate.int8_weight_nk, codes.t(), rtol=0, atol=0)
    torch.testing.assert_close(candidate.int8_channel_scale, channel_scale, rtol=0, atol=0)
    torch.testing.assert_close(candidate.dequantize_weight(), codes.float() * channel_scale, rtol=0, atol=0)
    for name in candidate.GPTQ_BUFFER_NAMES:
        torch.testing.assert_close(getattr(candidate, name), getattr(source, name), rtol=0, atol=0)
    for current_dtype in (dtype, torch.bfloat16, torch.float32, dtype):
        activation = torch.arange(7 * source.in_features).reshape(7, -1).remainder(9).sub(4).to(current_dtype) / 16
        # The kernel multiplies the packed codes by the canonical FP32 channel
        # scale and only rounds the final output, so the oracle must not
        # pre-round the scale to the activation dtype.
        arithmetic = (activation.double() @ codes.double()) * channel_scale.double()
        expected = arithmetic.to(current_dtype)
        with torch.inference_mode():
            output = candidate(activation)
        assert output.dtype == current_dtype
        # Visible error: what the low-precision output dtype can represent.
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
        # Arithmetic error: the kernel performs one output rounding on top of an
        # FP32 reduction, so it must stay inside half an output-dtype ulp plus
        # the FP32 accumulation slack. This constrains the kernel itself -- an
        # activation-dtype accumulation (or an FP32-scale pre-round) violates it.
        bound = (
            0.5 * torch.finfo(current_dtype).eps * arithmetic.abs()
            + 8 * torch.finfo(torch.float32).eps
            * ((activation.abs().double() @ codes.abs().double()) * channel_scale.double())
        )
        excess = (output.double() - arithmetic).abs() - bound
        assert int((excess > 0).sum()) == 0, (
            f"group_size={group_size} {current_dtype}: {(excess > 0).sum().item()}/{excess.numel()} "
            f"outputs exceed the FP32-reduction bound; max excess={excess.max().item():.3e}"
        )
        torch.testing.assert_close(candidate.int8_module.int8_channel_scale, channel_scale, rtol=0, atol=0)
        allowed = {candidate.int8_weight_nk.data_ptr(), candidate.int8_channel_scale.data_ptr()}
        allowed.update(getattr(candidate, name).data_ptr() for name in candidate.GPTQ_BUFFER_NAMES)
        assert all(tensor.untyped_storage().data_ptr() in allowed for tensor in state_tensors(candidate))
        assert candidate.int8_weight_nk.untyped_storage().nbytes() == source.in_features * source.out_features
        assert all(reference() is None for reference in references)


def _native_int8_oracle_from_checkpoint(state):
    """Decode GPTQ words independently, then apply per-channel INT8 quantization."""
    shifts = torch.arange(0, 32, 8, dtype=torch.int64)
    qweight = state["qweight"].to(torch.int64)
    qzeros = state["qzeros"].to(torch.int64)
    codes = ((qweight[:, None, :] >> shifts[None, :, None]) & 255).reshape(-1, qweight.shape[1])
    zeros = ((qzeros[:, :, None] >> shifts) & 255).reshape(-1, qweight.shape[1])
    groups = state["g_idx"].long()
    weight = (codes - zeros[groups]).float() * state["scales"].float()[groups]
    channel_scale = (weight.abs().amax(0) / 127).clamp_min(1e-10)
    int8_codes = (weight / channel_scale).round().clamp(-128, 127).to(torch.int8)
    return int8_codes, channel_scale


@pytest.mark.parametrize("group_size", [-1, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("assign", [False, True])
def test_native_int8_checkpoint_save_strict_reload_and_oracle(tmp_path, group_size, dtype, assign):
    source, _ = build_case(bits=8, group_size=group_size)
    constructor = {"bits": 8, "group_size": group_size, "sym": False, "desc_act": True,
                   "in_features": source.in_features, "out_features": source.out_features, "bias": True}
    checkpoint = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    checkpoint["bias"] = torch.linspace(-0.25, 0.25, source.out_features, dtype=torch.float16)
    candidate = TorchInt8Linear(**constructor)
    candidate.load_state_dict(checkpoint, strict=True)
    candidate.post_init()
    candidate.eval()

    saved = candidate.state_dict()
    assert set(saved) == set(checkpoint)
    assert not any(name.startswith("int8_") for name in saved)
    for name, tensor in checkpoint.items():
        torch.testing.assert_close(saved[name], tensor, rtol=0, atol=0)
    owner = torch.nn.Module()
    owner.add_module("projection", candidate)
    writer_sources = get_state_dict_for_save(owner)
    assert set(writer_sources) == {f"projection.{name}" for name in checkpoint}
    path = tmp_path / "model.safetensors"
    save_file({name: entry.source.contiguous() for name, entry in writer_sources.items()}, str(path))
    restored = TorchInt8Linear(**constructor)
    restored_owner = torch.nn.Module()
    restored_owner.add_module("projection", restored)
    restored_owner.load_state_dict(load_file(str(path)), strict=True, assign=assign)
    restored.post_init()
    restored.eval()

    activation = torch.arange(7 * source.in_features).reshape(7, -1).remainder(11).sub(5).to(dtype) / 16
    codes, channel_scale = _native_int8_oracle_from_checkpoint(checkpoint)
    torch.testing.assert_close(candidate.int8_weight_nk, codes.t(), rtol=0, atol=0)
    torch.testing.assert_close(candidate.int8_channel_scale, channel_scale, rtol=0, atol=0)
    torch.testing.assert_close(restored.int8_weight_nk, codes.t(), rtol=0, atol=0)
    torch.testing.assert_close(restored.int8_channel_scale, channel_scale, rtol=0, atol=0)
    expected = ((activation.double() @ codes.double()) * channel_scale.double() + checkpoint["bias"].double()).to(dtype)
    with torch.inference_mode():
        actual = candidate(activation)
        reloaded = restored(activation)
        repeated = restored(activation)
    assert set(get_state_dict_for_save(owner)) == set(writer_sources)
    assert actual.dtype == reloaded.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(reloaded, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(reloaded, repeated, rtol=0, atol=0)
    assert restored.int8_weight_nk.dtype == torch.int8
    assert not any(
        tensor.is_floating_point() and tuple(tensor.shape) in ((source.in_features, source.out_features),
                                                                 (source.out_features, source.in_features))
        for tensor in state_tensors(restored)
    )


@pytest.mark.parametrize("change", ["load", "load_assign", "inplace", "replace", "move", "clear"])
def test_native_int8_runtime_refreshes_after_checkpoint_change(change):
    source, _ = build_case(bits=8, group_size=32)
    constructor = {"bits": 8, "group_size": 32, "sym": False, "desc_act": True,
                   "in_features": source.in_features, "out_features": source.out_features}
    candidate = TorchInt8Linear(**constructor)
    candidate.load_state_dict(source.state_dict(), strict=True)
    candidate.post_init()
    candidate.eval()
    activation = torch.ones(3, source.in_features, dtype=torch.float32) / 8
    before = candidate(activation)

    if change in ("load", "load_assign"):
        state = {name: tensor.clone() for name, tensor in candidate.state_dict().items()}
        state["scales"].mul_(2)
        candidate.load_state_dict(state, strict=True, assign=change == "load_assign")
    elif change == "inplace":
        candidate.scales.mul_(2)
    elif change == "replace":
        candidate.scales = candidate.scales.clone() * 2
    elif change == "move":
        candidate.to(torch.float32)
    else:
        candidate.scales.mul_(2)
        candidate.clear_weight_cache()

    codes, channel_scale = _native_int8_oracle_from_checkpoint(candidate.state_dict())
    expected = (activation.double() @ codes.double()) * channel_scale.double()
    with torch.inference_mode():
        actual = candidate(activation)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-3, atol=2e-3)
    assert candidate.int8_module.int8_weight_nk.data_ptr() == candidate.int8_weight_nk.data_ptr()
    if change != "move":
        assert not torch.equal(before, actual)


def test_native_int8_inference_tensor_edit_requires_explicit_clear():
    source, _ = build_case(bits=8, group_size=32)
    candidate = TorchInt8Linear(bits=8, group_size=32, sym=False, desc_act=True,
                                in_features=source.in_features, out_features=source.out_features)
    with torch.inference_mode():
        state = {name: value.clone() for name, value in source.state_dict().items()}
    candidate.load_state_dict(state, strict=True, assign=True)
    candidate.post_init()
    candidate.eval()
    activation = torch.ones(3, source.in_features, dtype=torch.float32) / 8
    before = candidate(activation)
    with torch.inference_mode():
        candidate.scales.mul_(2)
    candidate.clear_weight_cache()
    codes, channel_scale = _native_int8_oracle_from_checkpoint(candidate.state_dict())
    expected = (activation.double() @ codes.double()) * channel_scale.double()
    actual = candidate(activation)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-3, atol=2e-3)
    assert not torch.equal(before, actual)


def test_native_int8_cold_refresh_builds_once_across_threads(monkeypatch):
    source, _ = build_case(bits=8, group_size=32)
    candidate = TorchInt8Linear(bits=8, group_size=32, sym=False, desc_act=True,
                                in_features=source.in_features, out_features=source.out_features)
    candidate.load_state_dict(source.state_dict(), strict=True)
    candidate.post_init()
    candidate.clear_weight_cache()
    original = candidate.transform_cpu
    calls = []

    def tracked(dtype):
        calls.append(dtype)
        sleep(0.01)
        original(dtype)

    monkeypatch.setattr(candidate, "transform_cpu", tracked)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: candidate._refresh_int8_runtime(), range(8)))
    assert calls == [torch.float32]


def test_native_int8_scale_cast_does_not_destroy_canonical_precision():
    weight = torch.arange(64, dtype=torch.int8).reshape(8, 8)
    scales = torch.linspace(.010123, .080987, 8)
    module = Int8PackedModule(weight, scales.clone())
    module(torch.ones(1, 8, dtype=torch.bfloat16))
    torch.testing.assert_close(module.int8_channel_scale, scales, rtol=0, atol=0)
    torch.testing.assert_close(module(torch.ones(1, 8)), weight.float().sum(1)[None] * scales,
                               rtol=1e-6, atol=1e-6)


def test_torch_int8_kernel_is_cpu_only():
    with pytest.raises(NotImplementedError):
        TorchInt8Linear.validate_device(DEVICE.XPU)


def test_torch_int8_supports_native_width_only():
    assert TorchInt8Linear.SUPPORTS_BITS == [8]
    supported, error = TorchInt8Linear.validate(bits=8, group_size=32, pack_dtype=torch.int32,
                                               device=DEVICE.CPU, dynamic={".*": {"bits": 4}})
    assert not supported and error is not None
