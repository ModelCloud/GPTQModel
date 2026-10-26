# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.

from __future__ import annotations

import weakref

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch_int8 import Int8PackedModule, TorchInt8Linear, _write_int8_buffers
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
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
    original = candidate.dequantize_weight

    def tracked(*args, **kwargs):
        dense = original(*args, **kwargs)
        references.append(weakref.ref(dense))
        return dense

    monkeypatch.setattr(candidate, "dequantize_weight", tracked)
    candidate.post_init()
    candidate.eval()
    assert all(reference() is None for reference in references)
    torch.testing.assert_close(candidate.int8_weight_nk, codes.t(), rtol=0, atol=0)
    torch.testing.assert_close(candidate.int8_channel_scale, channel_scale, rtol=0, atol=0)
    assert all(not hasattr(candidate, name) for name in candidate.GPTQ_BUFFER_NAMES)
    for current_dtype in (dtype, torch.bfloat16, torch.float32, dtype):
        activation = torch.arange(7 * source.in_features).reshape(7, -1).remainder(9).sub(4).to(current_dtype) / 16
        raw = (activation.float() @ codes.float()) * channel_scale.to(current_dtype).float()
        with torch.inference_mode():
            output = candidate(activation)
        assert output.dtype == current_dtype
        torch.testing.assert_close(output, raw.to(current_dtype), rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(candidate.int8_module.int8_channel_scale, channel_scale, rtol=0, atol=0)
        allowed = {candidate.int8_weight_nk.data_ptr(), candidate.int8_channel_scale.data_ptr()}
        assert all(tensor.untyped_storage().data_ptr() in allowed for tensor in state_tensors(candidate))
        assert candidate.int8_weight_nk.untyped_storage().nbytes() == source.in_features * source.out_features
        assert all(reference() is None for reference in references)


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
