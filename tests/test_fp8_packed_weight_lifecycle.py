# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""FP8 checkpoint and module moves retain native packed storage."""

import json
import weakref

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights


class _Model(torch.nn.Module):
    def __init__(self, *, fmt="float8_e5m2", scale_method="row"):
        super().__init__()
        self.proj = TorchFP8Linear(
            bits=8,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=32,
            out_features=32,
            bias=True,
            register_buffers=True,
            format=fmt,
            weight_scale_method=scale_method,
        )

    def tie_weights(self, **kwargs):
        del kwargs

    def forward(self, x):
        return self.proj(x)


def _payload(fmt="float8_e5m2", *, scale_method="row", wide_metadata=False):
    values = torch.tensor([0, 1, -1, 2, -2, 4, -4, 8], dtype=torch.float32)
    weight = values.repeat(128).reshape(32, 32).to(getattr(torch, fmt))
    scale_value = 65792.0 if wide_metadata else 2.0
    bias_value = 70000.0 if wide_metadata else 0.5
    scale_shape = () if scale_method == "tensor" else (32,)
    scale = torch.full(scale_shape, scale_value, dtype=torch.float32)
    bias = torch.full((32,), bias_value, dtype=torch.float32)
    return weight, scale, bias


def _install(model, payload):
    model.proj.weight, model.proj.weight_scale_inv, model.proj.bias = (part.clone() for part in payload)


def _assert_packed(module, payload):
    expected_weight, expected_scale, expected_bias = payload
    assert module.weight.dtype == expected_weight.dtype
    assert module.weight.numel() * module.weight.element_size() == 1024
    assert torch.equal(module.weight.cpu().view(torch.uint8), expected_weight.view(torch.uint8))
    assert module.weight_scale_inv.dtype == torch.float32
    assert module.bias.dtype == torch.float32
    torch.testing.assert_close(module.weight_scale_inv.cpu(), expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(module.bias.cpu(), expected_bias, rtol=0, atol=0)
    state = module.state_dict()
    assert state["weight"].dtype == expected_weight.dtype
    assert torch.equal(state["weight"].cpu().view(torch.uint8), expected_weight.view(torch.uint8))
    torch.testing.assert_close(state["weight_scale_inv"].cpu(), expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(state["bias"].cpu(), expected_bias, rtol=0, atol=0)
    assert set(dict(module.named_buffers())) == {"weight", "weight_scale_inv", "bias"}
    assert not any(isinstance(value, torch.Tensor) for value in vars(module).values())


def _assert_oracle_and_no_cache(model, payload, dtype, monkeypatch):
    weight, scale, bias = payload
    x = torch.tensor([[1.0] * 32, [0.5] * 32], dtype=dtype)
    expanded_scale = scale.float() if scale.ndim == 0 else scale.float()[:, None]
    reference_weight = weight.float() / expanded_scale
    oracle = (x.float() @ reference_weight.T + bias.float()).to(dtype)
    refs = []
    temporary_bytes = []
    dequantize = model.proj.dequantize_weight

    def tracked_dequantize(*args, **kwargs):
        temporary = dequantize(*args, **kwargs)
        refs.append(weakref.ref(temporary))
        temporary_bytes.append(temporary.numel() * temporary.element_size())
        return temporary

    monkeypatch.setattr(model.proj, "dequantize_weight", tracked_dequantize)
    with torch.inference_mode():
        first = model(x)
        second = model(x)

    assert first.dtype == dtype
    torch.testing.assert_close(first, oracle, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert temporary_bytes == [4096, 4096]
    assert all(ref() is None for ref in refs)
    _assert_packed(model.proj, payload)


def _save_checkpoint(tmp_path, payload, *, sharded):
    state = dict(zip(("proj.weight", "proj.weight_scale_inv", "proj.bias"), payload))
    if not sharded:
        path = tmp_path / "model.safetensors"
        save_file(state, path, metadata={"format": "pt"})
        return path
    weight_name = "model-00001-of-00002.safetensors"
    metadata_name = "model-00002-of-00002.safetensors"
    save_file({"proj.weight": state["proj.weight"]}, tmp_path / weight_name, metadata={"format": "pt"})
    save_file({key: value for key, value in state.items() if key != "proj.weight"},
              tmp_path / metadata_name, metadata={"format": "pt"})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {
        "proj.weight": weight_name,
        "proj.weight_scale_inv": metadata_name,
        "proj.bias": metadata_name,
    }}), encoding="utf-8")
    return index


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_reload_retains_packed_bytes_and_oracle(tmp_path, monkeypatch, dtype, sharded, fmt):
    payload = _payload(fmt)
    checkpoint = _save_checkpoint(tmp_path, payload, sharded=sharded)
    model = _Model(fmt=fmt)
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=dtype,
        device_map={"": "cpu"},
        keep_in_fp32_modules=["proj.weight_scale_inv", "proj.bias"],
        fp32_bf16_buffer_names=["proj.weight_scale_inv", "proj.bias"],
        packed_fp8_buffer_dtypes={"proj.weight": getattr(torch, fmt)},
    )
    _assert_oracle_and_no_cache(model, payload, dtype, monkeypatch)


def test_fp8_reload_restores_key_converted_packed_weight(tmp_path):
    payload = _payload()
    checkpoint = tmp_path / "model.safetensors"
    state = dict(zip(("proj.weight", "proj.weight_scale_inv", "proj.bias"), payload))
    save_file({f"hf_{name}": value for name, value in state.items()}, checkpoint, metadata={"format": "pt"})
    model = _Model()

    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        keep_in_fp32_modules=["proj.weight_scale_inv", "proj.bias"],
        fp32_bf16_buffer_names=["proj.weight_scale_inv", "proj.bias"],
        packed_fp8_buffer_dtypes={"proj.weight": torch.float8_e5m2},
        key_converter=lambda name: name.removeprefix("hf_"),
    )

    _assert_packed(model.proj, payload)


def test_fp8_reload_rejects_expanded_source_weight(tmp_path):
    packed, scale, bias = _payload()
    checkpoint = tmp_path / "model.safetensors"
    save_file({
        "proj.weight": packed.to(torch.float16),
        "proj.weight_scale_inv": scale,
        "proj.bias": bias,
    }, checkpoint, metadata={"format": "pt"})

    with pytest.raises(RuntimeError, match="packed dtype"):
        load_checkpoint_in_model_then_tie_weights(
            _Model(),
            checkpoint=str(checkpoint),
            dtype=torch.float16,
            device_map={"": "cpu"},
            keep_in_fp32_modules=["proj.weight_scale_inv", "proj.bias"],
            packed_fp8_buffer_dtypes={"proj.weight": torch.float8_e5m2},
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_reload_keeps_fp32_bias_for_cancellation(tmp_path, dtype):
    packed = torch.zeros((32, 32), dtype=torch.float8_e5m2)
    packed[:, 0] = -57344.0
    payload = (
        packed,
        torch.ones((32,), dtype=torch.float32),
        torch.full((32,), 65792.0, dtype=torch.float32),
    )
    checkpoint = _save_checkpoint(tmp_path, payload, sharded=False)
    model = _Model()
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=dtype,
        device_map={"": "cpu"},
        keep_in_fp32_modules=["proj.weight_scale_inv", "proj.bias"],
        fp32_bf16_buffer_names=["proj.weight_scale_inv", "proj.bias"],
        packed_fp8_buffer_dtypes={"proj.weight": torch.float8_e5m2},
    )
    x = torch.zeros((1, 32), dtype=dtype)
    x[0, 0] = 1.1484375
    oracle = (x[:, :1].float() * -57344.0 + 65792.0).expand(1, 32).to(dtype)

    with torch.inference_mode():
        actual = model(x)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)
    _assert_packed(model.proj, payload)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_dtype_move_retains_packed_bytes_and_metadata(monkeypatch, dtype, fmt):
    payload = _payload(fmt)
    model = _Model(fmt=fmt)
    _install(model, payload)
    model.to(dtype=dtype)
    _assert_oracle_and_no_cache(model, payload, dtype, monkeypatch)
    model.float()
    _assert_packed(model.proj, payload)
    restored = _Model(fmt=fmt)
    restored.load_state_dict(model.state_dict())
    _assert_packed(restored.proj, payload)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_dtype_move_retains_wide_scalar_scale_and_bias(dtype):
    payload = _payload(scale_method="tensor", wide_metadata=True)
    model = _Model(scale_method="tensor")
    _install(model, payload)
    model.to(dtype=dtype)
    _assert_packed(model.proj, payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm runtime required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_device_move_retains_native_storage(dtype):
    payload = _payload()
    model = _Model()
    _install(model, payload)
    model.to(device="cuda", dtype=dtype)
    assert model.proj.weight.device.type == "cuda"
    _assert_packed(model.proj, payload)
    model.cpu()
    _assert_packed(model.proj, payload)
