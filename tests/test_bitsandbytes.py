# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import gptqmodel.utils.model as model_utils
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.bitsandbytes import BITSANDBYTES_AVAILABLE, BitsAndBytesLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import BitsAndBytesConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend, select_quant_linear
from gptqmodel.utils.model_dequant import dequantize_model, detect_format


def _build_linear(bits: int) -> nn.Linear:
    torch.manual_seed(100 + bits)
    linear = nn.Linear(32, 24, bias=True).eval()
    if bits == 8:
        linear = linear.half()
    return linear


def _build_kernel(bits: int, linear: nn.Linear) -> BitsAndBytesLinear:
    kwargs = {
        "bits": bits,
        "group_size": -1,
        "sym": True,
        "desc_act": False,
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "bias": linear.bias is not None,
        "register_buffers": False,
    }
    if bits == 4:
        kwargs.update(
            {
                "format": "nf4",
                "block_size": 128,
                "compress_statistics": False,
            }
        )

    kernel = BitsAndBytesLinear(**kwargs)
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    return kernel.eval()


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_kernel_selection():
    assert get_kernel_for_backend(BACKEND.BITSANDBYTES, METHOD.BITSANDBYTES, FORMAT.BITSANDBYTES) is BitsAndBytesLinear

    for bits in (4, 8):
        selected = select_quant_linear(
            bits=bits,
            group_size=-1,
            desc_act=False,
            sym=True,
            backend=BACKEND.BITSANDBYTES,
            format=FORMAT.BITSANDBYTES,
            quant_method=METHOD.BITSANDBYTES,
            device=DEVICE.CPU,
            pack_dtype=torch.int32,
        )
        assert selected is BitsAndBytesLinear


def test_create_quant_module_uses_dynamic_bits_for_bitsandbytes_format_normalization():
    seen = {}

    class _DummyBitsAndBytesLinear(nn.Module):
        @classmethod
        def validate(cls, **kwargs):
            seen["validate_bits"] = kwargs.get("bits")
            return True, None

        def __init__(self, **kwargs):
            super().__init__()
            seen["init_bits"] = kwargs.get("bits")
            seen["format"] = kwargs.get("format")
            self.bias = None

    module = nn.Module()
    module.proj = nn.Linear(32, 32, bias=False)

    model_utils.create_quant_module(
        name="proj",
        linear_cls=_DummyBitsAndBytesLinear,
        bits=4,
        desc_act=False,
        dynamic={
            r"proj": {
                "bits": 8,
                "bnb_quant_type": "int8",
            }
        },
        group_size=-1,
        module=module,
        submodule=module.proj,
        sym=True,
        device=None,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=FORMAT.BITSANDBYTES,
        backend=BACKEND.BITSANDBYTES,
    )

    assert seen["validate_bits"] == 8
    assert seen["init_bits"] == 8
    assert seen["format"] == "int8"
    assert isinstance(module.proj, _DummyBitsAndBytesLinear)


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.parametrize("bits", [4, 8])
def test_bitsandbytes_forward_matches_dequantized_reference(bits: int):
    linear = _build_linear(bits)
    kernel = _build_kernel(bits, linear)

    x = torch.randn(7, linear.in_features, dtype=torch.float32)
    dequant_weight = kernel.dequantize_weight().to(torch.float32)
    expected = torch.matmul(x, dequant_weight.t())
    if kernel.bias is not None:
        expected = expected + kernel.bias.to(torch.float32)

    with torch.inference_mode():
        out = kernel(x)

    atol = 1e-3 if bits == 4 else 1e-2
    rtol = 1e-3 if bits == 4 else 5e-2
    torch.testing.assert_close(out.to(torch.float32), expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.parametrize("bits", [4, 8])
def test_bitsandbytes_state_dict_round_trip(bits: int):
    linear = _build_linear(bits)
    kernel = _build_kernel(bits, linear)

    reload_kwargs = {
        "bits": bits,
        "group_size": -1,
        "sym": True,
        "desc_act": False,
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "bias": linear.bias is not None,
        "register_buffers": True,
    }
    if bits == 4:
        reload_kwargs.update(
            {
                "format": "nf4",
                "block_size": 128,
                "compress_statistics": False,
            }
        )

    reloaded = BitsAndBytesLinear(**reload_kwargs).eval()
    reloaded.load_state_dict(kernel.state_dict(), strict=True)
    reloaded.post_init()

    x = torch.randn(5, linear.in_features, dtype=torch.float32)
    with torch.inference_mode():
        torch.testing.assert_close(
            reloaded.dequantize_weight().to(torch.float32),
            kernel.dequantize_weight().to(torch.float32),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            reloaded(x).to(torch.float32),
            kernel(x).to(torch.float32),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.parametrize("bits", [4, 8])
def test_detect_and_dequantize_bitsandbytes_checkpoint(tmp_path, bits: int):
    linear = _build_linear(bits)
    kernel = _build_kernel(bits, linear)

    prefix = "model.layers.0.mlp.up_proj"
    state_dict = {f"{prefix}.{name}": tensor for name, tensor in kernel.state_dict().items()}

    model_dir = tmp_path / f"bnb_{bits}bit"
    model_dir.mkdir()
    save_file(state_dict, str(model_dir / "model.safetensors"))

    quant_cfg = BitsAndBytesConfig(
        bits=bits,
        format="nf4" if bits == 4 else "int8",
        block_size=128 if bits == 4 else 64,
        compress_statistics=False if bits == 4 else True,
    )
    config_payload = {"quantization_config": quant_cfg.to_dict()}
    (model_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")

    detected = detect_format(model_dir, config_payload)
    assert detected == "bitsandbytes"

    output_dir = tmp_path / f"bnb_{bits}bit_dequantized"
    dequantize_model(model_dir, output_dir, target_dtype=torch.float16)

    with safe_open(output_dir / "model.safetensors", framework="pt", device="cpu") as reader:
        weight = reader.get_tensor(f"{prefix}.weight")
        bias = reader.get_tensor(f"{prefix}.bias")

    torch.testing.assert_close(
        weight.to(torch.float32),
        kernel.dequantize_weight().to(torch.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        bias.to(torch.float32),
        kernel.bias.to(torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
