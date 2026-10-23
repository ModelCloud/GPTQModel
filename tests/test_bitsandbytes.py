# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import gptqmodel.utils.model as model_utils
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.bitsandbytes import (
    BITSANDBYTES_AVAILABLE,
    BitsAndBytesLinear,
    _cpu_int8_dense_should_use,
    _cpu_packed_kernels,
    _dequantize_int8_weight,
)
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


def _build_cpu_packed_kernel(
    *,
    in_features: int = 128,
    out_features: int = 64,
    register_buffers: bool = False,
    bnb_format: str = "nf4",
    compress_statistics: bool = False,
    block_size: int = 128,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Linear, BitsAndBytesLinear]:
    torch.manual_seed(321)
    linear = nn.Linear(in_features, out_features, bias=True).to(compute_dtype).eval()
    kernel = BitsAndBytesLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=register_buffers,
        format=bnb_format,
        block_size=block_size,
        compress_statistics=compress_statistics,
        dtype=compute_dtype,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    return linear, kernel.eval()


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


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_packing_and_schema_support_inference_and_meta_contexts():
    from gptqmodel.nn_modules.qlinear.bitsandbytes import _buffer_spec_4bit

    _buffer_spec_4bit.cache_clear()
    kwargs = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "in_features": 64,
        "out_features": 128,
        "dtype": torch.float32,
    }
    with torch.inference_mode(), torch.device("meta"):
        shell = BitsAndBytesLinear(**kwargs)
    assert all(value.device.type == "meta" for value in shell.state_dict().values())
    kernel = BitsAndBytesLinear(**kwargs)
    with torch.inference_mode():
        source = torch.nn.Linear(64, 128, bias=False)
        kernel.pack_original(source, None, None)
    assert kernel.dequantize_weight().shape == source.weight.shape
    assert torch.equal(kernel._weight_to_matrix(source), source.weight)


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
def test_bitsandbytes_int8_dequant_matches_native_op():
    from gptqmodel.nn_modules.qlinear.bitsandbytes import import_bitsandbytes

    bnb = import_bitsandbytes()
    weight = torch.randint(-127, 128, (64, 128), dtype=torch.int8)
    weight_scb = torch.rand(64, dtype=torch.float32) + 0.5

    expected = bnb.functional.int8_vectorwise_dequant(weight, weight_scb)
    actual = _dequantize_int8_weight(weight, weight_scb)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_bitsandbytes_int8_dense_dispatch_gate(monkeypatch):
    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    monkeypatch.setattr(bnb_module, "has_avx512_bf16", lambda: True)

    assert _cpu_int8_dense_should_use(
        rows=16,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )
    assert _cpu_int8_dense_should_use(
        rows=32,
        in_features=640,
        out_features=1024,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=15,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=512,
        compute_dtype=torch.float16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=1024,
        out_features=1024,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=2048,
        compute_dtype=torch.bfloat16,
    )

    monkeypatch.setenv("GPTQMODEL_BITSANDBYTES_CPU_INT8_DENSE", "0")
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )

    monkeypatch.setattr(bnb_module, "has_avx512_bf16", lambda: False)
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_int8_small_bf16_uses_dense_without_caching(monkeypatch):
    import bitsandbytes.functional as bnb_functional

    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    monkeypatch.setattr(bnb_module, "has_avx512_bf16", lambda: True)

    torch.manual_seed(501)
    linear = nn.Linear(512, 512, bias=True).to(torch.bfloat16).eval()
    kernel = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=512,
        bias=True,
        dtype=torch.bfloat16,
    ).eval()
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()

    x = torch.randn(16, 512, dtype=torch.bfloat16)
    expected = (
        x.float() @ kernel.dequantize_weight().float().t()
        + kernel.bias.float()
    )

    calls = []
    original_dequant = bnb_module._dequantize_int8_weight

    def counted_dequant(weight, weight_scb):
        calls.append(weight.shape)
        return original_dequant(weight, weight_scb)

    monkeypatch.setattr(bnb_module, "_dequantize_int8_weight", counted_dequant)
    monkeypatch.setattr(
        bnb_functional,
        "int8_linear_matmul",
        lambda *_args, **_kwargs: pytest.fail("native INT8 path should not run"),
    )

    with torch.inference_mode():
        out = kernel(x)

    assert calls == [kernel.weight.shape]
    assert out.dtype == x.dtype
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    assert all(
        tensor.numel() != kernel.in_features * kernel.out_features
        or tensor.dtype == torch.int8
        for tensor in kernel._buffers.values()
    )


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
@pytest.mark.parametrize(
    ("bnb_format", "compress_statistics"),
    [
        ("nf4", False),
        ("nf4", True),
        ("fp4", False),
        ("fp4", True),
    ],
)
def test_bitsandbytes_cpu_packed_kernel_matches_oracle_and_preserves_checkpoint_schema(
    bnb_format: str,
    compress_statistics: bool,
):
    linear, kernel = _build_cpu_packed_kernel(
        bnb_format=bnb_format,
        compress_statistics=compress_statistics,
    )
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)
    canonical_state = {name: tensor.clone() for name, tensor in kernel.state_dict().items()}
    dequant_weight = kernel.dequantize_weight().float()
    expected = torch.matmul(x.float(), dequant_weight.t())
    if kernel.bias is not None:
        expected = expected + kernel.bias.float()

    with torch.inference_mode():
        out = kernel(x)

    assert kernel._cpu_packing_active
    assert out.dtype == torch.bfloat16
    assert tuple(kernel.weight.shape) == (linear.out_features, linear.in_features // 2)
    relative_error = (out.float() - expected).norm() / expected.norm()
    assert relative_error < 4e-3
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

    with torch.inference_mode():
        repeated = kernel(x)
    assert torch.equal(repeated, out)

    state_dict = kernel.state_dict()
    assert not kernel._cpu_packing_active
    for name, tensor in canonical_state.items():
        assert torch.equal(state_dict[name], tensor)
    assert tuple(state_dict["weight"].shape) == (
        linear.out_features * linear.in_features // 2,
        1,
    )
    torch.testing.assert_close(
        kernel.dequantize_weight().float(),
        dequant_weight,
        rtol=1e-6,
        atol=1e-6,
    )

    reloaded = BitsAndBytesLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=True,
        register_buffers=True,
        format=bnb_format,
        block_size=128,
        compress_statistics=compress_statistics,
        dtype=torch.bfloat16,
    ).eval()
    reloaded.load_state_dict(state_dict, strict=True)
    reloaded.post_init()

    with torch.inference_mode():
        reloaded_out = reloaded(x)
    assert reloaded._cpu_packing_active
    assert torch.equal(reloaded_out, out)


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
@pytest.mark.parametrize("block_size", [32, 64, 128, 256, 512])
def test_bitsandbytes_cpu_packed_kernel_supports_bnb_block_sizes(block_size: int):
    linear, kernel = _build_cpu_packed_kernel(
        in_features=512,
        block_size=block_size,
    )
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        out = kernel(x)

    assert kernel._cpu_packing_active
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
def test_bitsandbytes_cpu_packed_kernel_skips_unsupported_shapes_and_env(monkeypatch):
    linear, kernel = _build_cpu_packed_kernel(out_features=24)
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        kernel(x)
    assert not kernel._cpu_packing_active

    _, kernel = _build_cpu_packed_kernel()
    with torch.inference_mode():
        kernel(x)
    assert kernel._cpu_packing_active
    with torch.inference_mode():
        kernel(torch.randn(65, kernel.in_features, dtype=torch.bfloat16))
    assert not kernel._cpu_packing_active

    _, kernel = _build_cpu_packed_kernel(compute_dtype=torch.float16)
    with torch.inference_mode():
        kernel(x.to(torch.float16))
    assert not kernel._cpu_packing_active

    _, kernel = _build_cpu_packed_kernel()
    monkeypatch.setenv("GPTQMODEL_BITSANDBYTES_CPU_PACKED", "0")
    with torch.inference_mode():
        kernel(x)
    assert not kernel._cpu_packing_active


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
def test_bitsandbytes_cpu_packed_kernel_falls_back_when_accuracy_gate_fails(monkeypatch):
    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    monkeypatch.setattr(bnb_module, "_CPU_PACKED_REL_TOL", 0.0)
    linear, kernel = _build_cpu_packed_kernel()
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        out = kernel(x)

    assert not kernel._cpu_packing_active
    assert kernel._cpu_packing_disabled
    assert tuple(kernel.weight.shape) == (
        linear.out_features * linear.in_features // 2,
        1,
    )
    assert torch.isfinite(out).all()


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
