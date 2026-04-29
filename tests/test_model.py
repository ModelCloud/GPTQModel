# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# -- do not touch
import os
import tempfile

from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.utils.torch import torch_empty_cache


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import json
import sys
import types
import unittest  # noqa: E402
from importlib.metadata import PackageNotFoundError
from pathlib import Path

# isort: off
# isort: on
from parameterized import parameterized  # noqa: E402
from torch import nn

from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.looper.module_looper import ModuleLooper, StopMainLoop
from gptqmodel.models import loader
from gptqmodel.models.auto import _hide_unsupported_quantization_config_for_eval, _is_supported_quantization_config
from gptqmodel.models.definitions.phi4 import Phi4MMGPTQ


############ test_model_dequant.py ############

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.quantization.dtype import (
    available_float8_dtype_names,
    dequantize_f4_e2m1,
    dequantize_fp8,
)
from gptqmodel.utils.model_dequant import dequantize_model


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
except Exception:
    nvfp4_quantize = None


pytestmark = [pytest.mark.model, pytest.mark.slow, pytest.mark.gpu]


def pack_cols(values: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack per-column low-bit values into int32 words."""

    if values.dtype != torch.int32:
        values = values.to(torch.int32)

    rows, cols = values.shape
    pack_factor = 32 // bits
    if cols % pack_factor != 0:
        raise ValueError("columns must be divisible by pack factor")

    packed_cols = cols // pack_factor
    packed = torch.zeros(rows, packed_cols, dtype=torch.int32)
    mask = (1 << bits) - 1
    for col in range(cols):
        group = col // pack_factor
        shift = (col % pack_factor) * bits
        packed[:, group] |= (values[:, col] & mask) << shift
    return packed


def write_index(path: Path, shard: str, keys: list[str]) -> None:
    weight_map = dict.fromkeys(keys, shard)
    payload = {"weight_map": weight_map}
    (path / "model.safetensors.index.json").write_text(json.dumps(payload))

def _checkpoint_roundtrip_fp8_formats() -> list[str]:
    formats = []
    for format_name in available_float8_dtype_names():
        dtype = getattr(torch, format_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.safetensors"
            try:
                save_file({"probe.weight": torch.zeros((2, 2), dtype=dtype)}, str(path))
            except Exception:
                continue
        formats.append(format_name)
    return formats


@pytest.mark.parametrize("format_name", _checkpoint_roundtrip_fp8_formats())
def test_dequantize_model_fp8_infers_block_size(tmp_path, format_name: str):
    model_dir = tmp_path / "fp8_model_infer"
    output_dir = tmp_path / "fp8_output_infer"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "format": format_name,
            "quant_method": "fp8",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    weight = torch.randn(4, 8, dtype=torch.float32).to(getattr(torch, format_name))
    scale_inv = torch.ones(2, 2, dtype=torch.float32)
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight,
            "linear.weight_scale_inv": scale_inv,
        },
        str(model_dir / shard_name),
    )
    write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv"])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        weight_out = reader.get_tensor("linear.weight")
        assert weight_out.dtype is torch.bfloat16

    expected = dequantize_fp8(weight, scale_inv=scale_inv, axis=None, target_dtype=torch.bfloat16)
    assert torch.equal(weight_out, expected)


@pytest.mark.parametrize("format_name", _checkpoint_roundtrip_fp8_formats())
def test_dequantize_model_fp8(tmp_path, format_name: str):
    model_dir = tmp_path / "fp8_model"
    output_dir = tmp_path / "fp8_output"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "format": format_name,
            "quant_method": "fp8",
            "weight_block_size": [2, 4],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    weight = torch.randn(2, 4, dtype=torch.float32).to(getattr(torch, format_name))
    scale_inv = torch.ones(1, 1, dtype=torch.float32)
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight,
            "linear.weight_scale_inv": scale_inv,
            "linear.bias": torch.randn(4, dtype=torch.float32),
        },
        str(model_dir / shard_name),
    )
    write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        assert "linear.weight" in reader.keys()
        assert "linear.weight_scale_inv" not in reader.keys()
        weight_out = reader.get_tensor("linear.weight")
        bias_out = reader.get_tensor("linear.bias")

    expected = dequantize_fp8(weight, scale_inv=scale_inv, axis=None, target_dtype=torch.bfloat16)
    assert torch.equal(weight_out, expected)
    assert bias_out.dtype is torch.bfloat16

    updated_config = json.loads((output_dir / "config.json").read_text())
    assert "quantization_config" not in updated_config
    assert updated_config.get("dtype") == "bfloat16"
    assert "torch_dtype" not in updated_config

    new_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert "linear.weight" in new_index["weight_map"]
    assert "linear.weight_scale_inv" not in new_index["weight_map"]


@pytest.mark.skipif(nvfp4_quantize is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="float4 packed dtype not available")
def test_dequantize_model_nvfp4_float4_storage(tmp_path):
    model_dir = tmp_path / "nvfp4_model"
    output_dir = tmp_path / "nvfp4_output"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "format": "nvfp4",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    data = torch.randn(4, 16, dtype=torch.float32)
    scales, packed = nvfp4_quantize(data, block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2)
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": packed_float4,
            "linear.weight_scale": scales,
        },
        str(model_dir / shard_name),
    )
    write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale"])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        assert "linear.weight" in reader.keys()
        assert "linear.weight_scale" not in reader.keys()
        weight_out = reader.get_tensor("linear.weight")

    expected = dequantize_f4_e2m1(packed_float4, scale=scales, axis=None, target_dtype=torch.bfloat16)
    assert torch.allclose(weight_out, expected, atol=1e-3, rtol=1e-3)

    new_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert "linear.weight" in new_index["weight_map"]
    assert "linear.weight_scale" not in new_index["weight_map"]


def test_dequantize_model_awq(tmp_path):
    model_dir = tmp_path / "awq_model"
    output_dir = tmp_path / "awq_output"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "quant_method": "awq",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    rows, cols = 8, 16
    weight_values = torch.randint(0, 16, (rows, cols), dtype=torch.int32)
    zero_values = torch.randint(0, 16, (rows, cols), dtype=torch.int32)
    scales = torch.rand(rows, cols, dtype=torch.float32) * 0.5 + 0.5
    bias = torch.randn(cols, dtype=torch.float32)

    packed_weight = pack_cols(weight_values)
    packed_zero = pack_cols(zero_values)

    shard_name = "awq.safetensors"
    save_file(
        {
            "layer.qweight": packed_weight,
            "layer.qzeros": packed_zero,
            "layer.scales": scales,
            "layer.bias": bias,
        },
        str(model_dir / shard_name),
    )
    write_index(
        model_dir,
        shard_name,
        ["layer.qweight", "layer.qzeros", "layer.scales", "layer.bias"],
    )

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        assert "layer.weight" in keys
        assert "layer.qweight" not in keys
        assert "layer.qzeros" not in keys
        weight_out = reader.get_tensor("layer.weight")
        bias_out = reader.get_tensor("layer.bias")

    expected = ((weight_values.float() - zero_values.float()) * scales).t().contiguous().to(torch.bfloat16)
    assert torch.equal(weight_out, expected)
    assert bias_out.dtype is torch.bfloat16

    updated_config = json.loads((output_dir / "config.json").read_text())
    assert "quantization_config" not in updated_config

    new_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert "layer.weight" in new_index["weight_map"]
    assert "layer.qweight" not in new_index["weight_map"]


def test_dequantize_model_compressed_tensors_pack(tmp_path):
    pytest.importorskip("compressed_tensors")
    pytest.importorskip("transformers")

    from compressed_tensors.compressors.quantized_compressors.pack_quantized import pack_to_int32
    from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
    from compressed_tensors.quantization.quant_args import QuantizationArgs
    from compressed_tensors.quantization.utils import calculate_qparams
    from transformers import LlamaConfig

    model_dir = tmp_path / "compressed_model"
    output_dir = tmp_path / "compressed_output"
    model_dir.mkdir()

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
    )

    quant_cfg = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    config_dict = config.to_dict()
    config_dict["quantization_config"] = quant_cfg
    (model_dir / "config.json").write_text(json.dumps(config_dict))

    weight_cfg = quant_cfg["config_groups"]["group_0"]["weights"]
    quant_args = QuantizationArgs(**weight_cfg)

    def compress_weight(prefix: str, weight: torch.Tensor) -> tuple[dict, torch.Tensor]:
        rows, cols = weight.shape
        group_size = quant_args.group_size or cols
        groups = cols // group_size
        reshaped = weight.view(rows, groups, group_size)
        min_vals = reshaped.amin(dim=-1)
        max_vals = reshaped.amax(dim=-1)
        scales, zero_points = calculate_qparams(min_vals, max_vals, quant_args)

        quantized = quantize(
            weight,
            scale=scales,
            zero_point=zero_points,
            args=quant_args,
            dtype=torch.int8,
        )
        packed = pack_to_int32(quantized, quant_args.num_bits)
        expected = dequantize(
            quantized,
            scale=scales,
            zero_point=zero_points,
            args=quant_args,
            dtype=torch.float32,
        )
        payload = {
            f"{prefix}.weight_packed": packed,
            f"{prefix}.weight_scale": scales,
            f"{prefix}.weight_shape": torch.tensor(weight.shape, dtype=torch.int32),
        }
        return payload, expected

    prefix_q = "model.layers.0.self_attn.q_proj"
    prefix_k = "model.layers.0.self_attn.k_proj"

    hidden = config.hidden_size
    base_weight = torch.linspace(-0.75, 0.75, steps=hidden * hidden, dtype=torch.float32).view(
        hidden, hidden
    )
    payload_q, expected_q = compress_weight(prefix_q, base_weight)
    payload_k, expected_k = compress_weight(prefix_k, base_weight.neg())

    shard_name = "model.safetensors"
    tensors = {**payload_q, **payload_k}
    save_file(tensors, str(model_dir / shard_name))
    write_index(model_dir, shard_name, list(tensors.keys()))

    import gptqmodel.utils.model_dequant as model_dequant_module

    module_path = Path(model_dequant_module.__file__).resolve()
    assert REPO_ROOT in module_path.parents
    detected = model_dequant_module.detect_format(
        model_dir, model_dequant_module.load_json(model_dir / "config.json")
    )
    assert detected == "compressed-pack"

    dequantize_model(model_dir, output_dir, target_dtype=torch.float32, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        assert f"{prefix_q}.weight" in keys
        assert f"{prefix_k}.weight" in keys
        assert all(not key.endswith(("_packed", "_scale", "_shape")) for key in keys)

        weight_q = reader.get_tensor(f"{prefix_q}.weight")
        weight_k = reader.get_tensor(f"{prefix_k}.weight")

        torch.testing.assert_close(weight_q, expected_q)
        torch.testing.assert_close(weight_k, expected_k)

    updated_config = json.loads((output_dir / "config.json").read_text())
    assert "quantization_config" not in updated_config

    new_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert f"{prefix_q}.weight" in new_index["weight_map"]
    assert f"{prefix_k}.weight" in new_index["weight_map"]
    assert f"{prefix_q}.weight_scale" not in new_index["weight_map"]

############ test_model_require_pkgs.py ############

class DummyRequirePkgModel:
    require_pkgs = ["fakepkg>=1.0.0"]


def test_phi4_model_definition_requires_required_packages():
    assert "backoff>=2.2.1" in Phi4MMGPTQ.require_pkgs
    assert "optimum>=1.24.0" in Phi4MMGPTQ.require_pkgs


def test_check_versions_passes_when_version_matches(monkeypatch):
    monkeypatch.setattr(loader, "version", lambda _: "1.0.0")

    loader.check_versions(
        DummyRequirePkgModel,
        DummyRequirePkgModel.require_pkgs
    )


def test_check_versions_raises_when_version_mismatch(monkeypatch):
    # not install
    def fake_version(pkg):
        raise PackageNotFoundError(pkg)

    monkeypatch.setattr(loader, "version", fake_version)

    with pytest.raises(ValueError, match="not installed"):
        loader.check_versions(
            DummyRequirePkgModel,
            DummyRequirePkgModel.require_pkgs
        )

    # version mismatch
    monkeypatch.setattr(loader, "version", lambda _: "0.1.0")

    with pytest.raises(ValueError, match="but current fakepkg version"):
        loader.check_versions(
            DummyRequirePkgModel,
            DummyRequirePkgModel.require_pkgs
        )

############ test_model_save.py ############

class TestModelSave(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pretrained_model_id = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_id, use_fast=True)

        traindata = load_dataset(path="/monster/data/model/dataset/nm-calibration", name="LLM", split="train")
        cls.calibration_dataset = traindata.select(range(1))

    @parameterized.expand([
        True,
        False,
    ])
    def test_model_save_with_non_persistent_buffer(self, offload_to_disk):
        quantize_config = QuantizeConfig(
            bits=4,
            offload_to_disk=offload_to_disk,
        )

        model = GPTQModel.load(
            self.pretrained_model_id,
            quantize_config=quantize_config,
        )
        model.quantize(self.calibration_dataset, batch_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            model.save(tmp_dir_name)

            del model
            torch_empty_cache()

            with safe_open(tmp_dir_name+"/model.safetensors", framework="pt") as f:
                print("weight_map", f.keys())
                self.assertNotIn('model.rotary_emb.inv_freq', f.keys())

    def test_moe(self):
        quantize_config = QuantizeConfig(
            fallback=None,
        )

        model = GPTQModel.load(
            "/monster/data/model/Qwen3-30B-A3B-layers-1/",
            quantize_config=quantize_config,
        )

        assert len(self.calibration_dataset) == 1
        model.quantize(self.calibration_dataset, batch_size=1)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            model.save(tmp_dir_name)

            del model
            torch_empty_cache()

            new_model = GPTQModel.load(tmp_dir_name, device="cuda")
            print("new_model", new_model)

            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[2].gate_proj, MarlinLinear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[2].up_proj, MarlinLinear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[2].down_proj, MarlinLinear)

            # No calibration data was routed to these MoE expert modules.
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[10].gate_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[10].up_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[10].down_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[15].gate_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[15].up_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[15].down_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[20].gate_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[20].up_proj, nn.Linear)
            self.assertIsInstance(new_model.model.model.layers[0].mlp.experts[20].down_proj, nn.Linear)



############ test_module_looper_callback.py ############

class DummyQModel:
    def __init__(self):
        self.support_batch_quantize = False
        self.quantize_config = types.SimpleNamespace(device=None, moe_routing_bypass=lambda: None)
        self.layer_callback = None


def make_looper(layer_callback=None):
    model = DummyQModel()
    if layer_callback is not None:
        model.layer_callback = layer_callback
    processors = [types.SimpleNamespace()]
    return ModuleLooper(model=model, processors=processors)


def test_callbackup_invokes_model_layer_callback():
    calls = []

    class Recorder:
        def layer_complete(self, *, layer_idx, submodule_finalized):
            calls.append((layer_idx, submodule_finalized))

    looper = make_looper(layer_callback=Recorder())

    looper.callbackup(layer_idx=3, submodule_finalized=False)
    looper.callbackup(layer_idx=3, submodule_finalized=True)

    assert calls == [(3, False), (3, True)]


def test_callbackup_stop_request_via_returning_class():
    def stopper(**_):
        return StopMainLoop

    looper = make_looper(layer_callback=stopper)

    with pytest.raises(StopMainLoop):
        looper.callbackup(layer_idx=1, submodule_finalized=False)


def test_callbackup_stop_request_via_instance():
    def stopper(**_):
        return StopMainLoop("stop")

    looper = make_looper(layer_callback=stopper)

    with pytest.raises(StopMainLoop):
        looper.callbackup(layer_idx=1, submodule_finalized=False)


def test_emit_layer_complete_records_stop(monkeypatch):
    err = ValueError("boom")

    def raising_callback(*, layer_idx, submodule_finalized):
        raise err

    looper = make_looper(layer_callback=raising_callback)

    looper._emit_layer_complete(
        layer_idx=7,
        submodule_finalized=False,
        raise_in_place=False,
    )

    assert looper._loop_stop_exc is err
    assert looper._loop_stop_event.is_set()

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.wait",
        lambda *_, **__: None,
    )

    with pytest.raises(ValueError) as exc:
        looper._check_loop_stop()

    assert exc.value is err


def test_emit_layer_complete_propagates_when_requested():
    err = RuntimeError("direct")

    def raising_callback(*, layer_idx, submodule_finalized):
        raise err

    looper = make_looper(layer_callback=raising_callback)

    with pytest.raises(RuntimeError) as exc:
        looper._emit_layer_complete(
            layer_idx=2,
            submodule_finalized=True,
            raise_in_place=True,
        )

    assert exc.value is err


def test_emit_layer_complete_stops_cleanly_on_stop_main_loop(monkeypatch):
    class Stopper:
        def layer_complete(self, *, layer_idx, submodule_finalized):
            raise StopMainLoop()

    looper = make_looper(layer_callback=Stopper())

    looper._emit_layer_complete(
        layer_idx=0,
        submodule_finalized=True,
        raise_in_place=True,
    )

    assert looper._loop_stop_exc is None
    assert looper._loop_stop_event.is_set()

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.wait",
        lambda *_, **__: None,
    )

    assert looper._check_loop_stop() is True


def test_hide_unsupported_quantization_config_for_eval_temporarily_clears_gguf_bits():
    quantization_config = {
        "quant_method": "gguf",
        "format": "gguf",
        "bits": "q4_k_m",
    }
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(quantization_config=dict(quantization_config))
    )

    with _hide_unsupported_quantization_config_for_eval(model):
        assert model.config.quantization_config is None

    assert model.config.quantization_config == quantization_config


def test_hide_unsupported_quantization_config_for_eval_leaves_supported_gptq_alone():
    quantization_config = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
    }
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(quantization_config=dict(quantization_config))
    )

    with _hide_unsupported_quantization_config_for_eval(model):
        assert model.config.quantization_config == quantization_config

    assert model.config.quantization_config == quantization_config


def test_is_supported_quantization_config_rejects_input_activation_quantization():
    config = types.SimpleNamespace(
        quantization_config={
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": {
                    "input_activations": {"num_bits": 4, "type": "float", "dynamic": False},
                    "weights": {"num_bits": 4, "type": "float", "dynamic": False},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="activation quantized models"):
        _is_supported_quantization_config(config)


def test_is_supported_quantization_config_rejects_kv_cache_quantization():
    config = types.SimpleNamespace(
        quantization_config={
            "quant_method": "modelopt",
            "kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": False},
        }
    )

    with pytest.raises(ValueError, match="activation quantized models"):
        _is_supported_quantization_config(config)
