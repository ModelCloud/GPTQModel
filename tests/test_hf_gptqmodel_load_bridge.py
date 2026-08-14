# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.models.laguna.configuration_laguna import LagunaConfig

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import (
    HFGPTQModelLoadContext,
    _checkpoint_quantized_module_names,
    hf_gptqmodel_post_init_for_load,
    hf_gptqmodel_prepare_model_for_load,
)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(128, 32, bias=False, device="meta")
        self.g_proj = torch.nn.Linear(128, 48, bias=False, device="meta")


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Model(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            _name_or_path=str(checkpoint_path),
            dtype=torch.float16,
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block()])


def _write_checkpoint(checkpoint_path, *, native=True, missing_module=False):
    quantizer = ["gptqmodel:test"] if native else ["optimum:test"]
    dynamic = None
    if native:
        dynamic = {
            r"+:^model\.layers\.0\.self_attn\.q_proj$": {"bits": 4, "group_size": 32},
            r"+:^model\.layers\.0\.self_attn\.g_proj$": {"bits": 4, "group_size": 32},
        }
    config = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "format": "gptq",
        "quant_method": "gptq",
        "dynamic": dynamic,
        "meta": {"quantizer": quantizer},
    }
    (checkpoint_path / "quantize_config.json").write_text(json.dumps(config), encoding="utf-8")
    module_name = "model.layers.0.self_attn.missing_proj" if missing_module else "model.layers.0.self_attn.q_proj"
    save_file(
        {f"{module_name}.qweight": torch.zeros((16, 32), dtype=torch.int32)},
        checkpoint_path / "model.safetensors",
    )


def test_hf_load_bridge_uses_checkpoint_manifest_and_preserves_dense_module(tmp_path):
    _write_checkpoint(tmp_path)
    model = _Model(tmp_path)

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
    )

    q_proj = model.model.layers[0].self_attn.q_proj
    g_proj = model.model.layers[0].self_attn.g_proj
    assert isinstance(q_proj, TorchLinear)
    assert q_proj.group_size == 32
    assert q_proj.qweight.device.type == "meta"
    assert isinstance(g_proj, torch.nn.Linear)
    assert g_proj.out_features == 48
    assert context.quantize_config.dynamic is not None


def test_hf_load_bridge_leaves_legacy_optimum_checkpoint_to_existing_path(tmp_path):
    _write_checkpoint(tmp_path, native=False)
    model = _Model(tmp_path)

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
    )

    assert context is None
    assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)


def test_hf_load_bridge_discovers_qvq_modules_from_trellis_payload(tmp_path):
    config = {
        "bits": 2,
        "group_size": -1,
        "desc_act": False,
        "sym": True,
        "format": "qvq",
        "method": "qvq",
    }
    (tmp_path / "quantize_config.json").write_text(json.dumps(config), encoding="utf-8")
    prefix = "model.layers.0.self_attn.q_proj"
    save_file(
        {
            f"{prefix}.trellis": torch.zeros((16, 16), dtype=torch.int32),
            f"{prefix}.SU": torch.ones(128, dtype=torch.float16),
            f"{prefix}.SV": torch.ones(32, dtype=torch.float16),
        },
        tmp_path / "model.safetensors",
    )
    model = _Model(tmp_path)

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.QVQ,
        dtype=torch.float16,
    )

    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, QVQLinear)
    assert q_proj.trellis.device.type == "meta"
    assert isinstance(model.model.layers[0].self_attn.g_proj, torch.nn.Linear)
    assert context.quant_linear is QVQLinear
    assert context.quantize_config.format.value == "qvq"


def test_checkpoint_module_discovery_uses_format_owned_weight_payload():
    qvq_config = QuantizeConfig(method="qvq", format="qvq", bits=2)
    gptq_config = QuantizeConfig(method="gptq", format="gptq", bits=4)
    awq_config = QuantizeConfig(method="awq", format="gemm", bits=4)
    keys = {
        "model.q_proj.trellis",
        "model.q_proj.SU",
        "model.q_proj.SV",
        "model.k_proj.qweight",
    }

    assert _checkpoint_quantized_module_names(keys, qvq_config) == ("model.q_proj",)
    assert _checkpoint_quantized_module_names(keys, gptq_config) == ("model.k_proj",)
    assert _checkpoint_quantized_module_names(keys, awq_config) == ("model.k_proj",)
    assert _checkpoint_quantized_module_names(
        keys,
        gptq_config,
        candidates={"model.q_proj", "model.k_proj", "model.dense"},
    ) == ("model.k_proj",)
    assert _checkpoint_quantized_module_names(
        keys,
        awq_config,
        candidates={"model.q_proj", "model.k_proj", "model.dense"},
    ) == ("model.k_proj",)
    assert _checkpoint_quantized_module_names(
        keys,
        qvq_config,
        candidates={"model.k_proj"},
    ) == ()


def test_hf_load_bridge_fails_when_checkpoint_module_is_missing_after_conversion(tmp_path):
    _write_checkpoint(tmp_path, missing_module=True)

    with pytest.raises(ValueError, match="missing_proj"):
        hf_gptqmodel_prepare_model_for_load(
            _Model(tmp_path),
            checkpoint_files=[tmp_path / "model.safetensors"],
            device_map={"": "cpu"},
            backend=BACKEND.GPTQ_TORCH,
            dtype=torch.float16,
        )


def test_hf_load_bridge_defuses_laguna_experts_on_meta(tmp_path):
    config = LagunaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["full_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse"],
        num_attention_heads_per_layer=[2, 2],
    )
    config._name_or_path = str(tmp_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    quantize_config = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "format": "gptq",
        "quant_method": "gptq",
        "meta": {"quantizer": ["gptqmodel:test"]},
    }
    (tmp_path / "quantize_config.json").write_text(json.dumps(quantize_config), encoding="utf-8")
    save_file(
        {
            "model.layers.0.self_attn.q_proj.qweight": torch.zeros((8, 64), dtype=torch.int32),
            "model.layers.1.mlp.experts.0.gate_proj.qweight": torch.zeros((8, 32), dtype=torch.int32),
        },
        tmp_path / "model.safetensors",
    )

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
    )

    assert isinstance(model.model.layers[0].self_attn.q_proj, TorchLinear)
    assert isinstance(model.model.layers[0].self_attn.g_proj, torch.nn.Linear)
    assert model.model.layers[0].self_attn.g_proj.out_features == 2
    assert isinstance(model.model.layers[1].mlp.experts[0].gate_proj, TorchLinear)
    assert all(parameter.is_meta for parameter in model.parameters())
    assert context.quant_linear is TorchLinear


def test_hf_load_post_init_restores_gptqmodel_config_fields(monkeypatch):
    from transformers import GPTQConfig

    import gptqmodel.utils.model as model_utils

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        dynamic={r"+:^model\.layers\.0\.self_attn\.q_proj$": {"group_size": 32}},
        lm_head=True,
    )
    context = HFGPTQModelLoadContext(
        quantize_config=qcfg,
        quant_linear=TorchLinear,
    )
    model = torch.nn.Module()
    model.config = SimpleNamespace(quantization_config=GPTQConfig(bits=4))

    monkeypatch.setattr(model_utils, "hf_convert_gptq_v1_to_v2_format", lambda model, **kwargs: (model, False))
    monkeypatch.setattr(model_utils, "hf_gptqmodel_post_init", lambda model, **kwargs: model)

    assert hf_gptqmodel_post_init_for_load(model, context) is model
    assert model.config.quantization_config.dynamic == qcfg.dynamic
    assert model.config.quantization_config.lm_head is True
    assert model.config.quantization_config.method == qcfg.method
    assert model.config.quantization_config.pack_dtype == "int32"
