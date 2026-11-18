# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4400 item 2: a quantized MoE reload must not materialize experts.

Defuser's tensor defusion copies each fused 3-D expert projection to the host
and builds one dense ``nn.Linear`` per expert. On the quantized-load path those
linears are replaced by quantized kernels and filled from the checkpoint right
afterwards, so the copies only inflate host memory (OLMoE-1B-7B peaked at
12.17 GiB RssAnon for a 3.66 GiB checkpoint). The loader now parks the fused
projections on meta before defusion; the checkpoint loader materializes the
packed weights straight from disk.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import loader as loader_module
from gptqmodel.models.definitions.olmoe import OlmoeQModel
from gptqmodel.models.ports.native.olmoe.configuration_olmoe import OlmoeConfig
from gptqmodel.models.ports.native.olmoe.modeling_olmoe import OlmoeForCausalLM
from gptqmodel.quantization.config import FORMAT, METHOD
from gptqmodel.utils.hf import suspend_hf_weight_init


class _StopLoad(Exception):
    """Sentinel that aborts the loader once the defuser call is observed."""


def _olmoe_config() -> OlmoeConfig:
    return OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        num_local_experts=4,
        max_position_embeddings=64,
    )


def _tiny_olmoe() -> OlmoeForCausalLM:
    # Mirror the quantized-load shell: parameters are allocated but never
    # initialized, so only the storage footprint matters.
    with suspend_hf_weight_init():
        return OlmoeForCausalLM(_olmoe_config())


def _fused_expert_params(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if "experts" in name and parameter.dim() == 3
    }


def _defused_expert_linears(model):
    linears = []
    for module in model.modules():
        for child_name in ("gate_proj", "up_proj", "down_proj"):
            child = getattr(module, child_name, None)
            if isinstance(child, torch.nn.Linear):
                linears.append(child)
    return linears


def test_meta_fused_expert_params_parks_projections_on_meta():
    model = _tiny_olmoe()
    fused = _fused_expert_params(model)
    assert fused
    assert {parameter.device.type for parameter in fused.values()} == {"cpu"}

    moved = loader_module._meta_fused_expert_params(model)

    assert moved == len(fused)
    assert {parameter.device.type for parameter in _fused_expert_params(model).values()} == {"meta"}


def test_defuser_keeps_meta_experts_off_the_host():
    """Without meta-ization Defuser materializes a bf16 copy per expert."""

    host_model = _tiny_olmoe()
    loader_module._convert_model_with_defuser(SimpleNamespace(), host_model, cleanup_original=True)
    host_linears = _defused_expert_linears(host_model)
    assert host_linears
    # The old path pays for a real dense weight per expert.
    assert {linear.weight.device.type for linear in host_linears} == {"cpu"}
    assert all(linear.weight.numel() > 0 for linear in host_linears)

    meta_model = _tiny_olmoe()
    with torch.no_grad():
        meta_model.model.norm.weight.fill_(7.0)
    loader_module._meta_fused_expert_params(meta_model)
    loader_module._convert_model_with_defuser(SimpleNamespace(), meta_model, cleanup_original=True)
    meta_linears = _defused_expert_linears(meta_model)
    assert meta_linears
    # The quantized path keeps the same structure without host storage.
    assert {linear.weight.device.type for linear in meta_linears} == {"meta"}

    moved = loader_module._materialize_meta_expert_linears(meta_model)
    assert moved == len(meta_linears)
    # ``to_empty`` gives the split experts real host storage without copying
    # values, and leaves the real shell weights alone.
    assert {linear.weight.device.type for linear in meta_linears} == {"cpu"}
    assert torch.equal(meta_model.model.norm.weight, torch.full_like(meta_model.model.norm.weight, 7.0))


def test_quantized_loader_meta_izes_fused_experts_before_defuser(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "olmoe",
                "architectures": ["OlmoeForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_experts_per_tok": 2,
                "num_local_experts": 4,
                "vocab_size": 128,
                "max_position_embeddings": 64,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "quantize_config.json").write_text(
        json.dumps(
            {
                "bits": 4,
                "group_size": 32,
                "sym": True,
                "desc_act": False,
                "format": FORMAT.GPTQ,
                "method": METHOD.GPTQ,
            }
        ),
        encoding="utf-8",
    )

    model = _tiny_olmoe()
    observed = {}

    def fake_convert(cls, captured_model, cleanup_original):
        observed["devices"] = {
            name: parameter.device.type
            for name, parameter in _fused_expert_params(captured_model).items()
        }
        raise _StopLoad()

    monkeypatch.setattr(loader_module, "_convert_model_with_defuser", fake_convert)
    monkeypatch.setattr(loader_module, "get_model_local_path", lambda path, **kwargs: str(path))
    monkeypatch.setattr(loader_module, "load_hf_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        loader_module,
        "get_checkpoints",
        lambda *args, **kwargs: (False, str(checkpoint / "model.safetensors"), "model"),
    )
    class _ShellLoader(type(OlmoeQModel.loader)):
        def from_config(self, config, **kwargs):
            return model

    monkeypatch.setattr(OlmoeQModel, "loader", _ShellLoader(**dataclasses.asdict(OlmoeQModel.loader)))

    with pytest.raises(_StopLoad):
        loader_module.ModelLoader(OlmoeQModel).from_quantized(
            str(checkpoint),
            device="cpu",
        )

    assert observed["devices"]
    assert set(observed["devices"].values()) == {"meta"}
