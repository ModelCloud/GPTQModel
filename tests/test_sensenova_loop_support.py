# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Support contracts for OpenSenseNova's SenseNova-Loop decoder.

`sensenova/SenseNova-Loop-9B` is gated, so these are definition-level
contracts pinned to the public `SenseNova-Loop-9B-Preview` checkpoint layout
and to the upstream `modeling_sensenova_loop.py` semantics, which GPT-QModel
now owns as `gptqmodel.models.ports.sensenova_loop`. The tensor suffix
sets below are copied from that checkpoint's `model.safetensors.index.json`;
they are what proves the declared tree resolves against real weight names.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.sensenova_loop import SenseNovaLoopQModel
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.utils.model import find_modules
from gptqmodel.utils.security import BYPASS_ENV


# Every tensor of a full-attention layer (`layer_types[i] == "full_attention"`).
FULL_ATTENTION_LAYER_SUFFIXES = frozenset(
    {
        "input_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
)

# Every tensor of a linear-attention layer, plus the mHC routers that the
# looped block 4..27 adds on top of it.
LINEAR_ATTENTION_LAYER_SUFFIXES = frozenset(
    {
        "input_layernorm.weight",
        "linear_attn.A_log",
        "linear_attn.conv1d.weight",
        "linear_attn.dt_bias",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "mhc_attn.bias",
        "mhc_attn.weight",
        "mhc_ffn.bias",
        "mhc_ffn.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
)

CHECKPOINT_LAYER_SUFFIXES = FULL_ATTENTION_LAYER_SUFFIXES | LINEAR_ATTENTION_LAYER_SUFFIXES


def _declared_entries(definition):
    return [
        entry
        for block in definition.build_layer_modules(definition.module_tree, include_capture_only=True)
        for entry in block
    ]


def _strip_flags(entry):
    return entry.split(":", 1)[0]


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "1"])
def test_sensenova_loop_model_type_selects_definition_without_trust_or_bypass(tmp_path, monkeypatch, trust, bypass):
    checkpoint = str(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "sensenova_loop",
        "auto_map": {"AutoConfig": "configuration_sensenova_loop.SenseNovaLoopConfig"},
    }))
    if bypass is None:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    else:
        monkeypatch.setenv(BYPASS_ENV, bypass)

    def load_config(model_dir, *, trust_remote_code):
        assert model_dir == checkpoint
        # The package-owned loader never forwards a trust request.
        assert trust_remote_code is False
        return SimpleNamespace(model_type="sensenova_loop")

    monkeypatch.setattr(auto, "load_model_config", load_config)
    assert auto.check_and_get_model_definition(checkpoint, trust_remote_code=trust) is SenseNovaLoopQModel
    assert MODEL_MAP["sensenova_loop"] is SenseNovaLoopQModel


def test_sensenova_loop_is_package_owned_and_loads_a_processor():
    # The port in `gptqmodel.models.ports.sensenova_loop` replaces the
    # checkpoint's `modeling_sensenova_loop.py`; no remote code is needed.
    assert isinstance(SenseNovaLoopQModel.loader, LocalModelLoader)
    assert SenseNovaLoopQModel.require_trust_remote_code is False
    assert SenseNovaLoopQModel.require_load_processor is True
    assert SenseNovaLoopQModel.layer_modules_strict is False


def test_sensenova_loop_uses_the_vl_decoder_layout():
    assert SenseNovaLoopQModel.extract_layers_node() == ["model.language_model.layers"]
    assert SenseNovaLoopQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert SenseNovaLoopQModel.rotary_embedding == "model.language_model.rotary_emb"
    # `mtp.safetensors` carries `mtp.*` draft-head tensors the model
    # itself ignores; they must survive a quantize/save round trip.
    assert "mtp" in SenseNovaLoopQModel.out_of_model_tensors["prefixes"]


def test_sensenova_loop_tree_tracks_the_qwen3_5_decoder():
    # The loop reuses Qwen3.5's physical layers, so the declared tree must stay
    # identical: a Qwen3.5 tree fix must not be silently skipped here.
    assert SenseNovaLoopQModel.module_tree == Qwen3_5QModel.module_tree


def test_sensenova_loop_quantizes_only_decoder_projections():
    layer_modules = SenseNovaLoopQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["linear_attn.in_proj_qkv"],
        ["linear_attn.in_proj_z"],
        ["linear_attn.out_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    # `in_proj_b` / `in_proj_a` gate the delta-rule recurrence; quantizing them
    # would change the recurrent state, so they stay dense like Qwen3.5's.
    flat = [entry for block in layer_modules for entry in block]
    assert "linear_attn.in_proj_b" not in flat
    assert "linear_attn.in_proj_a" not in flat


def test_sensenova_loop_tree_never_declares_the_mhc_routers():
    declared = _declared_entries(SenseNovaLoopQModel)

    assert [entry for entry in declared if "mhc" in entry] == []
    # Both physical layer kinds are described, so a linear-attention layer is
    # still fully quantized even though it has no `self_attn` projections.
    assert "linear_attn:!" in declared
    assert "self_attn.q_proj" in declared


def test_sensenova_loop_declared_paths_exist_in_the_published_checkpoint():
    declared = sorted({_strip_flags(entry) for entry in _declared_entries(SenseNovaLoopQModel)})
    parents = {
        path
        for path in declared
        for other in declared
        if other != path and other.startswith(f"{path}.")
    }

    for path in declared:
        if path in parents:
            # A parent entry such as `linear_attn:!` groups its own children.
            continue
        assert f"{path}.weight" in CHECKPOINT_LAYER_SUFFIXES, path


class _FakeMHCRouter(nn.Module):
    """Mirror of the port's `MHCRouter`: raw FP32 parameters, no `nn.Linear`."""

    def __init__(self, dim=16, streams=2):
        super().__init__()
        output_size = 2 * streams + streams * streams
        self.weight = nn.Parameter(torch.zeros(output_size, streams * dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.float32))


class _FakeMHCDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(16)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(16, 16, bias=False)
        self.self_attn.k_proj = nn.Linear(16, 8, bias=False)
        self.self_attn.v_proj = nn.Linear(16, 8, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 16, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(16)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(16, 32, bias=False)
        self.mlp.up_proj = nn.Linear(16, 32, bias=False)
        self.mlp.down_proj = nn.Linear(32, 16, bias=False)
        self.mhc_attn = _FakeMHCRouter()
        self.mhc_ffn = _FakeMHCRouter()


def test_sensenova_loop_mhc_routers_are_never_quantization_candidates():
    layer = _FakeMHCDecoderLayer()

    candidates = find_modules(layer)

    assert "self_attn.q_proj" in candidates
    assert "mlp.down_proj" in candidates
    assert [name for name in candidates if "mhc" in name] == []


@pytest.mark.parametrize("attribute", ["mhc_attn", "mhc_ffn"])
def test_sensenova_should_quantize_module_keeps_mhc_routers_dense(attribute):
    module = _FakeMHCRouter()
    name = f"model.language_model.layers.5.{attribute}"

    assert (
        SenseNovaLoopQModel.should_quantize_module(None, name, module, None) is False
    )
    # A future checkpoint that stored the routers as `nn.Linear` must still not
    # be quantized: the router forward raises unless they stay FP32.
    assert (
        SenseNovaLoopQModel.should_quantize_module(None, name, nn.Linear(8, 8), None)
        is False
    )
    assert (
        SenseNovaLoopQModel.should_quantize_module(
            None, "model.language_model.layers.5.self_attn.q_proj", nn.Linear(8, 8), None
        )
        is True
    )


def test_sensenova_loop_fusion_is_withheld_until_qualified():
    assert SenseNovaLoopQModel.fusable is None
    with pytest.raises(ValueError):
        require_fusable_config(SenseNovaLoopQModel, "sensenova_loop")
