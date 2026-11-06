# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.afmoe import AfMoeQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks


def _config():
    return SimpleNamespace(num_experts=2, num_local_experts=2, num_shared_experts=1)


def test_afmoe_model_type_selects_definition(monkeypatch):
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="afmoe"),
    )

    assert auto.check_and_get_model_definition("/tmp/afmoe") is AfMoeQModel
    assert auto.MODEL_MAP["afmoe"] is AfMoeQModel


def test_afmoe_simple_layer_modules_cover_attention_and_mlp():
    layer_modules = AfMoeQModel.simple_layer_modules(
        model_config=_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    expected = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.gate_proj",
        # Dense layer 0 MLP.
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        # Shared experts plus routed expert projections expanded from the module
        # tree (they resolve only after the loader's Defuser splits the packed
        # expert tensors into per-expert Linears).
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    }
    for expert in (0, 1):
        expected |= {
            f"mlp.experts.{expert}.gate_proj",
            f"mlp.experts.{expert}.up_proj",
            f"mlp.experts.{expert}.down_proj",
        }

    assert expected <= flat_modules
    # `shared_expert` (singular) is not the AfMoe attribute; only the plural
    # container is emitted.
    assert not any(name.startswith("mlp.shared_expert.") for name in flat_modules)
    # The router is real (mlp.router.gate) and marked `:!`, so it is present in
    # the planning tree but filtered out of the quantized module list.
    full_modules = {name for block in AfMoeQModel.full_layer_modules(model_config=_config()) for name in block}
    assert "mlp.router.gate:!" in full_modules
    assert "mlp.gate:!" not in full_modules
    assert "mlp.router.gate" not in flat_modules

    assert AfMoeQModel.layer_modules_strict is False
    hooks = AfMoeQModel.moe_lifecycle_hooks
    assert isinstance(hooks, GateUpDownMoELifecycleHooks)
    # Container names now come from the model definition's tree, not a static
    # candidate list. AFMoE declares the routed and shared expert containers.
    assert hooks.expert_block_names == ()
    assert hooks.shared_expert_block_names == ()
