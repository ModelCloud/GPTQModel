# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-tree coverage for the Nemotron-H definition.

Transformers 5.22 renames the native tree root ``backbone`` -> ``model`` via
the ``nemotron_h`` checkpoint conversion map, so the definition must resolve
its decoder layers from ``model.layers`` and cover the native mixer
projections.
"""

from torch import nn

from gptqmodel.models.definitions.nemotron_h import NemotronHQModel
from gptqmodel.models.module_plan import ModulePlan


def _tiny_nemotron_h_config():
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    return NemotronHConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_d_head=16,
        mamba_n_heads=4,
        layers_block_type=["mamba", "attention", "mlp", "moe"],
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_nemotron_h_tree_root_matches_native_5_22_model_root():
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    assert NemotronHQModel.extract_layers_node() == ["model.layers"]

    model = NemotronHForCausalLM(_tiny_nemotron_h_config())
    assert not hasattr(model, "backbone")
    assert len(model.model.layers) == 4


def test_nemotron_h_definition_covers_native_mixer_projections():
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    model = NemotronHForCausalLM(_tiny_nemotron_h_config())
    plan = ModulePlan.from_tree(NemotronHQModel.module_tree)
    declared = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "gate_proj", "up_proj", "down_proj"}

    covered = 0
    for layer in model.model.layers:
        for name, module in layer.mixer.named_modules():
            if not name or not isinstance(module, nn.Linear) or name.split(".")[0] not in declared:
                continue
            spec = plan.find(f"mixer.{name}")
            assert spec is not None, (type(layer.mixer).__name__, name)
            assert spec.quantize
            covered += 1

    assert covered >= 4
