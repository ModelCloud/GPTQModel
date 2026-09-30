# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.granitemoe import GraniteMoeQModel


def test_granitemoe_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="granitemoe")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition("/tmp/power-moe") is GraniteMoeQModel
    assert auto.MODEL_MAP["granitemoe"] is GraniteMoeQModel


def test_granitemoe_module_tree_targets_defused_expert_projections():
    tree = GraniteMoeQModel.module_tree[-1]
    moe_tree = tree["block_sparse_moe:moe:?"]

    assert GraniteMoeQModel.dynamic_expert_index == "num_local_experts"
    assert GraniteMoeQModel.pre_lm_head_norm_module == "model.norm"
    assert moe_tree["input_linear"]["#"] == ("linear:0",)
    assert moe_tree["output_linear"]["#"] == ("linear:1",)
