# SPDX-License-Identifier: Apache-2.0
"""Pin the carried definition's explicit routed-expert ownership contract."""
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.module_plan import ModulePlan


def test_dots3_note_owns_declared_routed_projection():
    definition = MODEL_MAP["dots3_note"]
    plan = ModulePlan.from_tree(definition.module_tree)
    spec = plan.find("mlp.experts.7.gate_proj")
    assert spec is not None and spec.quantize
    assert plan.find("mlp.experts.7.gate_proj.undeclared") is None
