# SPDX-License-Identifier: Apache-2.0
"""Pin the carried definition's explicit routed-expert ownership contract."""
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.module_plan import ModulePlan


def test_kimi_linear_owns_declared_routed_projection():
    definition = MODEL_MAP["kimi_linear"]
    plan = ModulePlan.from_tree(definition.module_tree)
    spec = plan.find("block_sparse_moe.experts.7.w1")
    assert spec is not None and spec.quantize
    assert plan.find("block_sparse_moe.experts.7.w1.undeclared") is None
