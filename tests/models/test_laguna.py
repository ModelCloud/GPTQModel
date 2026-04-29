# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

from model_test import ModelTest

from gptqmodel.quantization.config import ExpertsRoutingOverride, Fallback, MoEConfig, VramStrategy
from gptqmodel.utils.backend import BACKEND


class TestLagunaXS2(ModelTest):
    NATIVE_MODEL_ID = os.environ.get("GPTQMODEL_LAGUNA_XS2_MODEL", "/monster/data/model/Laguna-XS.2")

    FALLBACK = Fallback()
    LOAD_BACKEND = BACKEND.AUTO
    DENSE_VRAM_STRATEGY = VramStrategy.EXCLUSIVE
    MOE_VRAM_STRATEGY = VramStrategy.BALANCED
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all"))

    # Layer 0 is dense; layers 1 and 2 are sparse MoE blocks. Using the first
    # three layers keeps the true-model path fast while covering two split MoE layers.
    MODEL_COMPAT_FAST_LAYER_COUNT = 3
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    DATASET_SIZE_FAST = 128
    DATASET_CONCAT_SIZE_FAST = 1024
    EVAL_BATCH_SIZE = "auto"

    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": False,
            "evalution_suite_kwargs": {
                "stream": True,
            },
            "acc": {
                "value": 0.45,
                "floor_pct": 0.50,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.45,
                "floor_pct": 0.50,
                "ceil_pct": 1.0,
            },
        },
    }
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    def _build_fast_model_compat_dynamic(self, model):
        dynamic = super()._build_fast_model_compat_dynamic(model)
        self._assert_fast_quant_covers_defused_moe_layers(model)
        return dynamic

    def _assert_fast_quant_covers_defused_moe_layers(self, model):
        layers_node, layers = self._resolve_layers_for_fast_model_compat(model)
        if layers_node is None or layers is None:
            raise AssertionError("Laguna fast quant test could not resolve decoder layers.")

        layer_count = len(layers)
        layer_limit = self._fast_model_layer_limit(layers)
        layer_position = self._resolve_fast_model_layer_position()

        if layer_count <= layer_limit:
            selected_indexes = range(layer_count)
        elif layer_position == "last":
            selected_indexes = range(layer_count - layer_limit, layer_count)
        else:
            selected_indexes = range(layer_limit)

        covered = []
        for layer_idx in selected_indexes:
            layer = layers[layer_idx]
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if experts is None:
                continue

            expert0 = getattr(experts, "0", None)
            has_split_expert = (
                expert0 is not None
                and hasattr(expert0, "gate_proj")
                and hasattr(expert0, "up_proj")
                and hasattr(expert0, "down_proj")
            )
            if has_split_expert:
                covered.append(layer_idx)

        assert len(covered) >= 2, (
            "Laguna fast quant test must cover at least two split MoE expert layers; "
            f"covered={covered}, selected={list(selected_indexes)}"
        )

    def test_laguna_xs2(self):
        self.quantize_and_evaluate()
