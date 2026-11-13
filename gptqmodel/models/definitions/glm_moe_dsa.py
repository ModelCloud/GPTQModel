# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import os
from pathlib import Path
from typing import Any

import torch

from ...utils.exl3_router_candidates import bind_sigmoid_grouped_router_recovery
from ...utils.model import move_to
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class GlmMoeDsaQModel(BaseQModel):
    # GLM-5 and GLM-5.1 currently share the same modeling config and both resolve
    # to transformers model_type `glm_moe_dsa`.
    # The first three decoder blocks are dense MLPs, with later blocks switching
    # to routed experts plus a shared-expert branch.
    layer_modules_strict = False

    # GLM-5.2 stores its native MTP block as model.layers.78 while the
    # transformers runtime instantiates only config.num_hidden_layers == 78
    # base blocks (indices 0--77). Preserve that checkpoint-only namespace
    # byte-for-byte until the separate MTP quantization phase replaces its
    # routed projection leaves.
    out_of_model_tensors = {"prefixes": ["model.layers.78"]}

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                # GLM-5 / GLM-5.1 use MLA attention plus a DSA indexer. `q_proj`
                # is an optional fallback path; current public configs use q_a/q_b.
                "q_proj:0",
                "q_a_proj:0",
                "kv_a_proj_with_mqa:0",
                "indexer.wk:0",
                "q_b_proj:1:in=q_a",
                "kv_b_proj:1:in=kv_a",
                "indexer.wq_b:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Dense fallback for the first `mlp_layer_types == "dense"` blocks.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def after_model_load(self, model, load_quantized_model=False):
        """Bind GLM's exact learned-router ranking for EXL3 recovery."""

        del load_quantized_model
        expected_layers = int(getattr(model.config, "num_hidden_layers", 0))
        first_routed = int(getattr(model.config, "first_k_dense_replace", 0))
        expected_routers = expected_layers - first_routed
        if expected_layers <= 0 or not 0 < expected_routers <= expected_layers:
            raise RuntimeError("GLM learned-router coverage geometry is invalid")

        patched = 0
        for name, module in model.named_modules():
            if not name.endswith(".mlp.gate"):
                continue
            correction = getattr(module, "e_score_correction_bias", None)
            if not isinstance(correction, torch.Tensor):
                continue
            if int(getattr(module, "top_k", 0)) <= 0:
                raise RuntimeError(f"GLM learned router `{name}` has no top-k")
            bind_sigmoid_grouped_router_recovery(module)
            patched += 1
        if patched != expected_routers:
            raise RuntimeError(
                "GLM learned-router recovery coverage mismatch: "
                f"patched={patched} expected={expected_routers}"
            )
        return model

    def configure_base_replay_store(
        self,
        root: str | os.PathLike[str],
        *,
        provenance: dict[str, Any],
    ) -> None:
        """Enable bounded, durable checkpoints for base-layer replay."""

        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("base replay storage requires immutable provenance")
        self._base_replay_store_root = os.fspath(Path(root).expanduser().resolve())
        self._base_replay_store_provenance = copy.deepcopy(provenance)

    def create_quantization_layer_output_writer(
        self,
        *,
        layer_index: int,
        expected_batches: int,
        progress_stage: str | None,
        apply_moe_config: bool,
    ):
        """Persist each authoritative post-quantization replay frontier."""

        if (
            layer_index < 3
            or apply_moe_config
            or progress_stage != "Forward replay"
        ):
            return None
        root = getattr(self, "_base_replay_store_root", None)
        provenance = getattr(self, "_base_replay_store_provenance", None)
        if not isinstance(root, str) or not isinstance(provenance, dict):
            return None
        from ...looper.input_cache import DiskBackedLayerOutputWriter

        return DiskBackedLayerOutputWriter(
            root,
            layer_index=layer_index,
            expected_batches=expected_batches,
            provenance={
                **copy.deepcopy(provenance),
                "layer_index": int(layer_index),
                "replay_contract": "glm-moe-dsa-base-post-quant-bf16-v1",
            },
            shard_batches=1,
        )

    def zero_route_recovery_context(
        self,
        *,
        looper,
        processor,
        layer_module,
        subset: dict[str, Any],
        task_names: tuple[str, ...],
    ):
        """Use the shared learned-router recovery with GLM-bound ranks 9--16.

        The implementation is model-tree agnostic for base routed layers. Its
        immutable family recipe supplies GLM's top-8-adjacent rank window while
        the original DeepSeek family keeps its historical ranks 7--12.
        """

        from .deepseek_v4 import DeepSeekV4MTPQuantizationModel

        return DeepSeekV4MTPQuantizationModel.zero_route_recovery_context(
            self,
            looper=looper,
            processor=processor,
            layer_module=layer_module,
            subset=subset,
            task_names=task_names,
        )

    def update_layer_replay_kwargs_from_output(self, layer, layer_output, layer_input_kwargs, target_device):
        """Propagate DSA top-k indices from full indexer layers to following shared layers."""

        if not isinstance(layer_output, tuple) or len(layer_output) < 2:
            return layer_input_kwargs

        topk_indices = layer_output[1]
        if topk_indices is None:
            return layer_input_kwargs

        layer_input_kwargs["prev_topk_indices"] = move_to(topk_indices, device=target_device)
        return layer_input_kwargs

__all__ = ["GlmMoeDsaQModel"]
