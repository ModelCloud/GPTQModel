# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class HymbaQModel(BaseQModel):
    supports_desc_act = [False]
    require_trust_remote_code = True
    require_monkeypatch = True
    write_shared_kv_cache = True
    require_pkgs = ["tiktoken>=0.7.0",
                            "sentencepiece>=0.2.0",
                            "protobuf>=5.28.3",
                            "ninja>=1.11.1.1",
                            "einops>=0.8.0",
                            "mamba_ssm>=2.2.2",
                            "causal_conv1d>=1.4.0",
                            "attn_gym>=0.0.3.dev5"]

    pre_lm_head_norm_module = "model.final_layernorm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # The conv/SSM path between `in_proj` and `out_proj` is not
            # homogeneous, so `in_proj` must never become a fold source:
            # folding a scale into it would scale that path's output. Same
            # guard as Falcon-H1's mixer (#4674). The mixer's dense leaves are
            # declared `:!` so the fold source resolution cannot skip over
            # them and land on `in_proj`.
            "mamba": ("in_proj:0:nosrc", "out_proj:1", "conv1d:!", "x_proj.0:!", "dt_proj.0:!"),
            # Hymba's MoE reads `pre_moe_layernorm`; it has no
            # `post_attention_layernorm`. Declaring a norm the layer does not
            # own left the routed-expert gate/up block without a producer of
            # its own, so it was folded into `input_layernorm` and every
            # layer's experts were scaled by the wrong norm (#4891).
            "pre_moe_layernorm": ("pre_moe_layernorm:!",),
            "moe:moe": {
                "experts": {
                    "0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                }
            }
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Score the mixer input projection instead of replaying the whole mixer.

        `HymbaBlock.forward` is not a replayable AWQ boundary. A `reuse_kv`
        layer projects no K/V of its own and reads the producing layer's
        `kv_last_layer`, a value the forward executor threads between layers and
        the per-linear activation capture never records. Replaying the block for
        the `input_layernorm -> mamba.in_proj` group therefore asserts
        `kv_last_layer is not None` (or, once a mask is supplied, fails on a
        packed causal mask the block never receives) on every real checkpoint
        (#5098).

        `mamba.in_proj` is a plain Linear and `input_layernorm`'s only consumer,
        so the projection's own reconstruction error is exactly the objective of
        that group. Anchor the search on the projection; the tree still declares
        the fold, and the scale search no longer needs the mixer's cross-layer KV
        or its packed attention mask.
        """

        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        mixer = getattr(module, "mamba", None)
        norm = getattr(module, "input_layernorm", None)
        if mixer is None or norm is None:
            return nodes
        for node in nodes:
            if node.get("prev_op") is not norm:
                continue
            layers = node.get("layers") or []
            if len(layers) != 1 or node.get("module2inspect") is not mixer:
                continue
            if node.get("_scale_consumers"):
                # A second consumer of the norm would be outside the Linear
                # boundary; keep the mixed replay so the existing validation
                # rejects or scores the whole group instead of silently
                # dropping a branch.
                continue
            target = layers[0]
            if not isinstance(target, torch.nn.Linear):
                continue
            node["module2inspect"] = target
        return nodes

    def monkey_patch(self):
        if hasattr(self.config, 'conv_dim'):
            new_conv_dim = {}
            try:
                for k, v in self.config.conv_dim.items():
                    if isinstance(k, str):
                        new_conv_dim[int(k)] = v
                self.config.conv_dim = new_conv_dim
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError("The key of HymbaConfig.conv_dim should be a string of numbers.")
