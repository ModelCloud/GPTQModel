# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class BailingMoeQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.bailing_moe.configuration_bailing.BailingConfig",
        model_class="gptqmodel.models.ports.bailing_moe.modeling_bailing.BailingForCausalLM",
    )

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        from ..ports.bailing_moe.configuration_bailing import BailingConfig

        if isinstance(self.model.config, BailingConfig):
            source = self._layer_type_replay_source or {}
            for name in ("input_ids", "attention_mask", "position_ids"):
                value = source.get(name)
                layer_input_kwargs[f"_bailing_{name}"] = None if value is None else value.to(batch_device)
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        from ..ports.bailing_moe.configuration_bailing import BailingConfig

        if not isinstance(self.model.config, BailingConfig):
            return super().prepare_layer_replay_kwargs(layer, layer_input, additional_inputs, target_device)

        import torch

        from ..ports.bailing_moe.modeling_bailing import LinearAttention, MLAttention

        with torch.no_grad():
            base = self.model.model
            hidden = layer_input[0]
            length = hidden.shape[1]
            ids = additional_inputs.get("_bailing_input_ids")
            padding_mask = additional_inputs.get("_bailing_attention_mask")
            positions = additional_inputs.get("_bailing_position_ids")
            if positions is None:
                positions = torch.arange(length, device=target_device).unsqueeze(0)
            linear = isinstance(layer.attention, LinearAttention)
            rotary = base.rotary_emb_mla if isinstance(layer.attention, MLAttention) else base.rotary_emb
            result = {"position_embeddings": rotary(hidden, positions), "attention_mask": padding_mask}
            if not linear:
                allowed = torch.ones(length, length, dtype=torch.bool, device=target_device).tril()[None, None]
                if padding_mask is not None:
                    allowed = allowed & padding_mask[:, None, None, :].bool()
                result["attention_mask"] = torch.zeros(allowed.shape, dtype=hidden.dtype, device=target_device).masked_fill(
                    ~allowed, torch.finfo(hidden.dtype).min,
                )
            if layer.is_mtp:
                if ids is None:
                    raise ValueError("Ling MTP calibration requires captured input_ids")
                offset = layer.layer_idx - self.model.config.num_hidden_layers + 1
                shifted = ids.roll(-offset, dims=-1).clone()
                shifted[:, -offset:] = 0
                embedding = self.shell_module_materialize(
                    target_submodule=base.word_embeddings, device=target_device, module_path="model.word_embeddings",
                )
                result["input_embeds"] = embedding(shifted)
                if offset == 1:
                    norm = self.shell_module_materialize(
                        target_submodule=base.norm, device=target_device, module_path="model.norm",
                    )
                    layer_input[0] = norm(hidden)
            return result

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        from ..ports.bailing_moe.configuration_bailing import BailingConfig

        # MLA, recurrent state and routing need separate full-output qualification.
        # Empty groups deliberately select AWQ's existing plain-quantization path.
        if isinstance(self.model.config, BailingConfig):
            return []
        return super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"
    layer_modules_strict = False

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "enorm": ("enorm:!",),
            "hnorm": ("hnorm:!",),
            "eh_proj": ("eh_proj:0",),
            "final_layernorm": ("final_layernorm:!",),
            # bailing_hybrid mixes delta/linear-attention, full-attention MoE and
            # MLA layers, but every variant keeps its attention projections under
            # `self_attn`. The remote-code `bailing_moe` layout (Ling 2.0) still
            # exposes `attention.query_key_value`/`attention.dense`, so both
            # branches are declared. `layer_modules_strict=False` lets each
            # checkpoint resolve only the branch it owns.
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                # `o_proj` consumes the gated attention output built from
                # `kv_b_proj` (MLA) or q/k/v (linear attention), so it must be
                # a later subset.
                "o_proj:2",
                "g_proj:0",
                "kv_a_proj_with_mqa:0",
                "kv_b_proj:1",
                "forget_gate.f_proj:0",
                "b_proj:0",
                # Linear-attention short convolution (checkpoint-owned, not a
                # quantizable projection).
                "conv1d:!",
            ),
            "attention": (
                "q_proj:0",
                # Remote-code Ling-2.6-flash (`bailing_hybrid` + `auto_map`)
                # owns the same `attention.*` namespace as the Ling 2.0
                # `bailing_moe` layout, but adds MLA projections when
                # `q_lora_rank` is set. `query_key_value`/`g_proj` belong to
                # linear-attention layers; the MLA q/kv low-rank pairs belong
                # to the full-attention layers. Both are declared here because
                # `layer_modules_strict=False` resolves only the branch a
                # layer owns. Keep `dense` after the MLA `q_b`/`kv_b` stages
                # so it is calibrated against their quantized outputs.
                "query_key_value:0",
                "q_a_proj:0",
                "kv_a_proj_with_mqa:0",
                "g_proj:0",
                "q_b_proj:1",
                "kv_b_proj:1",
                "dense:2",
                # Remote-code attention norms are checkpoint-owned but never
                # quantized. Linear-attention layers own the q/k/g norms;
                # MLA layers own the q_a/kv_a norms.
                "query_layernorm:!",
                "key_layernorm:!",
                "g_norm:!",
                "q_a_layernorm:!",
                "kv_a_layernorm:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",), # <-- 0.5MB per layer. Not worth quantizing
                # `down_proj` consumes the gate/up product, so it must be
                # calibrated (and AWQ-scaled) after gate/up are quantized.
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
            # delta/linear-attention layers have a dense MLP with no experts.
            # These entries do not resolve on MoE layers, which is fine because
            # layer_modules_strict=False.
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
