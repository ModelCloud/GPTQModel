# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForTextToWaveform
from ..base import BaseQModel
from .._const import CPU

class Qwen3OmniMoeGPTQ(BaseQModel):
    loader = AutoModelForTextToWaveform

    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "thinker.model.norm"

    support_offload_to_disk = False

    module_tree = [
        "thinker",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "gate": ("gate",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]

    def pre_quantize_generate_hook_start(self):
        self.model.thinker.model.embed_tokens = self.model.thinker.model.embed_tokens.to(self.quantize_config.device)
        self.model.thinker.visual = self.model.thinker.visual.to(self.quantize_config.device)
        self.model.thinker.audio_tower = self.model.thinker.audio_tower.to(self.quantize_config.device)

        self.model.thinker.visual.rotary_pos_emb = self.model.thinker.visual.rotary_pos_emb.to(self.quantize_config.device)
        self.model.thinker.model.rotary_emb = self.model.thinker.model.rotary_emb.to(self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.model.thinker.model.embed_tokens = self.model.thinker.model.embed_tokens.to(CPU)
        self.model.thinker.visual = self.model.thinker.visual.to(CPU)
        self.model.thinker.audio_tower = self.model.thinker.audio_tower.to(CPU)

        self.model.thinker.visual.rotary_pos_emb = self.model.thinker.visual.rotary_pos_emb.to(CPU)
        self.model.thinker.model.rotary_emb = self.model.thinker.model.rotary_emb.to(CPU)