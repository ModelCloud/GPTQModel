# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

import torch
from transformers import AutoModelForTextToWaveform, AutoProcessor

from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class Qwen3OmniMoeGPTQ(BaseQModel):
    ATTENTION_MASKS_REQUIRED_FOR_INPUT = True
    ATTENTION_MASKS_DTYPE = torch.long

    INPUT_EMBEDDING_EXTRA_ARGS = {
        "return_audio": False,
    }

    loader = AutoModelForTextToWaveform

    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "thinker.model.norm"

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
        spk_path = os.path.join(self.model_local_path, "spk_dict.pt")
        if os.path.isfile(spk_path):
            self.model.load_speakers(spk_path)

        self.shell_module_materialize(self.model.thinker.model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.visual, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.audio_tower, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.visual.rotary_pos_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.thinker.model.rotary_emb, self.quantize_config.device)
        if hasattr(self.model, "talker"):
            self.shell_module_materialize(self.model.talker, self.quantize_config.device)
        if hasattr(self.model, "code2wav"):
            self.shell_module_materialize(self.model.code2wav, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=self.model.thinker.model,
                            module=self.model.thinker.model.embed_tokens,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker,
                            module=self.model.thinker.visual,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker,
                            module=self.model.thinker.audio_tower,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker.visual,
                            module=self.model.thinker.visual.rotary_pos_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            offload_to_disk(model=self.model.thinker.model,
                            module=self.model.thinker.model.rotary_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )

            if hasattr(self.model, "talker"):
                offload_to_disk(model=self.model,
                                module=self.model.talker,
                                disk_path=self.quantize_config.offload_to_disk_path,
                                )
            if hasattr(self.model, "code2wav"):
                offload_to_disk(model=self.model,
                                module=self.model.code2wav,
                                disk_path=self.quantize_config.offload_to_disk_path,
                                )
            return

        self.model.thinker.model.embed_tokens = self.model.thinker.model.embed_tokens.to(CPU)
        self.model.thinker.visual = self.model.thinker.visual.to(CPU)
        self.model.thinker.audio_tower = self.model.thinker.audio_tower.to(CPU)
        if hasattr(self.model, "talker"):
            self.model.talker = self.model.talker.to(CPU)
        if hasattr(self.model, "code2wav"):
            self.model.code2wav = self.model.code2wav.to(CPU)

        self.model.thinker.visual.rotary_pos_emb = self.model.thinker.visual.rotary_pos_emb.to(CPU)
        self.model.thinker.model.rotary_emb = self.model.thinker.model.rotary_emb.to(CPU)

    def after_model_load(self, model, load_quantized_model=False):
        # need to load processor for save processor_config and chat_template
        if not load_quantized_model:
            self.processor = AutoProcessor.from_pretrained(self.model_local_path)

        return model
