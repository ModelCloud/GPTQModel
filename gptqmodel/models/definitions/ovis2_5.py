# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from typing import Dict

import torch

from ...utils.calibration import batched_conversations
from ...utils.image import load_image_from_url
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from ...utils.torch import as_torch_device
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ._vlm_calibration import prepare_vlm_calibration_dataset


class Ovis2_5QModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.ovis2_5.configuration_ovis2_5.Ovis2_5_Config",
        model_class="gptqmodel.models.ports.ovis2_5.modeling_ovis2_5.Ovis2_5",
        processor_class="gptqmodel.models.ports.ovis2_5.processing_ovis2_5.Ovis2_5Processor",
    )
    require_trust_remote_code = False
    require_monkeypatch = True
    lm_head = "llm.lm_head"
    fusable = None

    pre_lm_head_norm_module = "llm.model.norm"

    module_tree = [
        "llm",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "q_norm:!", "k_proj:0", "k_norm:!", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    modality = [MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    def monkey_patch(self):
        self.model.processor = self.processor
        self.model.visual_tokenizer.image_processor = self.processor.image_processor

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Decoder projections quantize, but native Q/K normalization and multimodal
        # replay/fold/search have not been qualified as a complete scale objective.
        return []

    def _sanitize_input_ids_for_embeddings(self, input_ids):
        # These negative values are source-defined multimodal embeddings, not
        # invalid text IDs. The model validates both the values and patch counts.
        return input_ids

    def pre_quantize_generate_hook_start(self):
        self.shell_module_materialize(self.model.llm.model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(self.model.llm.model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.visual_tokenizer, self.quantize_config.device)
        self.shell_module_materialize(self.model.vte, self.quantize_config.device)

        # VisionRotaryEmbedding cannot be correctly reconstructed via `_build_nonpersistent_buffer_template()`.
        # Therefore, VisionRotaryEmbedding is manually reconstructed here.
        rotary_pos_emb_cls = type(self.model.visual_tokenizer.vit.vision_model.encoder.rotary_pos_emb)
        config = self.model.config.vit_config
        assert "VisionRotaryEmbedding" in rotary_pos_emb_cls.__name__
        rotary_pos_emb = rotary_pos_emb_cls(config.hidden_size // config.num_attention_heads // 2).to(as_torch_device(self.quantize_config.device))
        self.model.visual_tokenizer.vit.vision_model.encoder.rotary_pos_emb = rotary_pos_emb

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=self.model.llm,
                            module=self.model.llm.model.embed_tokens,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=self.model.llm,
                            module=self.model.llm.model.rotary_emb,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=self.model,
                            module=self.model.visual_tokenizer,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            offload_to_disk(model=self.model,
                            module=self.model.vte,
                            disk_path=self.quantize_config.offload_to_disk_path,
                            )
            return

        self.model.llm.model.embed_tokens = move_to(self.model.llm.model.embed_tokens, device=CPU)
        self.model.llm.model.rotary_emb = move_to(self.model.llm.model.rotary_emb, device=CPU)
        self.model.visual_tokenizer = move_to(self.model.visual_tokenizer, device=CPU)
        self.model.vte = move_to(self.model.vte, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    @staticmethod
    def replace_image_with_pil(sample):
        """
        image url -> PIL.Image
        """

        for msg in sample:
            if not isinstance(msg.get("content"), list):
                continue

            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image" and isinstance(item.get("image"), str):
                    item["image"] = load_image_from_url(item["image"])

        return sample

    def _prepare_calibration_batches(self, conversations, batch_size):
        calib_data = []
        for batch in batched_conversations(conversations, batch_size, process_func=self.preprocess_dataset):
            for sample in batch:
                sample = self.replace_image_with_pil(sample)
                input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                    messages=sample,
                    add_generation_prompt=True,
                )
                attention_mask = torch.ne(input_ids, self.model.text_tokenizer.pad_token_id)

                if pixel_values is not None:
                    pixel_values = pixel_values.to(dtype=self.model.visual_tokenizer.vit.dtype)

                calib_data.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "grid_thws": grid_thws,
                    }
                )
        return calib_data

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        return prepare_vlm_calibration_dataset(
            self,
            calibration_dataset,
            batch_size=batch_size,
            options=kwargs,
            prepare_conversations=lambda rows: self._prepare_calibration_batches(rows, batch_size),
        )
