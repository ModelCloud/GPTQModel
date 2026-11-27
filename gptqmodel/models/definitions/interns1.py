# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

from transformers import ProcessorMixin

from ...utils.image import load_image_from_url
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from ._vlm_calibration import prepare_vlm_calibration_dataset


class InternS1QModel(BaseQModel):
    require_pkgs = ["sentencepiece>=0.2.0"]

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.interns1.configuration_interns1.InternS1Config",
        model_class="gptqmodel.models.ports.interns1.modeling_interns1.InternS1ForConditionalGeneration",
        processor_class="gptqmodel.models.ports.interns1.processing_interns1.InternS1Processor",
        tokenizer_class="gptqmodel.models.ports.interns1.tokenization_interns1.InternS1Tokenizer",
    )

    dynamic_expert_index = "num_experts"
    # Mixed dense/routed layers resolve only their declared MLP branch.
    layer_modules_strict = False
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()
    # The routed `mlp:moe:?` root exists only in the effective tree; AWQ must
    # resolve it there to capture the shared router/expert input.
    _MODULE_TREE_RUNTIME_METHODS = BaseQModel._MODULE_TREE_RUNTIME_METHODS | {"awq_input_feature_aggregation"}

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "q_proj:0",
                "k_norm:!",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    @classmethod
    def get_num_experts(cls, model_config):
        return getattr(model_config.get_text_config(), "num_experts", 0)

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector)
        config = getattr(model, "config", None)
        text = getattr(config, "text_config", None)
        blocks = tree[-1]
        if getattr(text, "model_type", None) == "qwen3_moe" and "mlp:moe:?" not in blocks:
            sparse_layers = [
                index for index in range(text.num_hidden_layers)
                if index not in text.mlp_only_layers and (index + 1) % text.decoder_sparse_step == 0
            ]
            if not sparse_layers:
                return tree
            moe = {
                "gate": ("gate:!",),
                "experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")},
            }
            if len(sparse_layers) < text.num_hidden_layers:
                for projection, order in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                    moe[projection] = (f"{projection}:{order}",)
            blocks["mlp:moe:?"] = moe
            blocks.pop("mlp", None)
        return tree

    def pre_quantize_generate_hook_start(self):
        language_model = self.model.model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.multi_modal_projector, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        language_model = self.model.model.language_model
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=language_model,
                module=language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model.model,
                module=self.model.model.vision_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model.model,
                module=self.model.model.multi_modal_projector,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)
        self.model.model.vision_tower = move_to(self.model.model.vision_tower, device=CPU)
        self.model.model.multi_modal_projector = move_to(self.model.model.multi_modal_projector, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]

        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

    @staticmethod
    def replace_image_with_pil(sample):
        """
        image url -> PIL.Image
        """

        for msg in sample:
            if "content" not in msg and not isinstance(msg["content"], dict):
                continue

            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    item["image"] = load_image_from_url(item["image"])

        return sample

    def _prepare_calibration_batch(self, processor, batch):
        batched_samples = [self.replace_image_with_pil(sample) for sample in batch]
        return self.prepare_inputs_for_conversations(processor, batched_samples)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        return prepare_vlm_calibration_dataset(
            self,
            calibration_dataset,
            batch_size=batch_size,
            options=kwargs,
            prepare_batch=self._prepare_calibration_batch,
            preprocess_func=self.preprocess_dataset,
        )
