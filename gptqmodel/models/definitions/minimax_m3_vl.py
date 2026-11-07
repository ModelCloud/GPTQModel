# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict, Optional

from PIL import Image
from transformers import AutoModelForImageTextToText, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class MiniMaxM3VLGPTQ(BaseQModel):
    fusable = FusableConfig("minimax_m3_vl").merge(LlamaFusableConfig(), profiles=("qkv",))

    loader = AutoModelForImageTextToText

    # The language tower consumes image embeddings at inference time, so the
    # calibration stream must include image-conditioned activations as well.
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = False
    support_batch_quantize = False

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    # MiniMax-M3 starts with dense MLP layers, then switches to sparse MoE.
    layer_modules_strict = False
    dynamic_expert_index = "num_local_experts"

    # Defuser splits MiniMax-M3 packed expert tensors into gate/up/down modules.
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "q_norm:0:!",
                "k_proj:0",
                "k_norm:0:!",
                "v_proj:0",
                "indexer.q_proj:0",
                "indexer.q_norm:0:!",
                "indexer.k_proj:0",
                "indexer.k_norm:0:!",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Dense fallback used by early decoder blocks.
                "": ("gate_up_proj:0", "down_proj:1"),
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "shared_experts": ("gate_up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    @staticmethod
    def process_vision_info(
        conversations,
    ) -> Optional[list[Image.Image]]:
        image_inputs = []
        for vision_info in extract_vision_info(conversations):
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError(
                    "MiniMax-M3 VL calibration supports image content only."
                )
        return image_inputs or None

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        # The checkpoint advertises a remote `processing_minimax.MiniMaxVLProcessor`
        # that requires trust_remote_code; enabling that globally also swaps in a
        # remote config class that cannot rebuild the model. Build the equivalent
        # native processor from the local image/tokenizer/video components instead.
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            AutoVideoProcessor,
        )
        from transformers.models.minimax_m3_vl.processing_minimax_m3_vl import (
            MiniMaxM3VLProcessor,
        )

        return MiniMaxM3VLProcessor(
            image_processor=AutoImageProcessor.from_pretrained(
                self.model_local_path
            ),
            tokenizer=AutoTokenizer.from_pretrained(self.model_local_path),
            video_processor=AutoVideoProcessor.from_pretrained(
                self.model_local_path
            ),
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.__dict__.get("processor") or self.load_processor()
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "MiniMax-M3 VL processor does not expose a chat-template tokenizer."
            )

        calib_data = []
        for batch_index, batch in enumerate(
            batched_conversations(
                calibration_dataset,
                batch_size,
                process_func=self.preprocess_dataset,
            )
        ):
            try:
                text = tokenizer.apply_chat_template(
                    batch, tokenize=False, add_generation_prompt=True
                )
                image_inputs = self.process_vision_info(batch)
                inputs = processor(
                    text=text,
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except Exception as exc:
                raise ValueError(
                    f"MiniMax-M3 VL calibration preparation failed for batch {batch_index}: {exc}"
                ) from exc

            if "input_ids" not in inputs:
                raise ValueError(
                    f"MiniMax-M3 VL processor output for batch {batch_index} is missing input_ids."
                )
            if image_inputs is not None:
                missing = [
                    name
                    for name in ("pixel_values", "image_grid_thw")
                    if inputs.get(name) is None
                ]
                if missing:
                    raise ValueError(
                        f"MiniMax-M3 VL processor output for batch {batch_index} "
                        f"is missing required field(s): {', '.join(missing)}."
                    )
            calib_data.append(inputs)

        del processor
        return calib_data


__all__ = ["MiniMaxM3VLGPTQ"]
