# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict

from transformers import ProcessorMixin

from ...utils.looper_helpers import normalize_device_like
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from ._vlm_calibration import prepare_vlm_calibration_dataset


class HunYuanVLPilProcessor:
    """Vendored Hunyuan-VL processor with the PIL image backend.

    The definition has always calibrated with ``backend="pil"`` image
    preprocessing; keep that while resolving every component to vendored code.
    """

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        from ..ports.native.auto import NativeProcessor
        from ..ports.native.hunyuan_vl.image_processing_pil_hunyuan_vl import HunYuanVLImageProcessorPil

        processor = NativeProcessor.from_pretrained(model_id_or_path, **kwargs)
        kwargs.pop("config", None)
        kwargs.pop("tokenizer", None)
        processor.image_processor = HunYuanVLImageProcessorPil.from_pretrained(model_id_or_path, **kwargs)
        return processor


class HunYuanVLQModel(BaseQModel):
    """Quantization definition for native Transformers Hunyuan-VL checkpoints."""
    fusable = FusableConfig("hunyuan_vl").merge(LlamaFusableConfig())


    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForImageTextToText",
        processor_class="gptqmodel.models.definitions.hunyuan_vl.HunYuanVLPilProcessor",
        tokenizer_class="gptqmodel.models.ports.native.auto.NativeTokenizer",
    )

    require_load_processor = True

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

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
                "query_layernorm:!",
                "q_proj:0",
                "key_layernorm:!",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    def _materialize_module(self, parent, name: str, module_path: str):
        module = getattr(parent, name)
        target_device = normalize_device_like(self.quantize_config.device) or CPU
        setattr(
            parent,
            name,
            self.shell_module_materialize(
                module,
                target_device,
                module_path=module_path,
            ),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self._materialize_module(
            language_model, "embed_tokens", "model.language_model.embed_tokens"
        )
        self._materialize_module(language_model, "norm", "model.language_model.norm")
        self._materialize_module(
            language_model, "rotary_emb", "model.language_model.rotary_emb"
        )
        self._materialize_module(core_model, "vision_tower", "model.vision_tower")

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, "embed_tokens"),
            (language_model, "norm"),
            (language_model, "rotary_emb"),
            (core_model, "vision_tower"),
        )

        if self.quantize_config.offload_to_disk:
            for parent, name in modules:
                offload_to_disk(
                    model=parent,
                    module=getattr(parent, name),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        for parent, name in modules:
            setattr(parent, name, move_to(getattr(parent, name), device=CPU))

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        # BaseQModel already loaded the vendored (PIL-backend) processor with the
        # normalized tokenizer through this definition's loader.
        return self.processor

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )

    def prepare_dataset(
        self,
        calibration_dataset,
        batch_size: int = 1,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        return prepare_vlm_calibration_dataset(
            self,
            calibration_dataset,
            batch_size=batch_size,
            options=kwargs,
            preprocess_func=self.preprocess_dataset,
        )


__all__ = ["HunYuanVLPilProcessor", "HunYuanVLQModel"]
