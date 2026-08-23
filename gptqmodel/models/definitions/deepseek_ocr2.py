# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from types import SimpleNamespace
from typing import Any, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.model import MODALITY, get_module, move_to
from ...utils.offload import offload_to_disk
from ...utils.structure import LazyTurtle
from .._const import CPU
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


def _checkpoint_renaming(source_pattern: str, target_pattern: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_patterns=[source_pattern],
        target_patterns=[target_pattern],
        operations=[],
    )


# Transformers 5.10 added these DeepSeek-OCR2 checkpoint conversions, while
# GPT-QModel's supported Transformers floor is still 5.4. Keep an adapter-owned
# copy of the upstream one-to-one rules so LazyTurtle can materialize an older
# checkpoint even when that registry entry is absent. Tensor converters remain
# owned by Transformers and are preserved whenever its registry is available.
_HF_CHECKPOINT_CONVERSION_MAP = (
    _checkpoint_renaming(
        r"sam_model\.blocks\.(\d+)\.norm1\.",
        r"vision_tower.sam_encoder.layers.\1.layer_norm1.",
    ),
    _checkpoint_renaming(
        r"sam_model\.blocks\.(\d+)\.norm2\.",
        r"vision_tower.sam_encoder.layers.\1.layer_norm2.",
    ),
    _checkpoint_renaming(
        r"sam_model\.blocks\.(\d+)\.attn\.",
        r"vision_tower.sam_encoder.layers.\1.attn.",
    ),
    _checkpoint_renaming(
        r"sam_model\.blocks\.(\d+)\.mlp\.",
        r"vision_tower.sam_encoder.layers.\1.mlp.",
    ),
    _checkpoint_renaming(
        r"sam_model\.patch_embed\.proj\.",
        "vision_tower.sam_encoder.patch_embed.projection.",
    ),
    _checkpoint_renaming(r"sam_model\.pos_embed", "vision_tower.sam_encoder.pos_embed"),
    _checkpoint_renaming(r"sam_model\.neck\.0\.", "vision_tower.sam_encoder.neck.conv1."),
    _checkpoint_renaming(r"sam_model\.neck\.1\.", "vision_tower.sam_encoder.neck.layer_norm1."),
    _checkpoint_renaming(r"sam_model\.neck\.2\.", "vision_tower.sam_encoder.neck.conv2."),
    _checkpoint_renaming(r"sam_model\.neck\.3\.", "vision_tower.sam_encoder.neck.layer_norm2."),
    _checkpoint_renaming(r"sam_model\.net_2\.", "vision_tower.sam_encoder.proj.conv1."),
    _checkpoint_renaming(r"sam_model\.net_3\.", "vision_tower.sam_encoder.proj.conv2."),
    _checkpoint_renaming(
        r"qwen2_model\.model\.model\.layers\.",
        "vision_tower.vision_encoder.layers.",
    ),
    _checkpoint_renaming(
        r"qwen2_model\.model\.model\.norm\.",
        "vision_tower.vision_encoder.norm.",
    ),
    _checkpoint_renaming(r"qwen2_model\.query_768\.", "vision_tower.query_768_resolution."),
    _checkpoint_renaming(r"qwen2_model\.query_1024\.", "vision_tower.query_1024_resolution."),
    _checkpoint_renaming(r"projector\.layers\.", "multi_modal_projector."),
    _checkpoint_renaming(r"view_seperator", "view_separator"),
    _checkpoint_renaming(r"(^|model\.)embed_tokens\.", r"\1language_model.embed_tokens."),
    _checkpoint_renaming(r"(^|model\.)layers\.", r"\1language_model.layers."),
    _checkpoint_renaming(r"(^|model\.)norm\.", r"\1language_model.norm."),
)


class DeepSeekOCR2QModel(BaseQModel):
    loader = AutoModelForImageTextToText
    modules_with_direct_meta_tensors = ["model"]

    HF_CONVERSION_MAP_REVERSED = tuple(
        LazyTurtle.reverse_hf_conversion_map(_HF_CHECKPOINT_CONVERSION_MAP) or ()
    )

    require_load_processor = True
    support_batch_quantize = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    layer_modules_strict = False
    dynamic_expert_index = "n_routed_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "gate": ("gate:!",),
                "shared_experts:shared": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "experts:routed:expert_activation=experts.act_fn": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
            },
        },
    ]

    @classmethod
    def resolve_hf_conversion_map_reversed(cls, target_model=None):
        """Merge adapter fallbacks behind any Transformers-owned conversions."""

        upstream_map = LazyTurtle.infer_hf_conversion_map_reversed(target_model=target_model)
        fallback_map = copy.deepcopy(cls.HF_CONVERSION_MAP_REVERSED)
        if upstream_map is None:
            return fallback_map

        merged_map = copy.deepcopy(list(upstream_map))
        upstream_sources = {
            source
            for entry in merged_map
            for source in LazyTurtle._coerce_patterns(
                getattr(entry, "_original_source_patterns", getattr(entry, "source_patterns", None))
            )
        }
        for entry in fallback_map:
            entry_sources = LazyTurtle._coerce_patterns(
                getattr(entry, "_original_source_patterns", getattr(entry, "source_patterns", None))
            )
            if any(source in upstream_sources for source in entry_sources):
                continue
            merged_map.append(entry)
            upstream_sources.update(entry_sources)
        return merged_map

    @classmethod
    def get_base_modules(cls, model):
        base_modules = []
        for module_name in (
            "model.vision_tower",
            "model.multi_modal_projector",
            "model.language_model.embed_tokens",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
        ):
            if get_module(model, module_name) is not None:
                base_modules.append(module_name)
        return base_modules

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.norm, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(core_model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(core_model.multi_modal_projector, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=language_model,
                module=language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.norm,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.vision_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.multi_modal_projector,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.norm = move_to(language_model.norm, device=CPU)
        language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)
        core_model.vision_tower = move_to(core_model.vision_tower, device=CPU)
        core_model.multi_modal_projector = move_to(core_model.multi_modal_projector, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            inputs = processor(
                images=[sample["image"] for sample in batch],
                text=[sample["text"] for sample in batch],
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        del processor
        return calib_data

    def move_input_capture_example(self, example: Dict[str, Any], data_device: torch.device) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        if torch.is_tensor(pixel_values):
            vision_tower = self.model.model.vision_tower
            first_parameter = next(vision_tower.parameters(), None)
            vision_device = getattr(first_parameter, "device", pixel_values.device)
            vision_dtype = getattr(vision_tower, "dtype", None)
            if not isinstance(vision_dtype, torch.dtype):
                vision_dtype = getattr(first_parameter, "dtype", None)
            if isinstance(vision_dtype, torch.dtype):
                example["pixel_values"] = pixel_values.to(device=vision_device, dtype=vision_dtype)
        return example


__all__ = ["DeepSeekOCR2QModel"]
