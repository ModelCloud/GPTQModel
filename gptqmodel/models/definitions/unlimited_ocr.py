# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from importlib import import_module
from math import ceil
from typing import Any, Dict

import torch
from PIL import ImageOps
from transformers import AutoModel

from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, get_module, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class UnlimitedOCRQModel(BaseQModel):
    """Quantization definition for Baidu Unlimited-OCR checkpoints."""

    loader = AutoModel
    modules_with_direct_meta_tensors = ["model"]

    require_trust_remote_code = True
    support_batch_quantize = False

    require_pkgs = [
        "addict>=2.4.0",
        "easydict>=1.13",
        "einops>=0.8.0",
        "matplotlib>=3.10.0",
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    layer_modules_strict = False
    dynamic_expert_index = "n_routed_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.norm"
    _direct_parameter_names = ("image_newline", "view_seperator")

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = []
        for module_name in (
            "model.sam_model",
            "model.vision_model",
            "model.projector",
            "model.embed_tokens",
            "model.norm",
        ):
            if get_module(model, module_name) is not None:
                base_modules.append(module_name)
        return base_modules

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        self.shell_module_materialize(
            core_model.embed_tokens, self.quantize_config.device
        )
        self.shell_module_materialize(core_model.norm, self.quantize_config.device)
        self.shell_module_materialize(core_model.sam_model, self.quantize_config.device)
        self.shell_module_materialize(
            core_model.vision_model, self.quantize_config.device
        )
        self.shell_module_materialize(core_model.projector, self.quantize_config.device)
        self._restore_vision_position_ids(core_model)
        self._move_direct_parameters(torch.device(self.quantize_config.device))

    def _move_direct_parameters(self, device: torch.device) -> None:
        core_model = self.model.model
        for name in self._direct_parameter_names:
            parameter = getattr(core_model, name, None)
            if (
                not isinstance(parameter, torch.nn.Parameter)
                or parameter.device == device
            ):
                continue
            setattr(
                core_model,
                name,
                torch.nn.Parameter(
                    parameter.to(device=device),
                    requires_grad=parameter.requires_grad,
                ),
            )

    @staticmethod
    def _restore_vision_position_ids(core_model) -> None:
        embeddings = core_model.vision_model.embeddings
        position_embedding = embeddings.position_embedding
        position_ids = getattr(embeddings, "position_ids", None)
        if torch.is_tensor(position_ids) and position_ids.device.type != "meta":
            if position_ids.device != position_embedding.weight.device:
                embeddings.position_ids = position_ids.to(
                    device=position_embedding.weight.device
                )
            return

        position_ids = torch.arange(
            position_embedding.num_embeddings,
            device=position_embedding.weight.device,
            dtype=torch.long,
        ).expand((1, -1))
        if "position_ids" in embeddings._buffers:
            embeddings._buffers["position_ids"] = position_ids
            embeddings._non_persistent_buffers_set.add("position_ids")
        else:
            embeddings.register_buffer("position_ids", position_ids, persistent=False)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        self._move_direct_parameters(CPU)
        if self.quantize_config.offload_to_disk:
            for module in (
                core_model.embed_tokens,
                core_model.norm,
                core_model.sam_model,
                core_model.vision_model,
                core_model.projector,
            ):
                offload_to_disk(
                    model=core_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        core_model.embed_tokens = move_to(core_model.embed_tokens, device=CPU)
        core_model.norm = move_to(core_model.norm, device=CPU)
        core_model.sam_model = move_to(core_model.sam_model, device=CPU)
        core_model.vision_model = move_to(core_model.vision_model, device=CPU)
        core_model.projector = move_to(core_model.projector, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def _prepare_image_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the checkpoint's own image transform and image-token layout."""

        remote_module = import_module(self.model.__class__.__module__)
        image = fetch_image({"image": sample["image"]}).convert("RGB")
        prompt = sample.get("text") or "<image>\nFree OCR."
        if prompt.count("<image>") != 1:
            raise ValueError(
                "Unlimited-OCR calibration samples must contain exactly one <image> token."
            )

        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        if not isinstance(image_token_id, int) or image_token_id < 0:
            raise ValueError(
                "Unlimited-OCR tokenizer does not define the <image> token."
            )

        base_size = 1024
        image_size = 640
        patch_size = 16
        downsample_ratio = 4
        image_transform = remote_module.BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True,
        )

        width_crop_num = height_crop_num = 1
        images_crop_raw = []
        if image.width > image_size or image.height > image_size:
            images_crop_raw, (width_crop_num, height_crop_num) = (
                remote_module.dynamic_preprocess(
                    image,
                    image_size=image_size,
                )
            )

        global_view = ImageOps.pad(image, (base_size, base_size), color=(127, 127, 127))
        images_ori = image_transform(global_view).to(torch.bfloat16).unsqueeze(0)

        if width_crop_num > 1 or height_crop_num > 1:
            images_crop = torch.stack(
                [image_transform(crop).to(torch.bfloat16) for crop in images_crop_raw]
            )
        else:
            images_crop = torch.zeros(
                (1, 3, base_size, base_size), dtype=torch.bfloat16
            )

        text_before, text_after = prompt.split("<image>")
        tokenized_before = tokenizer.encode(text_before, add_special_tokens=False)
        tokenized_after = tokenizer.encode(text_after, add_special_tokens=False)

        num_queries = ceil((image_size // patch_size) / downsample_ratio)
        num_queries_base = ceil((base_size // patch_size) / downsample_ratio)
        image_tokens = (
            [image_token_id] * num_queries_base + [image_token_id]
        ) * num_queries_base
        image_tokens.append(image_token_id)
        if width_crop_num > 1 or height_crop_num > 1:
            image_tokens.extend(
                ([image_token_id] * (num_queries * width_crop_num) + [image_token_id])
                * (num_queries * height_crop_num)
            )

        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = 0
        input_ids = torch.tensor(
            [bos_token_id, *tokenized_before, *image_tokens, *tokenized_after],
            dtype=torch.long,
        ).unsqueeze(0)
        images_seq_mask = torch.tensor(
            [False] * (1 + len(tokenized_before))
            + [True] * len(image_tokens)
            + [False] * len(tokenized_after),
            dtype=torch.bool,
        ).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "images": [(images_crop, images_ori)],
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": torch.tensor(
                [[width_crop_num, height_crop_num]],
                dtype=torch.long,
            ),
        }

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del batch_size, kwargs
        calibration_data = []
        for batch in batched(
            calibration_dataset, 1, process_func=self.preprocess_dataset
        ):
            calibration_data.append(self._prepare_image_sample(batch[0]))
        return calibration_data

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        images = example.get("images")
        if images is None:
            return example

        vision_model = self.model.model.sam_model
        first_parameter = next(vision_model.parameters(), None)
        vision_device = getattr(first_parameter, "device", data_device)
        vision_dtype = getattr(vision_model, "dtype", None)
        if not isinstance(vision_dtype, torch.dtype):
            vision_dtype = getattr(first_parameter, "dtype", None)
        if isinstance(vision_dtype, torch.dtype):
            example["images"] = nested_move_to(
                images,
                device=vision_device,
                dtype=vision_dtype,
            )
        return example


__all__ = ["UnlimitedOCRQModel"]
