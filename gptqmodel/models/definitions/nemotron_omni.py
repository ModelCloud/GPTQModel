# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import Dict, Optional

from PIL import Image
from transformers import AutoProcessor, ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from .nemotron_h import NemotronHQModel


class NemotronOmniQModel(NemotronHQModel):
    """GPTQ wrapper for Nemotron Omni models that embed a Nemotron-H language backbone."""

    # The language backbone is calibrated on activations that already contain
    # the vision tower's embeddings, so text-only calibration is not
    # representative of inference.
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    # The dynamic-resolution image processor returns `pixel_values` as a list
    # when image sizes differ, but the remote `generate()` only accepts a
    # stacked tensor. Keep calibration single-sample so batching cannot mix
    # differently sized images.
    support_batch_quantize = False

    pre_lm_head_norm_module = "language_model.backbone.norm_f"

    module_tree = [
        "language_model",
        "backbone",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            "mixer": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "in_proj:2",
                "out_proj:2",
                "gate_proj:3",
                "up_proj:3",
                "down_proj:4",
            ),
        }
    ]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        for name, _ in model.named_children():
            if name != "language_model":
                base_modules.append(name)
        return base_modules

    @staticmethod
    def _has_multimodal_inputs(kwargs) -> bool:
        return any(
            kwargs.get(name) is not None
            for name in ("pixel_values", "pixel_values_videos", "sound_clips", "sound_length")
        )

    def _materialize_top_level_module(self, attr_name: str):
        module = getattr(self.model, attr_name, None)
        if module is None:
            return
        if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
            setattr(self.model, attr_name, move_to(module, device=self.quantize_config.device))
            return
        setattr(
            self.model,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    def pre_quantize_generate_hook_start(self):
        for attr_name in ("vision_model", "mlp1", "sound_encoder", "sound_projection"):
            if hasattr(self.model, attr_name):
                self._materialize_top_level_module(attr_name)

        self.shell_module_materialize(self.model.language_model.backbone.norm_f, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            for attr_name in ("vision_model", "mlp1", "sound_encoder", "sound_projection"):
                module = getattr(self.model, attr_name, None)
                if module is not None:
                    offload_to_disk(
                        model=self.model,
                        module=module,
                        disk_path=self.quantize_config.offload_to_disk_path,
                    )

            offload_to_disk(
                model=self.model.language_model.backbone,
                module=self.model.language_model.backbone.norm_f,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        for attr_name in ("vision_model", "mlp1", "sound_encoder", "sound_projection"):
            module = getattr(self.model, attr_name, None)
            if module is not None:
                setattr(self.model, attr_name, move_to(module, device=CPU))

        self.model.language_model.backbone.norm_f = move_to(self.model.language_model.backbone.norm_f, device=CPU)

    def forward(self, *args, **kwargs):
        # Text-only quantization/eval paths should bypass the multimodal wrapper,
        # whose `forward()` requires image tensors even when no visual inputs exist.
        if self._has_multimodal_inputs(kwargs):
            return self.model(*args, **kwargs)
        return self.model.language_model(*args, **kwargs)

    @staticmethod
    def process_vision_info(conversations) -> Optional[list[Image.Image]]:
        image_inputs = []
        for vision_info in extract_vision_info(conversations):
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError(
                    "Nemotron Omni calibration supports image content only."
                )
        return image_inputs or None

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(
            self.model_local_path, trust_remote_code=True
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.__dict__.get("processor") or self.load_processor()
        calib_data = []
        for batch_index, batch in enumerate(
            batched_conversations(
                calibration_dataset,
                batch_size,
                process_func=self.preprocess_dataset,
            )
        ):
            try:
                text = processor.apply_chat_template(
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
                    f"Nemotron Omni calibration preparation failed for batch {batch_index}: {exc}"
                ) from exc

            if "input_ids" not in inputs:
                raise ValueError(
                    f"Nemotron Omni processor output for batch {batch_index} is missing input_ids."
                )
            if image_inputs is not None:
                missing = [
                    name
                    for name in ("pixel_values", "num_patches", "num_tokens", "imgs_sizes")
                    if inputs.get(name) is None
                ]
                if missing:
                    raise ValueError(
                        f"Nemotron Omni processor output for batch {batch_index} "
                        f"is missing required field(s): {', '.join(missing)}."
                    )
            calib_data.append(inputs)

        del processor
        return calib_data

    def run_input_capture(self, example, use_cache: bool, data_device):
        if self._has_multimodal_inputs(example):
            # The top-level `forward()` requires `image_flags` and calls
            # `torch.distributed.get_rank()` unconditionally, which raises
            # outside a process group. `generate()` performs the same vision
            # merge without either requirement; the capture hook aborts it at
            # the first decoder layer, so no tokens are actually generated.
            capture_keys = (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "pixel_values_videos",
                "sound_clips",
                "sound_length",
            )
            capture_inputs = {
                key: example[key] for key in capture_keys if example.get(key) is not None
            }
            return self.model.generate(**capture_inputs, max_new_tokens=1)
        return self.model.language_model(**example, use_cache=use_cache)

    def monkey_patch(self):
        if not self.load_quantized_model:
            return

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_attn_2_available,
            is_mamba_2_ssm_available,
        )

        if is_mamba_2_ssm_available():
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
        else:
            mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

        if is_causal_conv1d_available():
            from gptqmodel.hf_kernels.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        else:
            causal_conv1d_update, causal_conv1d_fn = None, None

        if is_flash_attn_2_available():
            pass

        is_fast_path_available = all(
            (
                selective_state_update,
                mamba_chunk_scan_combined,
                mamba_split_conv1d_scan_combined,
                causal_conv1d_fn,
                causal_conv1d_update,
            )
        )

        def forward(
                self,
                hidden_states,
                cache_params=None,
                cache_position=None,
                attention_mask=None,
        ):
            def call_mixer_compat(fn):
                call_args = (hidden_states, cache_params, cache_position, attention_mask)
                try:
                    arity = len(inspect.signature(fn).parameters)
                except (TypeError, ValueError):
                    arity = 3
                arity = max(1, min(arity, len(call_args)))
                return fn(*call_args[:arity])

            if is_fast_path_available and "cuda" in self.in_proj.qweight.device.type:
                return call_mixer_compat(self.cuda_kernels_forward)
            dtype = hidden_states.dtype
            if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

            return call_mixer_compat(self.torch_forward)

        for layer in self.model.language_model.backbone.layers:
            if layer.mixer.__class__.__name__ == "NemotronHMamba2Mixer":
                mixer_class = type(layer.mixer)
                mixer_class.forward = forward
