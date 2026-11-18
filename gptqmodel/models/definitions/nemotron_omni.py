# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Optional

from PIL import Image
from transformers import ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..local import LocalModelLoader
from ._vlm_calibration import prepare_vlm_calibration_dataset
from .nemotron_h import NEMOTRON_H_LAYER_MODULES, NemotronHQModel


class NemotronOmniQModel(NemotronHQModel):
    """GPTQ wrapper for Nemotron Omni models that embed a Nemotron-H language backbone."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.nemotron_omni.configuration.NemotronOmniConfig",
        model_class="gptqmodel.models.ports.nemotron_omni.modeling.NemotronH_Omni_Reasoning_V3",
        processor_class="gptqmodel.models.ports.nemotron_omni.processing.NemotronOmniProcessor",
        tokenizer_class="gptqmodel.models.ports.nemotron_omni.processing.NemotronOmniTokenizer",
    )
    require_trust_remote_code = False
    require_monkeypatch = False
    fusable = None
    defuser_module_paths = ("language_model",)
    out_of_model_tensors = {"prefixes": ["vision_model.radio_model.input_conditioner",
                                        "sound_encoder.encoder.feature_extractor"]}

    # The language backbone is calibrated on activations that already contain
    # the vision tower's embeddings, so text-only calibration is not
    # representative of inference.
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    # Calibrate one multimodal conversation at a time.
    support_batch_quantize = False

    pre_lm_head_norm_module = "language_model.model.norm_f"

    # The embedded language backbone is the Nemotron-H decoder, so the layer
    # inventory must be the one `NemotronHQModel` declares. Reusing it keeps
    # routed experts, shared experts, the top-k router and the SSM convolution
    # declared here too; a local copy previously dropped all of them and left
    # most of the checkpoint in BF16 with no error (issue #4355).
    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        NEMOTRON_H_LAYER_MODULES,
    ]

    @classmethod
    def resolve_hf_conversion_map_reversed(cls, target_model=None):
        from ..ports.nemotron_omni.checkpoint import reversed_mapping

        return reversed_mapping()

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
            for name in ("pixel_values", "pixel_values_videos", "input_features", "input_features_mask")
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
        for attr_name in ("vision_model", "multi_modal_projector", "audio_tower", "embed_audio"):
            if hasattr(self.model, attr_name):
                self._materialize_top_level_module(attr_name)

        self.shell_module_materialize(self.model.language_model.model.norm_f, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            for attr_name in ("vision_model", "multi_modal_projector", "audio_tower", "embed_audio"):
                module = getattr(self.model, attr_name, None)
                if module is not None:
                    offload_to_disk(
                        model=self.model,
                        module=module,
                        disk_path=self.quantize_config.offload_to_disk_path,
                    )

            offload_to_disk(
                model=self.model.language_model.model,
                module=self.model.language_model.model.norm_f,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        for attr_name in ("vision_model", "multi_modal_projector", "audio_tower", "embed_audio"):
            module = getattr(self.model, attr_name, None)
            if module is not None:
                setattr(self.model, attr_name, move_to(module, device=CPU))

        self.model.language_model.model.norm_f = move_to(self.model.language_model.model.norm_f, device=CPU)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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
        return self.loader.load_processor(
            self.model_local_path, config=self.model.config, tokenizer=self.tokenizer.tokenizer,
        )

    def _prepare_calibration_batches(self, conversations, batch_size):
        processor = self.__dict__.get("processor") or self.load_processor()
        calib_data = []
        for batch_index, batch in enumerate(
            batched_conversations(
                conversations,
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
                    for name in ("pixel_values", "image_grid_hw")
                    if inputs.get(name) is None
                ]
                if missing:
                    raise ValueError(
                        f"Nemotron Omni processor output for batch {batch_index} "
                        f"is missing required field(s): {', '.join(missing)}."
                    )
            calib_data.append(inputs)

        return calib_data

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        return prepare_vlm_calibration_dataset(
            self,
            calibration_dataset,
            batch_size=batch_size,
            options=kwargs,
            prepare_conversations=lambda rows: self._prepare_calibration_batches(rows, batch_size),
        )

    def run_input_capture(self, example, use_cache: bool, data_device):
        return self.model(**example, use_cache=use_cache)

    def monkey_patch(self):
        """Native Transformers owns optional-kernel dispatch; never mutate classes."""

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # SSM gates, relu2 and routed/shared branches need joint replay qualification.
        return []

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        tree[-1]["mixer"] = tuple(dict.fromkeys((*tree[-1]["mixer"], "A_log:!", "D:!", "dt_bias:!")))
        tree[-1]["mixer.norm"] = ("norm:!",)
        return tree
