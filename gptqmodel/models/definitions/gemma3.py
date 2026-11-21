# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace
from typing import Dict

from transformers import ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from ...utils.structure import LazyTurtle
from .._const import CPU
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from . import LlamaQModel
from ._vlm_calibration import (
    finalize_conversation_batches,
    pop_conversation_calibration_options,
    split_calibration_dataset,
)


_GEMMA3_REPLAY_MASK_MAPPING = "__gptqmodel_gemma3_replay_mask_mapping"


class Gemma3QModel(LlamaQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    fusable = FusableConfig("gemma3_text").merge(LlamaFusableConfig())

    shared_input_verified_model_types = frozenset({"gemma3_text"})

    # Gemma 3 keeps Gemma 2's MLP norms and adds per-head Q/K RMSNorm inside
    # attention. All four own checkpoint weights and stay unquantized, so they
    # are declared as excluded leaves of the tree.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            "self_attn": (
                "q_proj:0:q:in=x",
                "k_proj:0:k:in=x",
                "v_proj:0:v:in=x",
                "q_norm:!",
                "k_norm:!",
                "o_proj:1:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]


class Gemma3ForConditionalGenerationGPTQ(BaseQModel):
    fusable = FusableConfig("gemma3").merge(Gemma3QModel.fusable)

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
        processor_class="gptqmodel.models.ports.native.auto.NativeProcessor",
        tokenizer_class="gptqmodel.models.ports.native.auto.NativeTokenizer",
    )

    support_batch_quantize = False
    pre_lm_head_norm_module = "model.language_model.norm"

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True

    @classmethod
    def checkpoint_key_for_save(cls, name: str) -> str:
        """Use the model's HF conversion map when writing checkpoint keys."""
        return LazyTurtle.convert_hf_checkpoint_key(name, cls.HF_CONVERSION_MAP_REVERSED)

    @classmethod
    def checkpoint_key_for_load(cls, name: str) -> str:
        """Reverse the model's HF conversion map when loading checkpoint keys."""
        return LazyTurtle.convert_hf_checkpoint_key(
            name,
            cls.HF_CONVERSION_MAP_REVERSED,
            to_runtime=True,
        )

    def prepare_quantize_config_for_save(self, quantize_config) -> None:
        # Gemma 3's vision tower and projector remain in their source dtype.
        # Tell vLLM's GPTQ loader not to allocate quantized linear layers for them.
        dynamic = dict(quantize_config.dynamic or {})
        dynamic.setdefault(r"-:vision_tower\..*", {})
        dynamic.setdefault(r"-:multi_modal_projector\..*", {})
        quantize_config.dynamic = dynamic

    def copy_checkpoint_auxiliary_files_for_save(self, save_dir: str) -> None:
        # Keep the standalone processor/tokenizer files required by Transformers 4.x
        # and vLLM releases that predate the consolidated Transformers 5 format.
        import os
        import shutil

        filenames = (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        )
        for filename in filenames:
            source = os.path.join(self.model_local_path, filename)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(save_dir, filename))

    HF_CONVERSION_MAP_REVERSED = (
        # Gemma 3 shells expose `model.vision_tower.*`, but checkpoint tensors
        # live under `vision_tower.vision_model.*`, so restore that missing hop first.
        SimpleNamespace(
            source_patterns=[r"^model\.vision_tower\.(?!vision_model\.)(.+)$"],
            target_patterns=[r"^vision_tower.vision_model.\1"],
            operations=[],
        ),
        SimpleNamespace(
            # Runtime `model.language_model.<rest>` <-> checkpoint
            # `language_model.model.<rest>`. Keep the explicit capture group so
            # both directions carry the suffix instead of relying on a prefix
            # replace (issue #4802).
            source_patterns=[r"^model\.language_model\.(.+)$"],
            target_patterns=[r"^language_model.model.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"lm_head"],
            target_patterns=[r"^language_model.lm_head"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.vision_tower"],
            target_patterns=[r"^vision_tower"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.multi_modal_projector"],
            target_patterns=[r"^multi_modal_projector"],
            operations=[],
        ),
    )

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0:q:in=x",
                "k_proj:0:k:in=x",
                "v_proj:0:v:in=x",
                "q_norm:!",
                "k_norm:!",
                "o_proj:1:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        }
    ]

    lm_head_module = "model.lm_head"

    def _language_model(self):
        return self.model.model.language_model

    def _start_replay_mask_capture(self) -> None:
        """Capture the per-layer-type masks Gemma-3's outer model builds.

        ``Gemma3Model.forward`` hands the text decoder a ``full_attention`` /
        ``sliding_attention`` mask dict built from ``token_type_ids`` so image
        tokens attend bidirectionally inside each image block. The first-layer
        hook only sees the tensor selected for layer 0, so keep the whole dict
        for replay instead of rebuilding it without the image-block overlay.
        """

        self._stop_replay_mask_capture()
        language_model = self._language_model()

        def capture_outer_inputs(module, args, kwargs):
            del args
            attention_mask = kwargs.get("attention_mask")
            if isinstance(attention_mask, dict):
                setattr(module, _GEMMA3_REPLAY_MASK_MAPPING, attention_mask)

        self._gemma3_mask_capture_handle = language_model.register_forward_pre_hook(
            capture_outer_inputs,
            with_kwargs=True,
        )

    def _stop_replay_mask_capture(self) -> None:
        handle = getattr(self, "_gemma3_mask_capture_handle", None)
        if handle is not None:
            handle.remove()
            delattr(self, "_gemma3_mask_capture_handle")

        language_model = self._language_model()
        if hasattr(language_model, _GEMMA3_REPLAY_MASK_MAPPING):
            delattr(language_model, _GEMMA3_REPLAY_MASK_MAPPING)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args,
            kwargs,
            batch_device,
            layer_input_kwargs,
        )
        mask_mapping = getattr(self._language_model(), _GEMMA3_REPLAY_MASK_MAPPING, None)
        if isinstance(mask_mapping, dict):
            layer_input_kwargs[_GEMMA3_REPLAY_MASK_MAPPING] = nested_move_to(
                mask_mapping,
                device=batch_device,
            )
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        mask_mapping = additional_inputs.pop(_GEMMA3_REPLAY_MASK_MAPPING, None)
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        if not isinstance(mask_mapping, dict):
            return additional_inputs

        layer_types = self._declared_layer_types() or ()
        layer_type = self._resolve_layer_replay_type(layer, layer_types)
        mask = mask_mapping.get(layer_type) if layer_type is not None else None
        if mask is None:
            # Single-type stacks keep the layer-0 mask for every layer.
            mask = next(iter(mask_mapping.values()), None)
        if mask is not None:
            additional_inputs["attention_mask"] = move_to(mask, device=target_device)
        return additional_inputs

    def pre_quantize_generate_hook_start(self):
        language_model = self.model.model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(language_model.norm, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.multi_modal_projector, self.quantize_config.device)
        self._start_replay_mask_capture()

    def pre_quantize_generate_hook_end(self):
        self._stop_replay_mask_capture()
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
                model=language_model,
                module=language_model.norm,
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
        language_model.norm = move_to(language_model.norm, device=CPU)
        self.model.model.vision_tower = move_to(self.model.model.vision_tower, device=CPU)
        self.model.model.multi_modal_projector = move_to(self.model.model.multi_modal_projector, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        # BaseQModel already loaded the vendored processor with the normalized
        # tokenizer through this definition's loader.
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
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        label = type(self).__name__
        pretokenized, conversations = split_calibration_dataset(calibration_dataset, label=label)
        if pretokenized is not None:
            return super().prepare_dataset(pretokenized, batch_size=batch_size, **kwargs)

        sort, min_length = pop_conversation_calibration_options(kwargs, label=label)
        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(conversations, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return finalize_conversation_batches(
            calib_data, sort=sort, min_length=min_length, label=label
        )
