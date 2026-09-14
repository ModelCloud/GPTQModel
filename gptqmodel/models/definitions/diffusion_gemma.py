# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import MethodType, SimpleNamespace
from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoModelForMultimodalLM, AutoProcessor, ProcessorMixin
from transformers.pytorch_utils import id_tensor_storage

from ...nn_modules.qlinear import BaseQuantLinear
from ...utils.calibration import batched_conversations
from ...utils.device import get_device
from ...utils.model import MODALITY, get_module_by_name_prefix, move_to, nested_move_to
from ...utils.torch import torch_empty_cache
from ..base import BaseQModel


_DIFFUSION_GEMMA_MASKS = "__gptqmodel_diffusion_gemma_attention_masks"


def _language_model(model):
    return model.model.encoder.language_model


def _patch_attention_mask_capture(model):
    language_model = _language_model(model)
    if getattr(language_model, "_gptqmodel_attention_mask_capture_patched", False):
        return

    original = language_model.forward

    # DiffusionGemma supplies one mask per attention layer type. The regular
    # capture hook preserves tensor arguments, but this dict is a model-level
    # side input, so retain it while the first layer is being observed.
    def patched(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, dict):
            self._gptqmodel_attention_mask_mapping = attention_mask
        return original(*args, **kwargs)

    language_model._gptqmodel_original_forward = original
    language_model.forward = MethodType(patched, language_model)
    language_model._gptqmodel_attention_mask_capture_patched = True


def _restore_attention_mask_capture(model):
    language_model = _language_model(model)
    original = getattr(language_model, "_gptqmodel_original_forward", None)
    if original is not None:
        language_model.forward = original
        delattr(language_model, "_gptqmodel_original_forward")
    for name in (
        "_gptqmodel_attention_mask_capture_patched",
        "_gptqmodel_attention_mask_mapping",
    ):
        if hasattr(language_model, name):
            delattr(language_model, name)


def _sync_decoder_quantized_modules(model, encoder_layer: nn.Module | None = None) -> int:
    """Share quantized encoder projections with the weight-tied diffusion decoder."""

    core = model.model
    encoder_layers = core.encoder.language_model.layers
    decoder_layers = core.decoder.layers
    if encoder_layer is None:
        layer_pairs = zip(encoder_layers, decoder_layers)
    else:
        layer_index = next((index for index, layer in enumerate(encoder_layers) if layer is encoder_layer), None)
        if layer_index is None:
            return 0
        layer_pairs = ((encoder_layer, decoder_layers[layer_index]),)

    synced = 0
    for source_layer, target_layer in layer_pairs:
        for relative_name, source_module in source_layer.named_modules():
            if not relative_name or not isinstance(source_module, BaseQuantLinear):
                continue
            # Calibration quantizes only the encoder tree, while generation
            # executes the tied decoder too. Reuse the same module instance so
            # both paths expose one quantized representation.
            parent_path, _, leaf_name = relative_name.rpartition(".")
            target_parent = target_layer.get_submodule(parent_path) if parent_path else target_layer
            setattr(target_parent, leaf_name, source_module)
            synced += 1
    return synced


def _sync_decoder_tied_parameters(model) -> int:
    """Restore dense encoder/decoder parameter aliases after checkpoint loading."""

    core = model.model
    synced = 0
    for encoder_layer, decoder_layer in zip(core.encoder.language_model.layers, core.decoder.layers):
        # Checkpoint loading may materialize two independent dense Parameters
        # even when Transformers declared them tied. Restore the alias before
        # rebuilding save-time metadata; quantized leaves are handled above.
        decoder_parameters = dict(decoder_layer.named_parameters(remove_duplicate=False))
        for relative_name, encoder_parameter in encoder_layer.named_parameters(remove_duplicate=False):
            decoder_parameter = decoder_parameters.get(relative_name)
            if decoder_parameter is None or decoder_parameter.shape != encoder_parameter.shape:
                continue
            parent_path, _, leaf_name = relative_name.rpartition(".")
            decoder_parent = decoder_layer.get_submodule(parent_path) if parent_path else decoder_layer
            setattr(decoder_parent, leaf_name, encoder_parameter)
            synced += 1
    return synced


def _same_tensor_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device.type == "meta" or right.device.type == "meta":
        return left is right
    return id_tensor_storage(left) == id_tensor_storage(right)


def _refresh_tied_weight_metadata(model, include_layer_ties: bool = False) -> None:
    """Refresh tie metadata after defusion/quantization changes module aliases."""

    core = model.model
    tied_weights = getattr(core, "_tied_weights_keys", None) or {}
    layer_markers = ("encoder.language_model.layers", "decoder.layers")
    # Defusion changes the module tree after the original tie map was built.
    # Keep only stable non-layer aliases here, then infer layer aliases from
    # live storage below so stale mappings cannot hide checkpoint tensors.
    stable_weights = {
        target: source
        for target, source in tied_weights.items()
        if not any(marker in str(target) or marker in str(source) for marker in layer_markers)
    }

    if include_layer_ties:
        encoder_layers = core.encoder.language_model.layers
        decoder_layers = core.decoder.layers
        for layer_index, (encoder_layer, decoder_layer) in enumerate(zip(encoder_layers, decoder_layers)):
            encoder_tensors = dict(encoder_layer.named_parameters(remove_duplicate=False))
            encoder_tensors.update(encoder_layer.named_buffers(remove_duplicate=False))
            decoder_tensors = dict(decoder_layer.named_parameters(remove_duplicate=False))
            decoder_tensors.update(decoder_layer.named_buffers(remove_duplicate=False))

            for relative_name, encoder_tensor in encoder_tensors.items():
                decoder_tensor = decoder_tensors.get(relative_name)
                if decoder_tensor is None or not _same_tensor_storage(encoder_tensor, decoder_tensor):
                    continue
                stable_weights[f"decoder.layers.{layer_index}.{relative_name}"] = (
                    f"encoder.language_model.layers.{layer_index}.{relative_name}"
                )

    core._tied_weights_keys = stable_weights
    core.all_tied_weights_keys = core.get_expanded_tied_weights_keys(all_submodels=False)
    model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)


class DiffusionGemmaQModel(BaseQModel):
    """Quantization definition for native Transformers DiffusionGemma checkpoints."""

    loader = AutoModelForMultimodalLM
    require_load_processor = True
    support_batch_quantize = False
    layer_modules_strict = False
    turtle_materialize_tie_weights = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    dynamic_expert_index = "num_experts"
    pre_lm_head_norm_module = "model.decoder.norm"
    rotary_embedding = "model.encoder.language_model.rotary_emb"

    # Released checkpoints store tied text weights only under the decoder. The
    # quantization shell captures the encoder, so LazyTurtle must source those
    # tensors from the decoder namespace while materializing one layer at a time.
    HF_CONVERSION_MAP_REVERSED = (
        SimpleNamespace(
            source_patterns=[r"model\.encoder\.language_model"],
            target_patterns=[r"^model.decoder"],
            operations=[],
        ),
    )

    # Encoder and decoder text projections are weight-tied. Quantize the encoder
    # stack because its first-layer inputs can be captured without materializing
    # the complete decoder, then share the resulting QuantLinear modules below.
    module_tree = [
        "model",
        "encoder",
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
                "v_norm:!",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "post_feedforward_layernorm_1": ("post_feedforward_layernorm_1:!",),
            "pre_feedforward_layernorm_2": ("pre_feedforward_layernorm_2:!",),
            "router": ("proj:!",),
            "experts:moe": {
                "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
            "post_feedforward_layernorm_2": ("post_feedforward_layernorm_2:!",),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]

    @classmethod
    def get_base_modules(cls, model):
        # The decoder is intentionally excluded during first-layer capture. Its
        # quantized leaves are restored from the encoder after quantization.
        return [name for name in super().get_base_modules(model) if name != "model.decoder"]

    @classmethod
    def after_defuser_conversion(cls, model):
        # Defuser changes the layer tree, so discard broad layer aliases before
        # Transformers expands the mappings. Exact aliases are restored after
        # quantization, once encoder and decoder modules actually share storage.
        _refresh_tied_weight_metadata(model)

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        if load_quantized_model:
            # A quantized checkpoint contains the encoder owner only. Reattach
            # decoder leaves and dense aliases before Transformers computes its
            # expanded tied-key map for subsequent saves.
            _sync_decoder_quantized_modules(model)
            synced_parameters = _sync_decoder_tied_parameters(model)
            _refresh_tied_weight_metadata(model, include_layer_ties=True)
            if synced_parameters:
                torch_empty_cache()
        return model

    def after_quantize(self) -> None:
        # Disk offload deliberately bypasses per-layer post_quantize hooks. A
        # final full-stack pass keeps the tied decoder correct in both modes.
        _sync_decoder_quantized_modules(self.model)
        _sync_decoder_tied_parameters(self.model)
        _refresh_tied_weight_metadata(self.model, include_layer_ties=True)

    def post_quantize(self, module: nn.Module) -> nn.Module:
        module = super().post_quantize(module)
        encoder_layers = self.model.model.encoder.language_model.layers
        if any(module is layer for layer in encoder_layers):
            _sync_decoder_quantized_modules(self.model, encoder_layer=module)
        return module

    def pre_quantize_generate_hook_start(self):
        _patch_attention_mask_capture(self.model)

        # These modules sit outside the quantized layer list but participate in
        # the calibration forward. Quantize them before capturing inputs so the
        # replayed activations have the same embedding/vision path as inference.
        encoder = self.model.model.encoder
        encoder.vision_tower = self.pre_quantize(encoder.vision_tower)
        encoder.embed_vision = self.pre_quantize(encoder.embed_vision)
        language_model = encoder.language_model
        language_model.embed_tokens = self.pre_quantize(language_model.embed_tokens)
        language_model.rotary_emb = self.pre_quantize(language_model.rotary_emb)

    def pre_quantize_generate_hook_end(self):
        _restore_attention_mask_capture(self.model)
        super().pre_quantize_generate_hook_end()

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args,
            kwargs,
            batch_device,
            layer_input_kwargs,
        )
        mask_mapping = getattr(_language_model(self.model), "_gptqmodel_attention_mask_mapping", None)
        if isinstance(mask_mapping, dict):
            layer_input_kwargs[_DIFFUSION_GEMMA_MASKS] = nested_move_to(mask_mapping, device=batch_device)
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        if not layer_input:
            return additional_inputs

        # Only the first-layer capture retains these model-specific kwargs.
        # Re-select the mask and rotary embeddings for each replayed layer.
        mask_mapping = additional_inputs.pop(_DIFFUSION_GEMMA_MASKS, None)
        layer_type = getattr(getattr(layer, "self_attn", None), "layer_type", None)
        if isinstance(mask_mapping, dict) and layer_type in mask_mapping:
            additional_inputs["attention_mask"] = move_to(mask_mapping[layer_type], device=target_device)

        rotary, _ = get_module_by_name_prefix(self.model, [self.rotary_embedding])
        if rotary is None or layer_type is None:
            return additional_inputs

        hidden_states = layer_input[0]
        batch_size = hidden_states.shape[0] if hidden_states.ndim >= 2 else 1
        sequence_length = hidden_states.shape[1] if hidden_states.ndim >= 2 else hidden_states.shape[0]
        position_ids = additional_inputs.get("position_ids")
        if position_ids is None or position_ids.shape[-1] != sequence_length:
            position_ids = torch.arange(sequence_length, device=target_device).unsqueeze(0).expand(batch_size, -1)
            additional_inputs["position_ids"] = position_ids

        try:
            rotary_device = get_device(rotary)
        except Exception:
            rotary_device = position_ids.device
        additional_inputs["position_embeddings"] = nested_move_to(
            rotary(
                torch.empty(1, device=rotary_device, dtype=hidden_states.dtype),
                move_to(position_ids, device=rotary_device),
                layer_type,
            ),
            device=target_device,
        )
        return additional_inputs

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calibration_data = []
        for batch in batched_conversations(calibration_dataset, batch_size):
            calibration_data.append(
                processor.apply_chat_template(
                    batch,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            )
        del processor
        return calibration_data

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        example = super().move_input_capture_example(example, data_device)
        pixel_values = example.get("pixel_values")
        if not torch.is_tensor(pixel_values) or not pixel_values.is_floating_point():
            return example

        first_parameter = next(self.model.model.encoder.vision_tower.parameters(), None)
        if first_parameter is not None:
            example["pixel_values"] = pixel_values.to(
                device=first_parameter.device,
                dtype=first_parameter.dtype,
            )
        return example


__all__ = ["DiffusionGemmaQModel"]
