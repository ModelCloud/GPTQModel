# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Intern-S2-Mobius quantization definition.

Mobius keeps the routed experts in four globally shared ``meta_mlp`` blocks;
decoder layers only own their small shared expert.  The second module-tree
variant deliberately exposes the shared blocks as auxiliary layer units so
their canonical paths remain ``model.language_model.meta_mlp.N``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from ...looper.input_cache import InputCache
from ...utils.model import MODALITY, get_module
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from .intern_s2_preview import InternS2PreviewQModel


class InternS2MobiusMoELifecycleHooks(GateUpDownMoELifecycleHooks):
    """Use the generic all-experts replay for a root auxiliary MoE unit."""

    def get_moe_block_for_subset(
        self,
        layer_module: nn.Module,
        model_class: type,
        current_subset: Dict[str, Any] | None = None,
    ) -> nn.Module | None:
        if hasattr(layer_module, "gate") and hasattr(layer_module, "experts"):
            return layer_module
        return super().get_moe_block_for_subset(
            layer_module,
            model_class,
            current_subset=current_subset,
        )


class InternS2MobiusQModel(InternS2PreviewQModel):
    """GPTQModel support for ``model_type=interns2_mobius``."""

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    force_serial_layer_replay = True
    moe_lifecycle_hooks = InternS2MobiusMoELifecycleHooks()
    auxiliary_layer_nodes = ("model.language_model.meta_mlp",)

    # Keep the preview decoder layout, but replace its routed MoE with the
    # per-layer shared expert and add the globally shared auxiliary stack.
    module_tree = [
        [
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
                "linear_attn": (
                    "norm:!",
                    "conv1d:!",
                    "in_proj_qkv:0",
                    "in_proj_z:1",
                    "in_proj_b:!:1",
                    "in_proj_a:!:1",
                    "out_proj:2",
                ),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": {
                    "shared_expert:0": (
                        "gate_proj:0",
                        "up_proj:0",
                        "down_proj:1",
                    ),
                    "shared_expert_gate": ("shared_expert_gate:!",),
                },
            },
        ],
        [
            "model",
            "language_model",
            "meta_mlp",
            "#",
            {
                # The router is a [num_experts, hidden] Parameter rather than
                # an nn.Linear; retain it in float while quantizing experts.
                "gate": ("gate:!",),
                "experts:moe": {
                    "#": (
                        "gate_proj:0",
                        "up_proj:0",
                        "down_proj:1",
                    ),
                },
            },
        ],
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._intern_aux_capture_handles: List[Any] = []
        self._intern_aux_capture_sinks: Dict[int, Dict[str, Any]] = {}
        self._intern_aux_capture_active: int | None = None
        self._intern_aux_capture_context: tuple[int, int] | None = None
        self._intern_aux_active_block_index: int | None = None

    def resolve_auxiliary_layer_name(self, layer_name: str | None) -> str | None:
        if not isinstance(layer_name, str):
            return None
        prefix = "model.language_model.meta_mlp."
        if layer_name.startswith(prefix) and layer_name[len(prefix):].isdigit():
            return layer_name
        return None

    def begin_auxiliary_input_capture(self, processor: Any = None) -> None:
        sink_key = id(processor) if processor is not None else 0
        self._intern_aux_capture_sinks[sink_key] = {
            "batches": {},
            "seen": set(),
            "finalized": False,
        }
        self._intern_aux_capture_active = sink_key
        self._intern_aux_capture_context = None

        try:
            meta_mlp = self.model.model.language_model.meta_mlp
        except AttributeError:
            return

        if self._intern_aux_capture_handles:
            return

        def capture(module, args):
            del module
            context = self._intern_aux_capture_context
            sink = self._intern_aux_capture_sinks.get(self._intern_aux_capture_active)
            if sink is None or sink["finalized"] or context is None or not args or not torch.is_tensor(args[0]):
                return
            layer_index, batch_index = context
            block_index = layer_index % len(meta_mlp)
            key = (block_index, layer_index, batch_index)
            if key in sink["seen"]:
                return
            sink["seen"].add(key)
            hidden_states = args[0].detach().to(device="cpu")
            sink["batches"].setdefault(block_index, {}).setdefault(batch_index, []).append(
                hidden_states.reshape(-1, hidden_states.shape[-1])
            )

        for block in meta_mlp:
            self._intern_aux_capture_handles.append(block.register_forward_pre_hook(capture))

    def finalize_auxiliary_input_capture(self, processor: Any = None):
        sink_key = id(processor) if processor is not None else self._intern_aux_capture_active
        sink = self._intern_aux_capture_sinks.get(sink_key)
        if sink is None:
            return {}
        cached = sink.get("caches")
        if cached is not None:
            return cached
        sink["finalized"] = True
        self._intern_aux_capture_context = None

        # Decoder replay may leave its final shared block on the accelerator.
        # Auxiliary quantization starts with block 0 and manages its own expert
        # residency, so release the decoder's resident block at this boundary.
        try:
            meta_mlp = self.model.model.language_model.meta_mlp
            if self._intern_aux_active_block_index is not None:
                self._offload_auxiliary_block(meta_mlp[self._intern_aux_active_block_index])
        except (AttributeError, IndexError):
            pass
        self._intern_aux_active_block_index = None

        caches = {}
        prefix = "model.language_model.meta_mlp"
        for block_index, batches in sink["batches"].items():
            if not batches:
                continue
            batch_count = max(batches) + 1
            layer_inputs = []
            for batch_index in range(batch_count):
                pieces = batches.get(batch_index, [])
                layer_inputs.append([torch.cat(pieces, dim=0)] if pieces else [])
            cache = InputCache(
                layer_inputs=layer_inputs,
                layer_input_kwargs=[{} for _ in range(batch_count)],
                position_ids=[],
                attention_masks=[None for _ in range(batch_count)],
            )
            caches[f"{prefix}.{block_index}"] = cache
        sink["caches"] = caches
        return caches

    def close_auxiliary_input_capture(self) -> None:
        # The decoder replay keeps only the block used by its current layer on
        # the replay device.  Release that final resident block as well; the
        # other blocks were offloaded before their successors were loaded.
        try:
            meta_mlp = self.model.model.language_model.meta_mlp
            if self._intern_aux_active_block_index is not None:
                self._offload_auxiliary_block(meta_mlp[self._intern_aux_active_block_index])
        except (AttributeError, IndexError):
            pass
        for handle in self._intern_aux_capture_handles:
            handle.remove()
        self._intern_aux_capture_handles = []
        self._intern_aux_capture_sinks = {}
        self._intern_aux_capture_active = None
        self._intern_aux_capture_context = None
        self._intern_aux_active_block_index = None

    @staticmethod
    def _offload_auxiliary_block(block: nn.Module) -> None:
        """Move one shared block off the replay device when it is no longer active."""

        tensors = [*block.parameters(), *block.buffers()]
        if any(tensor.device.type not in ("cpu", "meta") for tensor in tensors):
            block.to(device=torch.device("cpu"))

    def before_layer_forward(
        self,
        layer: nn.Module,
        layer_index: int,
        batch_index: int,
        processor: Any,
        layer_input: List[torch.Tensor],
        additional_inputs: Dict[str, Any],
        target_device: torch.device,
    ) -> None:
        del layer_input, additional_inputs, target_device
        sink_key = id(processor)
        if sink_key in self._intern_aux_capture_sinks:
            self._intern_aux_capture_active = sink_key
        try:
            meta_mlp = self.model.model.language_model.meta_mlp
        except AttributeError:
            self._intern_aux_capture_context = None
            return
        if any(layer is block for block in meta_mlp):
            self._intern_aux_capture_context = None
        else:
            self._intern_aux_capture_context = (layer_index, batch_index)

    def prepare_layer_replay_kwargs(
        self,
        layer: nn.Module,
        layer_input: List[torch.Tensor],
        additional_inputs: Dict[str, Any],
        target_device: torch.device,
    ) -> Dict[str, Any]:
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        language_model = self.model.model.language_model
        meta_mlp = language_model.meta_mlp
        if any(layer is block for block in meta_mlp):
            # A MetaMoeBlock is replayed directly from its independent cache;
            # it accepts only hidden_states, unlike decoder layers which take
            # attention/cache kwargs.
            for key in ("attention_mask", "position_ids", "kv_last_layer", "use_cache"):
                additional_inputs.pop(key, None)
            return additional_inputs
        if not hasattr(layer, "layer_idx"):
            return additional_inputs

        # Only the block used by this decoder layer needs to be resident on the
        # replay device.  Other global blocks remain lazy/offloaded.
        block_index = int(layer.layer_idx) % len(meta_mlp)
        previous_index = self._intern_aux_active_block_index
        if previous_index is not None and previous_index != block_index:
            self._offload_auxiliary_block(meta_mlp[previous_index])
        block = meta_mlp[block_index]
        if any(parameter.device.type == "meta" for parameter in block.parameters()):
            block = self.shell_module_materialize(
                target_submodule=block,
                device=target_device,
                module_path=f"model.language_model.meta_mlp.{block_index}",
            )
            meta_mlp[block_index] = block
        elif next(block.parameters(), None) is not None:
            block_device = next(block.parameters()).device
            if block_device != target_device:
                block.to(target_device)
        self._intern_aux_active_block_index = block_index
        additional_inputs["meta_mlp"] = meta_mlp
        return additional_inputs

    @classmethod
    def after_defuser_conversion(cls, model):
        """Defuse packed Mobius experts when Defuser's registry is absent."""

        try:
            from defuser.modeling.moe_experts_interface import _unfuse_experts_weights_inplace
        except Exception:
            return
        language_model = get_module(model, "model.language_model")
        meta_mlp = getattr(language_model, "meta_mlp", None)
        if meta_mlp is None:
            return
        for block in meta_mlp:
            experts = getattr(block, "experts", None)
            if experts is not None:
                _unfuse_experts_weights_inplace(experts)


__all__ = ["InternS2MobiusQModel", "InternS2MobiusMoELifecycleHooks"]
