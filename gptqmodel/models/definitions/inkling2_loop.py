# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from inspect import signature
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor, ProcessorMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from ...hf_inkling2_loop import Inkling2LoopConfig, Inkling2LoopForConditionalGeneration

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


# Transformers 5.24 renamed the cache argument of the mask builders from
# ``past_key_values`` to ``cache``. ``Inkling2Loop`` only exists on releases
# with the new spelling, but the definition must import on the installed
# release too, so resolve the accepted name from the live signature.
_MASK_CACHE_KWARG = {
    factory: ("cache" if "cache" in signature(factory).parameters else "past_key_values")
    for factory in (create_causal_mask, create_sliding_window_causal_mask)
}

# The architecture is vendored in `gptqmodel/hf_inkling2_loop` (Apache-2.0, from
# huggingface/transformers#51050) so the released checkpoint loads on any
# supported Transformers release instead of waiting for the modeling code to
# land upstream. A native implementation still wins when the installed release
# ships one.
if "inkling2_loop" not in CONFIG_MAPPING:
    AutoConfig.register("inkling2_loop", Inkling2LoopConfig)
AutoModelForMultimodalLM.register(
    Inkling2LoopConfig, Inkling2LoopForConditionalGeneration, exist_ok=True
)

# Every decoder layer takes the attention-residual bank as its second
# positional argument. The generic capture keeps only ``hidden_states``, so the
# definition carries the bank across layer replays in a per-batch holder. A
# plain object (not a dict/list/tensor) survives the looper's device moves
# unchanged, which is what keeps one holder per calibration batch.
_BLOCK_STATES_CARRY_KEY = "_inkling2_loop_block_states_carry"


class _BlockStatesCarry:
    """Running attention-residual bank for one calibration batch."""

    __slots__ = ("states", "layer")

    def __init__(self):
        self.states = None
        self.layer = None


class Inkling2LoopQModel(BaseQModel):
    """Inkling 2 Mini (fixed-loop, hybrid sliding/full attention, routed MoE).

    Checkpoint layout notes:

    - ``Inkling2Loop`` stores every decoder layer as one of four classes:
      sliding/full attention, each with a looped and a non-looped variant. The
      looped variants (layers ``loop_layer_index_start``..``loop_layer_index_end``)
      add the q/k/v/mlp message gates and are visited ``loop_num_passes`` times.
    - Attention projections are wrappers: ``q_proj``/``k_proj``/``v_proj`` hold a
      ``linear`` leaf (plus short-convolution and L2-clip children), ``o_proj``
      holds a gated output pair, and routed experts are 3D parameters that Defuser
      expands into per-expert ``gate_proj``/``up_proj``/``down_proj`` linears.
    - The released checkpoint stores NVFP4/MXFP8 experts; transformers converts
      them to dense tensors while loading, so GPT-QModel sees regular weights.
    """

    loader = AutoModelForMultimodalLM

    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    dynamic_expert_index = "n_routed_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"

    # ``o_proj.linear`` consumes the concatenated attention heads, whose width
    # differs from the key/value projections under GQA.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj.linear"]

    @classmethod
    def after_defuser_conversion(cls, model):
        """Require every decoder layer's routed experts to be expanded.

        ``Inkling2Loop`` keeps routed experts as 3-D parameters. Without a
        Defuser registry entry for ``inkling2_loop`` the conversion silently
        returns False and all experts would keep their source dtype while the
        declared expert subsets resolve to nothing.
        """

        config = getattr(model.config, "text_config", model.config)
        expected = int(config.num_hidden_layers)
        containers = [
            module for name, module in model.named_modules() if name.endswith(".mlp.experts")
        ]
        if len(containers) != expected:
            raise RuntimeError(
                "Inkling2Loop expert defusion coverage mismatch: "
                f"converted={len(containers)} expected={expected}. The installed Defuser has "
                "no `inkling2_loop` registry entry; upgrade Defuser before quantizing."
            )
        for experts in containers:
            if getattr(experts, "gate_up_proj", None) is not None or getattr(experts, "down_proj", None) is not None:
                raise RuntimeError(
                    "Inkling2Loop routed experts were not defused into per-expert Linear modules. "
                    "The installed Defuser has no `inkling2_loop` registry entry; upgrade Defuser "
                    "before quantizing."
                )
            for index in range(int(config.n_routed_experts)):
                expert = getattr(experts, str(index), None)
                if expert is None or not all(
                    hasattr(expert, part) for part in ("gate_proj", "up_proj", "down_proj")
                ):
                    raise RuntimeError(
                        f"Inkling2Loop expert {index} has no gate/up/down projections after defusion."
                    )

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "q_input_layernorm": ("q_input_layernorm:!",),
            "k_input_layernorm": ("k_input_layernorm:!",),
            "v_input_layernorm": ("v_input_layernorm:!",),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "attention_input": ("attention_input:!",),
            "mlp_input": ("mlp_input:!",),
            # ``q_proj``/``k_proj``/``v_proj`` are wrappers around a ``linear``
            # leaf plus non-quantized short-convolution / L2-clip children. The
            # three leaves share one quantization subset (subset ids only order
            # work; every leaf still captures its own input). ``o_proj`` is a
            # gated output projection: its ``linear`` consumes the attention
            # output, its ``gate_proj`` the attention input, so they are split.
            "self_attn": (
                "q_proj.linear:0:q",
                "k_proj.linear:0:k",
                "v_proj.linear:0:v",
                "o_proj.linear:1:o",
                "o_proj.gate_proj:2:gate",
                # Per-head L2-clip scales, short-convolution taps (static for
                # sliding layers, dynamic for full layers), the attention sink
                # and the relative-position bias bank stay in full precision.
                "q_proj.norm:!",
                "k_proj.sconv:!",
                "k_proj.sconv.conv1d:!",
                "k_proj.sconv.weight_proj:!",
                "v_proj.sconv:!",
                "v_proj.sconv.conv1d:!",
                "v_proj.sconv.weight_proj:!",
                "sink_proj:!",
                "relative_position_bias:!",
                "relative_position_bias.linear:!",
                "relative_position_bias.norm:!",
            ),
            "post_attention_convolution.conv1d": ("conv1d:!",),
            "post_mlp_convolution.conv1d": ("conv1d:!",),
            "q_message_gate.down_proj": ("down_proj:!",),
            "q_message_gate.forget_gate_proj": ("forget_gate_proj:!",),
            "q_message_gate.message_gate_proj": ("message_gate_proj:!",),
            "k_message_gate.down_proj": ("down_proj:!",),
            "k_message_gate.forget_gate_proj": ("forget_gate_proj:!",),
            "k_message_gate.message_gate_proj": ("message_gate_proj:!",),
            "v_message_gate.down_proj": ("down_proj:!",),
            "v_message_gate.forget_gate_proj": ("forget_gate_proj:!",),
            "v_message_gate.message_gate_proj": ("message_gate_proj:!",),
            "mlp_message_gate.down_proj": ("down_proj:!",),
            "mlp_message_gate.forget_gate_proj": ("forget_gate_proj:!",),
            "mlp_message_gate.message_gate_proj": ("message_gate_proj:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.final_input, self.quantize_config.device)
        self.shell_module_materialize(language_model.message_input, self.quantize_config.device)
        self.shell_module_materialize(core_model.vision_tower, self.quantize_config.device)

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
                module=language_model.final_input,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.message_input,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.vision_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.final_input = move_to(language_model.final_input, device=CPU)
        language_model.message_input = move_to(language_model.message_input, device=CPU)
        core_model.vision_tower = move_to(core_model.vision_tower, device=CPU)

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calibration_data = []
        for batch in batched_conversations(calibration_dataset, batch_size):
            calibration_data.append(self.prepare_inputs_for_conversations(processor, batch))
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

        first_parameter = next(self.model.model.vision_tower.parameters(), None)
        if first_parameter is not None:
            example["pixel_values"] = pixel_values.to(
                device=first_parameter.device,
                dtype=first_parameter.dtype,
            )
        return example

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Install the per-batch attention-residual bank used by every layer replay."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args,
            kwargs,
            batch_device,
            layer_input_kwargs,
        )
        layer_input_kwargs[_BLOCK_STATES_CARRY_KEY] = _BlockStatesCarry()
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        carry = additional_inputs.pop(_BLOCK_STATES_CARRY_KEY, None)
        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs

        self_attn = getattr(layer, "self_attn", None)
        layer_config = getattr(self_attn, "config", None)
        if layer_config is None:
            return additional_inputs

        # ``block_states`` is the bank of block-boundary inputs appended by the
        # model's outer loop. Rebuild it from the boundary layers already
        # replayed in this batch: the bank a layer consumes is the bank before
        # that layer appends its own input, and every subset of the same layer
        # must see the identical bank.
        hidden_states = layer_input[0]
        layer_idx = getattr(self_attn, "layer_idx", None)
        if carry is None or (carry.states is None and layer_idx):
            # The bank only exists as state carried from the earlier replays;
            # a cache captured without this definition cannot supply it.
            raise RuntimeError(
                "Inkling2Loop replay has no attention-residual bank for layer "
                f"{layer_idx}; restart without a mid-model checkpoint so the "
                "calibration inputs are captured by this definition."
            )
        if carry.states is None:
            # Layer 0 consumes the empty bank; it then appends its own input.
            carry.states = hidden_states.new_zeros(
                *hidden_states.shape[:-1], 0, hidden_states.shape[-1]
            ).to(device=CPU)
        block_states = carry.states
        if carry.layer is not layer:
            if getattr(layer, "is_block_boundary", False):
                carry.states = torch.cat(
                    [
                        carry.states,
                        move_to(hidden_states.detach(), device=CPU).unsqueeze(-2),
                    ],
                    dim=-2,
                )
            carry.layer = layer
        additional_inputs["block_states"] = move_to(block_states, device=target_device)

        # The first captured layer provides the sliding-window mask. Rebuild the
        # mask for every replayed layer so full-attention blocks do not inherit
        # the sliding window (and vice versa).
        padding_mask = additional_inputs.get("conv_mask")
        if not torch.is_tensor(padding_mask) or padding_mask.ndim != 2:
            captured_mask = additional_inputs.get("attention_mask")
            padding_mask = (
                captured_mask
                if torch.is_tensor(captured_mask) and captured_mask.ndim == 2
                else None
            )

        mask_factory = (
            create_sliding_window_causal_mask
            if getattr(layer, "attention_type", None) == "sliding_attention"
            else create_causal_mask
        )
        mask_kwargs = {
            "config": layer_config,
            "inputs_embeds": layer_input[0],
            "attention_mask": padding_mask,
            "position_ids": additional_inputs.get("position_ids"),
            "layer_idx": getattr(self_attn, "layer_idx", None),
        }
        mask_kwargs[_MASK_CACHE_KWARG[mask_factory]] = additional_inputs.get("past_key_values")
        additional_inputs["attention_mask"] = mask_factory(**mask_kwargs)
        return additional_inputs


__all__ = ["Inkling2LoopQModel"]
