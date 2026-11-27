# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict, List, Optional

import torch
from transformers import ProcessorMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from ...quantization.awq.utils.module import get_op_name
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..looped_capture import LoopedLayerCaptureMixin
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from ..ports.inkling2_loop.experts import (
    Inkling2LoopPackedExpertLinear,
    Inkling2LoopPackedExperts,
    defuse_packed_experts,
)
from ._vlm_calibration import prepare_vlm_calibration_dataset


# Every decoder layer takes the attention-residual bank as its second
# positional argument. The generic capture keeps only ``hidden_states``, so the
# definition carries the bank across layer replays in a per-batch holder. A
# plain object (not a dict/list/tensor) survives the looper's device moves
# unchanged, which is what keeps one holder per calibration batch.
_BLOCK_STATES_CARRY_KEY = "_inkling2_loop_block_states_carry"


def declare_expert_storage(model) -> None:
    """Make ``experts_formats`` describe the storage the model now holds.

    A defused layer holds one quantized ``Linear`` per expert, so its
    ``experts_formats`` entry is cleared: leaving ``nvfp4``/``mxfp8`` there makes
    a reload build the packed container again, and the quantized expert tensors
    would have nowhere to land. A layer the quantization plan skipped is re-fused
    instead, so it keeps both its packed declaration and its packed storage.
    """

    config = getattr(model.config, "text_config", model.config)
    formats = getattr(config, "experts_formats", None)
    if not formats:
        return
    for name, experts in model.named_modules():
        if not name.endswith(".mlp.experts") or not getattr(experts, "defused", False):
            continue
        layer_index = _layer_index_from_name(name)
        if layer_index is None or layer_index >= len(formats):
            continue
        if any(
            isinstance(module, Inkling2LoopPackedExpertLinear)
            for module in experts.modules()
        ):
            # Re-fusion is contract-checked: the layer may only go back to
            # packed storage while every leaf still holds the checkpoint's own
            # payload with one identical quantization (contracts/model/refusion.md).
            experts.refuse(name)
            continue
        formats[layer_index] = None


def _layer_index_from_name(name: str) -> Optional[int]:
    """Decoder-layer index from a ``...layers.<i>.mlp.experts`` module name."""

    parts = name.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None


class _BlockStatesCarry:
    """Running attention-residual bank for one calibration batch."""

    __slots__ = ("states", "layer", "index")

    def __init__(self):
        self.states = None
        self.layer = None
        # Position of this calibration batch in the captured inputs. The
        # lm_head stage receives the cached batches in the same order, so the
        # final bank of each batch can be selected without a global bank.
        self.index = 0


class Inkling2LoopQModel(LoopedLayerCaptureMixin, BaseQModel):
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
    - The released checkpoint stores packed NVFP4 (layers 0-19) and MXFP8
      (20-23) experts under `mlp.experts.w13_*` / `w2_*`, plus a checkpoint-side
      attention naming (`attn.wq_du`, `attn.wo_ud`, ...) that the installed
      Transformers release does not rename yet. Loading those tensors needs the
      conversion mapping and the packed-expert decode hook; the decode kernels
      themselves are in `gptqmodel/utils/nvfp4_triton.py`.
    - A pre-quantized checkpoint is re-quantized, not passed through: every
      packed expert is defused into per-expert `Linear` leaves (packed views)
      before the layer is planned, its calibration/Hessian forwards run the
      W4A16/W8A16 kernels, and the dense bf16 weight is materialized only when
      the quantization processor packs that module.
    """

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.inkling2_loop.Inkling2LoopConfig",
        model_class="gptqmodel.models.ports.inkling2_loop.Inkling2LoopForConditionalGeneration",
        processor_class="gptqmodel.models.ports.inkling2_loop.processing.Inkling2Processor",
        tokenizer_class="gptqmodel.models.ports.native.auto.NativeTokenizer",
    )

    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    dynamic_expert_index = "n_routed_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Keep only the per-expert ``up_proj -> down_proj`` fold.

        The generic planner follows module-tree order, which is not this layer's
        data flow: ``q_proj``/``k_proj``/``v_proj`` read ``input_layernorm``
        (through ``attention_input``) rather than the
        ``post_attention_layernorm`` the tree places before them, and
        ``o_proj.gate_proj`` reads the attention input rather than
        ``o_proj.linear``'s output. Folding either generic group changes the
        layer output (measured 8e-2..4e-1 relative), so both are dropped.

        The routed activation multiplies ``up_proj``'s output into
        ``down_proj``; the only intervening step is the bounded ``up``
        softcap, whose saturation is negligible at the calibrated scale
        (measured <4e-6 relative layer drift). Keep exactly one expert's
        ``up_proj -> down_proj`` pair per group; every other projection falls
        back to plain quantization.
        """
        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        prefix, suffix = "mlp.experts.", ".up_proj"
        kept = []
        for node in nodes:
            prev_op = node.get("prev_op")
            layers = node.get("layers") or []
            if not isinstance(prev_op, torch.nn.Linear) or len(layers) != 1:
                continue
            target = layers[0]
            if not isinstance(target, torch.nn.Linear):
                continue
            source_name = get_op_name(module, prev_op)
            if not (source_name.startswith(prefix) and source_name.endswith(suffix)):
                continue
            expert_index = source_name[len(prefix):-len(suffix)]
            if expert_index.isdigit() and get_op_name(module, target) == (
                f"{prefix}{expert_index}.down_proj"
            ):
                kept.append(node)
        return kept

    def pre_quantize(self, module, module_path=None):
        """Materialize a layer, then split its packed experts into quantizable leaves.

        A checkpoint that already stores NVFP4/MXFP8 experts is still
        re-quantized: every method needs per-expert ``Linear`` modules, and
        calibration/Hessian collection must run the packed W4A16/W8A16 kernels
        instead of a dense bf16 copy. Defusion keeps the packed payload (views,
        no decode) so the layer's forwards stay on the kernel until the
        quantization processor asks for ``weight``, which materializes the fully
        dequantized bf16 source for packing.
        """

        module = super().pre_quantize(module, module_path=module_path)
        defuse_packed_experts(module)
        return module

    def after_quantize(self) -> None:
        """Declare the expert storage the saved checkpoint will really have."""

        super().after_quantize()
        declare_expert_storage(self.model)
        # The head stage is the last consumer of the per-batch banks; do not
        # retain calibration activations after the run.
        self._inkling2_loop_final_block_banks = {}
        self._inkling2_loop_final_block_bank_index = 0

    @classmethod
    def after_defuser_conversion(cls, model):
        """Validate the expert layout Defuser leaves behind.

        Packed (pre-quantized) experts stay packed here: they are defused per
        layer in :meth:`pre_quantize`, right after the lazy shell materializes
        the checkpoint payload and before the layer's modules are resolved.
        Dense experts must already be expanded into per-expert ``Linear``
        modules by Defuser.
        """

        config = getattr(model.config, "text_config", model.config)
        expected = int(config.num_hidden_layers)
        containers = [
            module for name, module in model.named_modules() if name.endswith(".mlp.experts")
        ]
        if len(containers) != expected:
            raise RuntimeError(
                "Inkling2Loop expert coverage mismatch: "
                f"converted={len(containers)} expected={expected}."
            )
        for experts in containers:
            if isinstance(experts, Inkling2LoopPackedExperts):
                for part in ("gate_up_weight", "gate_up_scales", "down_weight", "down_scales"):
                    if getattr(experts, part, None) is None:
                        raise RuntimeError(f"Packed Inkling2Loop experts lost `{part}` during conversion.")
                continue
            if getattr(experts, "gate_up_proj", None) is not None or getattr(experts, "down_proj", None) is not None:
                raise RuntimeError(
                    "Inkling2Loop dense routed experts were not defused into per-expert Linear modules. "
                    "Upgrade Defuser before quantizing."
                )
            for index in range(int(config.n_routed_experts)):
                expert = getattr(experts, str(index), None)
                if expert is None or not all(
                    hasattr(expert, part) for part in ("gate_proj", "up_proj", "down_proj")
                ):
                    raise RuntimeError(
                        f"Inkling2Loop expert {index} has no gate/up/down projections after defusion."
                    )
            # Defuser's generic forward applies routing after the down projection.
            # Bind the model's routed forward on this instance to preserve FP32 routing.
            experts.forward = experts._forward_routed

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
        # Capture starts a new batch set: give its carries fresh indices and
        # drop the previous run's final banks.
        self._inkling2_loop_capture_index = 0
        self._inkling2_loop_final_block_banks = {}
        self._inkling2_loop_final_block_bank_index = 0
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
            padding=True,
        )

    def lm_head_pre_quantize_generate_hook(
        self, inputs: List[List[torch.Tensor]]
    ) -> List[List[torch.Tensor]]:
        """Feed the head its real input: ``norm(final_input(block_states, h))``.

        The generic hook applies only the final norm to the last decoder layer's
        output. Inkling 2 first retrieves the final hidden state from the bank of
        block-boundary inputs with ``language_model.final_input``, so calibrating
        the head on ``norm(h)`` uses activations the model never feeds it.

        The per-batch bank is recorded by :meth:`prepare_layer_replay_kwargs`
        when the final decoder layer is replayed. Overriding this hook also makes
        :meth:`BaseQModel.last_block_objective_modules` return ``None`` so G2PTQ
        falls back to block MSE instead of scoring a different head path.
        """

        if not inputs:
            return inputs

        language_model = self.model.model.language_model
        banks = getattr(self, "_inkling2_loop_final_block_banks", None) or {}
        final_input = self.pre_quantize(language_model.final_input)
        resolved = self._resolved_pre_lm_head_norm_candidates()
        if not resolved:
            raise ValueError(
                f"{type(self).__name__}: cannot resolve `pre_lm_head_norm_module`."
            )
        norm = self.pre_quantize(resolved[0][0])
        try:
            for batch_index, element in enumerate(inputs):
                bank = banks.get(batch_index)
                if bank is None:
                    raise RuntimeError(
                        "Inkling2Loop lm_head calibration is missing the "
                        f"attention-residual bank for batch {batch_index}; replay "
                        "the final decoder layer before preparing the lm_head inputs."
                    )
                for index, tensor in enumerate(element):
                    if not torch.is_tensor(tensor):
                        continue
                    element[index] = norm(
                        final_input(
                            move_to(bank, device=tensor.device, dtype=tensor.dtype),
                            tensor,
                        )[0]
                    ).to(tensor.dtype)
        finally:
            self.post_quantize(final_input)
            self.post_quantize(norm)
        return inputs

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        # Route through the shared #4228/#4308 contract: pre-tokenized rows and
        # columnar mappings keep the base text path, and `concat_size` is
        # rejected for conversations. The local conversation loop bypassed it
        # and handed `[{"input_ids": ...}]` rows to `apply_chat_template`.
        return prepare_vlm_calibration_dataset(
            self, calibration_dataset, batch_size=batch_size, options=kwargs
        )

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
        carry = _BlockStatesCarry()
        carry.index = getattr(self, "_inkling2_loop_capture_index", 0)
        self._inkling2_loop_capture_index = carry.index + 1
        layer_input_kwargs[_BLOCK_STATES_CARRY_KEY] = carry
        return layer_input_kwargs

    def capture_looped_layer_invocation_inputs(self, module, args, kwargs):
        """Record the residual bank and loop messages passed positionally.

        Every decoder layer takes ``block_states`` as its second positional
        argument and loop-core passes 2..N take the q/k/v/mlp ``messages`` as
        the third. The looped full-forward capture replays each invocation, so
        it must keep the exact bank and messages of that invocation instead of
        rebuilding a single pass-1 chain.
        """

        del module, kwargs
        captured: Dict[str, Any] = {}
        if len(args) > 1 and torch.is_tensor(args[1]):
            captured["block_states"] = args[1].detach()
        if len(args) > 2 and torch.is_tensor(args[2]):
            captured["messages"] = args[2].detach()
        return captured

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        captured_block_states = additional_inputs.pop("block_states", None)
        captured_messages = additional_inputs.pop("messages", None)
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs

        if captured_block_states is not None:
            # The looped full-forward capture recorded this invocation's own
            # bank and messages, so replay is exact: post-loop layers see the
            # final-pass output and loop-core layers see every pass.
            self_attn = getattr(layer, "self_attn", None)
            layer_idx = getattr(self_attn, "layer_idx", None)
            if layer_idx == 0:
                # A new replay pass starts here; do not keep banks from the
                # previous processor or quantization run.
                self._inkling2_loop_final_block_banks = {}
                self._inkling2_loop_final_block_bank_index = 0
            additional_inputs["block_states"] = move_to(
                captured_block_states, device=target_device
            )
            if captured_messages is not None:
                additional_inputs["messages"] = move_to(
                    captured_messages, device=target_device
                )
            if layer_idx == len(self.model.model.language_model.layers) - 1:
                # ``final_input`` consumes the bank after the final decoder
                # layer appends its input. Record it here because the lm_head
                # stage runs after every layer replay.
                bank = captured_block_states
                if getattr(layer, "is_block_boundary", False):
                    bank = torch.cat(
                        [
                            bank,
                            move_to(layer_input[0].detach(), device=CPU).unsqueeze(-2),
                        ],
                        dim=-2,
                    )
                banks = getattr(self, "_inkling2_loop_final_block_banks", None)
                if banks is None:
                    banks = {}
                    self._inkling2_loop_final_block_banks = banks
                index = getattr(
                    self, "_inkling2_loop_final_block_bank_index", 0
                )
                banks[index] = bank
                self._inkling2_loop_final_block_bank_index = index + 1
            return additional_inputs

        carry = additional_inputs.pop(_BLOCK_STATES_CARRY_KEY, None)

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
        if layer_idx == 0:
            # A new replay pass starts here; do not keep banks from the previous
            # processor or quantization run.
            self._inkling2_loop_final_block_banks = {}
            self._inkling2_loop_final_block_bank_index = 0
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
        if layer_idx == len(self.model.model.language_model.layers) - 1:
            # ``final_input`` consumes the bank after the final decoder layer.
            # The lm_head stage runs once per batch set, so keep one bank per
            # captured batch.
            banks = getattr(self, "_inkling2_loop_final_block_banks", None)
            if banks is None:
                banks = {}
                self._inkling2_loop_final_block_banks = banks
            banks[carry.index] = carry.states
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
        mask_kwargs["past_key_values"] = additional_inputs.get("past_key_values")
        additional_inputs["attention_mask"] = mask_factory(**mask_kwargs)
        return additional_inputs


__all__ = ["Inkling2LoopQModel"]
