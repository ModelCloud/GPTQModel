# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""G2PTQ processor: refresh first- and second-order statistics per block.

Clean-room implementation of "Improving LLM Post-Training Quantization with
Generalized Gradient Compensation" (Liu et al., arXiv:2609.31009), Section 4
and Algorithms 1-2. The solver itself lives in
:mod:`gptqmodel.quantization.g2ptq`; this module owns the per-block refresh.

The paper's block objective (Eq. 10) compares two streams:

``h``
    the full-precision block output, produced from the *clean* calibration
    stream, i.e. the output of the pristine previous blocks applied to the
    original activations;
``h_hat``
    the output of the block being quantized, applied to the *noisy* stream
    that the live calibration cache carries.

Because the block weights are still float when the refresh runs, the two
streams differ only by the accumulated quantization error of the previous
blocks; that difference is what supplies a non-zero first-order term. Both
objectives of Algorithm 2 are then built from the same differentiable replay:

``ell_H``
    the Fisher proxy of Theorem 1/2, whose output-activation gradients build
    the grouped Hessian of Eq. 8/9;
``ell_G``
    the supervised objective of Eq. 10, whose weight gradient is the exact
    first-order term of Eq. 6/15.

Intermediate blocks use the block-wise MSE pair. The final Transformer block
uses the KL / NLL pair through the model's final norm and output head, and a
quantized output head uses the same pair directly. Both statistics are
installed on the ``G2PTQ`` task, which falls back to plain GPTQ when they are
absent (a module the refresh never reached, or a run where the refresh was
skipped).
"""

from __future__ import annotations

import inspect
import math
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..quantization.config import G2PTQConfig
from ..quantization.g2ptq import G2PTQ
from ..utils.attn_mask import normalize_seq_mask
from ..utils.logger import setup_logger
from .gptq_processor import GPTQProcessor
from .loop_processor import (
    CalibrationTypes,
    CheckpointSupport,
    ForwardConfig,
    ProcessorLifecycleConfig,
)


log = setup_logger()

#: Keywords the block replay must not forward: a calibration cache cannot
#: supply a consistent KV state, and passing one silently truncates the graph.
_REPLAY_KWARG_BLOCKLIST = frozenset({"past_key_values", "past_key_value"})

#: Replay keys the executor always supplies from the cache, not from the
#: captured per-batch kwargs.
_REPLAY_MANAGED_KWARGS = frozenset({"attention_mask", "position_ids", "use_cache"})

#: Largest number of logits elements the final-block KL/NLL pair materializes
#: at once. The pair runs on the default path, so a calibration batch times a
#: large vocabulary must not hold several float32 ``[tokens, vocab]`` tensors.
#: Rows are independent, so the row chunking is exact.
_LAST_BLOCK_LOGIT_CHUNK_ELEMENTS = 1 << 25


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    """Return the first tensor in a nested forward output, or ``None``."""

    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _move_value(value: Any, device: torch.device) -> Any:
    """Move tensors (and the common containers of them) onto ``device``."""

    if torch.is_tensor(value):
        return value if value.device == device else value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_value(item, device) for item in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    return value



def _fresh_value(value: Any) -> Any:
    """Detach and clone tensors so autograd may save them for backward."""

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _fresh_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_fresh_value(item) for item in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    return value


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


class _LastBlockHead:
    """The final norm and linear output head that turn a block output into logits."""

    def __init__(self, norm: Optional[torch.nn.Module], head: torch.nn.Module):
        weight = getattr(head, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise TypeError(
                f"G2PTQ cannot use `{type(head).__name__}` as a logits head."
            )
        if not weight.is_floating_point() or weight.device.type == "meta":
            raise TypeError(
                f"G2PTQ cannot use a {weight.dtype} `{type(head).__name__}` "
                "weight on "
                f"{weight.device} as a logits head."
            )

        from transformers.pytorch_utils import Conv1D

        if isinstance(head, Conv1D):
            # Conv1D stores ``[in, out]`` and computes ``x @ weight + bias``.
            self._weight = weight.t()
        elif isinstance(head, (torch.nn.Linear, torch.nn.Embedding)):
            self._weight = weight
        else:
            raise TypeError(
                f"G2PTQ cannot use `{type(head).__name__}` as a logits head."
            )

        bias = getattr(head, "bias", None)
        if bias is not None and (
            not torch.is_tensor(bias) or bias.device != weight.device
        ):
            raise TypeError("G2PTQ: the logits head bias is not on the weight device.")

        self.norm = norm
        self.head = head
        self.device = weight.device
        self.vocab = int(self._weight.shape[0])
        self._bias = bias
        self._dtype = weight.dtype
        self._release = None

    def close(self) -> None:
        """Release any live modules staged for this objective."""

        release = self._release
        self._release = None
        if release is not None:
            release()

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project ``hidden`` to logits without building an autograd graph."""

        if hidden.device != self.device:
            hidden = hidden.to(self.device)
        if hidden.dtype != self._dtype:
            hidden = hidden.to(self._dtype)
        with torch.no_grad():
            return F.linear(hidden, self._weight, self._bias)

    def input_grad(self, grad_logits: torch.Tensor) -> torch.Tensor:
        """Return ``d loss / d hidden`` for an upstream logits gradient."""

        weight = self._weight
        with torch.no_grad():
            return torch.matmul(grad_logits.to(weight.dtype), weight).to(
                dtype=torch.float32
            )


def _accepts_keyword(module: torch.nn.Module, keyword: str) -> bool:
    """Whether ``module.forward`` accepts ``keyword`` by name or ``**kwargs``."""

    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return True
    if keyword in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _relative_path(root: torch.nn.Module, target: torch.nn.Module) -> Optional[str]:
    """Return ``target``'s dotted path under ``root``, or ``None``."""

    for path, module in root.named_modules():
        if module is target:
            return path
    return None


class G2PTQProcessor(GPTQProcessor):
    """GPTQ capture plus one gradient refresh per block.

    The lifecycle differs from :class:`GPTQProcessor` in two ways, both forced
    by the refresh pass:

    * ``subset_early_stop`` is off, because the refresh needs the *complete*
      block output and the early-stop flag truncates the block forward;
    * ``capture_layer_context`` is on, so the stage hands the processor the
      clean reference output, the live block module, and the activations.
    """

    lifecycle = ProcessorLifecycleConfig(
        forward=ForwardConfig(
            required=True,
            replay_after_process=True,
            subset_early_stop=False,
            capture_layer_context=True,
        ),
        calibration=CalibrationTypes.EACH_MODULE_NEEDS_INPUT,
        checkpointing=CheckpointSupport.PRIMARY,
    )

    def __init__(
        self,
        tokenizer,
        qcfg,
        calibration,
        forward: Optional[ForwardConfig] = None,
        calculate_w_wq_diff: bool = False,
        gptq_model=None,
        model=None,
        **kwargs,
    ):
        """Initializes G2PTQ capture and the per-block refresh state."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            forward=forward,
            calculate_w_wq_diff=calculate_w_wq_diff,
        )
        del kwargs
        self.gptq_model = gptq_model
        self.model = model

        self._g2ptq_lock = threading.Lock()
        # layer_index -> {"inputs", "kwargs", "targets", "layer"}
        self._g2ptq_layer_state: Dict[int, Dict[str, Any]] = {}
        # layer_index -> untouched float block used for the differentiable replay
        self._g2ptq_pristine: Dict[int, torch.nn.Module] = {}
        # Modules whose capture hook fired for the subset in flight.
        self._g2ptq_subset_names: List[str] = []
        # True while the refresh pass runs: the GPTQ capture hooks must not
        # accumulate the refresh forward into the plain Hessian.
        self._g2ptq_suppress_capture = False
        self._g2ptq_is_lm_head = False
        self._g2ptq_unavailable: set = set()
        # layer_index -> (layer_index, _LastBlockHead); rebuilt after resume.
        self._g2ptq_last_block_head: Optional[Tuple[int, _LastBlockHead]] = None
        self._g2ptq_last_block_warned: set = set()
        # Clean calibration stream, advanced by the pristine replay of each
        # layer. ``None`` means "the live stream is still clean" (layer 0).
        self._g2ptq_clean_inputs: Optional[List[List[torch.Tensor]]] = None

    def name(self) -> str:
        return "G2PTQ"

    # ------------------------------------------------------------------
    # clean-stream plumbing
    # ------------------------------------------------------------------
    def receive_input_cache(self, input_cache):
        """Seed the clean stream from the first layer's calibration input."""

        super().receive_input_cache(input_cache)
        self._g2ptq_clean_inputs = input_cache.layer_inputs

    def continuation_state_dict(self):
        state = super().continuation_state_dict()
        state["g2ptq_clean_inputs"] = self._g2ptq_clean_inputs
        return state

    def load_continuation_state_dict(self, state):
        super().load_continuation_state_dict(state)
        self._g2ptq_clean_inputs = state.get("g2ptq_clean_inputs")

    def clean_group_layer_inputs(
        self,
        *,
        layer_index: int,
        layer_inputs: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Return the clean stream used to build the full-precision block output."""

        del layer_index
        return self._g2ptq_clean_inputs or layer_inputs

    def receive_clean_layer_inputs(
        self,
        *,
        layer_index: int,
        layer_inputs: List[List[torch.Tensor]],
    ) -> None:
        """Advance the clean stream to this layer's full-precision output."""

        del layer_index
        self._g2ptq_clean_inputs = layer_inputs

    # ------------------------------------------------------------------
    # stage hooks
    # ------------------------------------------------------------------
    def uses_grouped_optimization(self) -> bool:
        """The block objective needs a float reference block output."""

        return True

    def needs_pristine_layer_clone(self) -> bool:
        """Keep an untouched float block for the differentiable replay."""

        return True

    def receive_pristine_layer_module(
        self,
        *,
        layer_index: int,
        layer_module: torch.nn.Module,
    ) -> None:
        """Keep the stage's float snapshot of the block about to be quantized."""

        with self._g2ptq_lock:
            self._g2ptq_pristine[layer_index] = layer_module

    def begin_shared_input_capture(
        self,
        model,
        subset_names: List[str],
        is_lm_head_module: bool = False,
    ) -> Dict[str, str]:
        """Records the block role, then defers to GPTQ's (empty) dedup plan.

        G2PTQ modules never share a Hessian: each one owns a Fisher estimate
        built from its own output-activation gradients, so
        ``_hessian_capture_signature`` returns ``None`` for every task.
        """

        if model is not None:
            self.gptq_model = model
        self._g2ptq_is_lm_head = bool(is_lm_head_module)
        return super().begin_shared_input_capture(
            model, subset_names, is_lm_head_module=is_lm_head_module
        )

    def pre_process_fwd_hook(self, name: str):
        """Wraps GPTQ's capture hook and records the subset membership."""

        base_hook = super().pre_process_fwd_hook(name)

        def hook(module, inp, out):
            if self._g2ptq_suppress_capture:
                return
            with self._g2ptq_lock:
                if name not in self._g2ptq_subset_names:
                    self._g2ptq_subset_names.append(name)
            base_hook(module, inp, out)

        return hook

    def receive_layer_forward_context(
        self,
        *,
        layer_index: int,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        layer_outputs: List[List[torch.Tensor]],
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
        layer_module: Optional[torch.nn.Module] = None,
        position_ids: Optional[List[torch.Tensor]] = None,
        attention_masks: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Remembers the clean reference, then refreshes statistics per subset."""

        if subset_index is None:
            with self._g2ptq_lock:
                self._g2ptq_layer_state[layer_index] = {
                    "inputs": layer_inputs,
                    "kwargs": layer_input_kwargs,
                    "targets": layer_outputs,
                    "layer": layer_module,
                    "position_ids": position_ids,
                    "attention_masks": attention_masks,
                }
                self._g2ptq_subset_names = []
            return

        with self._g2ptq_lock:
            names = self._g2ptq_subset_names
            self._g2ptq_subset_names = []

        if names:
            try:
                self._refresh_block(layer_index=layer_index, names=names)
            except Exception as exc:  # noqa: BLE001 - the refresh must never abort a run
                self._report_unavailable(layer_index, exc)

        release_head = None
        if subset_total is not None and subset_index + 1 >= subset_total:
            with self._g2ptq_lock:
                self._g2ptq_layer_state.pop(layer_index, None)
                self._g2ptq_pristine.pop(layer_index, None)
                if (
                    self._g2ptq_last_block_head is not None
                    and self._g2ptq_last_block_head[0] == layer_index
                ):
                    release_head = self._g2ptq_last_block_head[1]
                    self._g2ptq_last_block_head = None
        if release_head is not None:
            release_head.close()

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------
    def _refresh_block(self, *, layer_index: int, names: List[str]) -> None:
        """Build and install one block's Fisher Hessians and weight gradients."""

        with self._g2ptq_lock:
            state = self._g2ptq_layer_state.get(layer_index)
            block = self._g2ptq_pristine.get(layer_index)
        if state is None:
            return
        if block is not None:
            block = self._differentiable_block(block, layer=state.get("layer"))
            with self._g2ptq_lock:
                self._g2ptq_pristine[layer_index] = block

        # Modules with different seeds need different Fisher draws, so the
        # replay groups by both objective and seed. The default run-level seed
        # keeps the common case at one replay per subset.
        layer = state.get("layer")
        last_block = self._is_last_transformer_block(layer_index) and not getattr(
            layer, "is_mtp", False
        )
        device = None
        if last_block:
            target = layer if layer is not None else block
            if target is not None:
                device = _module_device(target)

        tasks_by_objective: Dict[Tuple[bool, int], Dict[str, G2PTQ]] = {}
        for name in names:
            task = self.tasks.get(name)
            if isinstance(task, G2PTQ):
                config = task.qcfg.g2ptq
                use_kl = self._uses_kl_objective(
                    [name],
                    config=config,
                    layer_index=layer_index if last_block else None,
                )
                tasks_by_objective.setdefault((use_kl, config.seed), {})[name] = task

        head_path = None
        if last_block and any(use_kl for use_kl, _ in tasks_by_objective):
            if device is not None:
                head_path = self._last_block_head_path(layer_index, device)
            else:
                self._report_last_block_unavailable(layer_index, None)
            if head_path is None:
                # The final block has no supported logits path. Keep the
                # paper's block-wise MSE fallback rather than treating hidden
                # states as logits.
                for (use_kl, seed), tasks in list(tasks_by_objective.items()):
                    if use_kl:
                        tasks_by_objective.setdefault((False, seed), {}).update(tasks)
                        del tasks_by_objective[(use_kl, seed)]

        for (use_kl, seed), tasks in tasks_by_objective.items():
            try:
                statistics = self._collect_block_statistics(
                    state,
                    block,
                    tasks,
                    use_kl=use_kl,
                    seed=seed,
                    layer_index=layer_index,
                    head_path=head_path if use_kl else None,
                )
            except Exception as exc:  # noqa: BLE001 - replay can fail on unsupported blocks
                self._report_unavailable(layer_index, exc)
                continue

            for name, record in statistics.items():
                task = tasks[name]
                config = task.qcfg.g2ptq
                try:
                    task.set_block_statistics(
                        group_hessians=record["hessian"],
                        weight_gradient=record["gradient"],
                        activation_square=record["activation_square"],
                        trust_alpha=config.alpha_kl if use_kl else config.alpha_mse,
                    )
                except ValueError as exc:
                    self._report_unavailable(layer_index, exc)

    def _report_unavailable(self, layer_index: int, exc: BaseException) -> None:
        """Warn once per layer and let those tasks fall back to plain GPTQ."""

        with self._g2ptq_lock:
            if layer_index in self._g2ptq_unavailable:
                return
            self._g2ptq_unavailable.add(layer_index)
        log.warn(
            "G2PTQ: block %s could not refresh its gradient statistics (%s: %s); "
            "affected modules fall back to GPTQ compensation.",
            layer_index,
            type(exc).__name__,
            exc,
        )

    def _report_last_block_unavailable(
        self, layer_index: int, exc: Optional[BaseException]
    ) -> None:
        """Warn once when the final block cannot reach the model's logits."""

        with self._g2ptq_lock:
            if layer_index in self._g2ptq_last_block_warned:
                return
            self._g2ptq_last_block_warned.add(layer_index)
        if exc is None:
            log.warn(
                "G2PTQ: the final block %s has no supported final-norm + head "
                "path; using block-wise MSE for it.",
                layer_index,
            )
            return
        log.warn(
            "G2PTQ: the final block %s cannot compose its final norm and head "
            "(%s: %s); using block-wise MSE for it.",
            layer_index,
            type(exc).__name__,
            exc,
        )

    def _is_last_transformer_block(self, layer_index: Optional[int]) -> bool:
        """Whether ``layer_index`` is the final quantized Transformer block."""

        layer_count = getattr(self, "layer_count", None)
        return (
            not self._g2ptq_is_lm_head
            and isinstance(layer_index, int)
            and isinstance(layer_count, int)
            and layer_index == layer_count - 1
        )

    def _last_block_head_path(
        self, layer_index: int, device: torch.device
    ) -> Optional[_LastBlockHead]:
        """Build (or reuse) the final norm + head path for the last block."""

        with self._g2ptq_lock:
            cached = self._g2ptq_last_block_head
            if cached is not None and cached[0] == layer_index:
                return cached[1]
            self._g2ptq_last_block_head = None

        resolver = getattr(self.gptq_model, "last_block_objective_modules", None)
        if not callable(resolver):
            return None
        try:
            modules = resolver()
        except Exception as exc:  # noqa: BLE001 - an unsupported model falls back to MSE
            self._report_last_block_unavailable(layer_index, exc)
            return None
        if modules is None:
            self._report_last_block_unavailable(layer_index, None)
            return None

        norm, norm_name, head, head_name = modules
        pre_quantize = getattr(self.gptq_model, "pre_quantize", None)
        post_quantize = getattr(self.gptq_model, "post_quantize", None)
        staged: List[Tuple[torch.nn.Module, torch.device]] = []
        try:
            # Lazy checkpoints keep the head and final norm on META until their
            # own stage. Materialize them for this objective, then hand them
            # back through the normal post-quantize offload path.
            if norm is not None and callable(pre_quantize):
                original = _module_device(norm)
                norm = pre_quantize(norm, module_path=norm_name)
                staged.append((norm, original))
            if callable(pre_quantize):
                original = _module_device(head)
                head = pre_quantize(head, module_path=head_name)
                staged.append((head, original))
                head = self._materialize_tied_head(head, head_name, staged)
            if norm is not None:
                norm_copy = self._differentiable_block(
                    norm, layer=None, device=device
                )
            else:
                norm_copy = None
            path = _LastBlockHead(norm_copy, head)
        except Exception as exc:  # noqa: BLE001 - fall back instead of aborting the run
            self._release_staged(staged, post_quantize)
            self._report_last_block_unavailable(layer_index, exc)
            return None

        def release() -> None:
            self._release_staged(staged, post_quantize)

        path._release = release
        with self._g2ptq_lock:
            self._g2ptq_last_block_head = (layer_index, path)
        return path

    @staticmethod
    def _release_staged(
        staged: List[Tuple[torch.nn.Module, torch.device]],
        post_quantize,
    ) -> None:
        """Return staged modules to their pre-refresh placement."""

        for module, original in reversed(staged):
            if original.type == "meta":
                if not callable(post_quantize):
                    continue
                try:
                    post_quantize(module)
                except Exception:  # noqa: BLE001 - best-effort release
                    pass
                continue
            try:
                module.to(device=original)
            except Exception:  # noqa: BLE001 - best-effort release
                pass

    def _materialize_tied_head(
        self,
        head: torch.nn.Module,
        head_name: str,
        staged: List[Tuple[torch.nn.Module, torch.device]],
    ) -> torch.nn.Module:
        """Expose a tied head's storage by materializing its input embedding.

        A tied ``lm_head`` has no checkpoint tensor of its own, so the lazy
        loader leaves it on META until the shared input embedding is
        materialized. Only that case needs the extra staging: an untied head
        owns its checkpoint tensor and materializes directly.
        """

        weight = getattr(head, "weight", None)
        if torch.is_tensor(weight) and weight.device.type != "meta":
            return head
        pre_quantize = getattr(self.gptq_model, "pre_quantize", None)
        get_embeddings = getattr(self.gptq_model, "get_input_embeddings", None)
        get_name = getattr(self.gptq_model, "get_input_embeddings_name", None)
        if not callable(pre_quantize) or not callable(get_embeddings):
            return head
        embed = get_embeddings()
        if (
            embed is None
            or embed is head
            or getattr(embed, "weight", None) is not weight
        ):
            return head
        original = _module_device(embed)
        embed_name = get_name() if callable(get_name) else None
        embed = pre_quantize(embed, module_path=embed_name)
        staged.append((embed, original))
        return pre_quantize(head, module_path=head_name)

    def _uses_kl_objective(
        self,
        names: Iterable[str],
        *,
        config: Optional[G2PTQConfig] = None,
        layer_index: Optional[int] = None,
    ) -> bool:
        """Whether this subset uses Algorithm 2's KL/NLL pair.

        The paper applies the pair to the last Transformer block by composing
        the final norm and the language head. A quantized output head uses the
        pair directly because its output is already logits.
        """

        config = config if config is not None else self.qcfg.g2ptq
        if config is None or not config.kl_last_block:
            return False
        if self._g2ptq_is_lm_head:
            return True
        if layer_index is not None and self._is_last_transformer_block(layer_index):
            return True
        lm_head = getattr(self.gptq_model, "lm_head", None)
        if not isinstance(lm_head, str) or not lm_head:
            return False
        return any(name == lm_head or name.endswith("." + lm_head) for name in names)

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def _collect_block_statistics(
        self,
        state: Dict[str, Any],
        block: Optional[torch.nn.Module],
        tasks: Dict[str, G2PTQ],
        *,
        use_kl: bool,
        seed: int = 0,
        layer_index: int = 0,
        head_path: Optional[_LastBlockHead] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run the differentiable block replay and fold it into statistics."""

        layer = state.get("layer")
        target = block if block is not None else layer
        if target is None:
            raise RuntimeError("G2PTQ has no float block to replay for this layer.")
        device = _module_device(layer if layer is not None else target)
        block = self._differentiable_block(target, layer=layer, device=device)

        if getattr(layer, "reuse_kv", False) or getattr(block, "reuse_kv", False):
            raise RuntimeError(
                "G2PTQ cannot replay a block that consumes a previous layer's KV cache."
            )

        names = list(tasks)
        modules: Dict[str, torch.nn.Module] = {}
        for name in names:
            modules[name] = self._snapshot_module(
                block, layer, tasks[name], name=name
            )

        inputs = state.get("inputs") or []
        targets = state.get("targets") or []
        kwargs_list = state.get("kwargs") or []
        if not inputs or len(inputs) != len(targets):
            raise RuntimeError("G2PTQ requires matching input and target calibration batches.")

        position_ids = state.get("position_ids") or []
        attention_masks = state.get("attention_masks") or []
        # Repeat the same Fisher draws for each subset of this block. Derive
        # them from the configured seed and layer index, never from the
        # ambient process RNG state, so repeated runs match.
        draw_seed = (int(seed) + layer_index) % (1 << 64)
        generator = torch.Generator(device=device).manual_seed(draw_seed)

        hessians: Dict[str, Optional[torch.Tensor]] = dict.fromkeys(names)
        gradients: Dict[str, Optional[torch.Tensor]] = dict.fromkeys(names)
        activation_squares: Dict[str, Optional[torch.Tensor]] = dict.fromkeys(names)

        self._g2ptq_suppress_capture = True
        try:
            for batch_index, input_batch in enumerate(inputs):
                if not input_batch:
                    continue
                reference = _first_tensor(targets[batch_index])
                if reference is None:
                    continue
                moved_inputs = [_move_value(item, device) for item in input_batch]
                raw_mask = attention_masks[batch_index] if batch_index < len(attention_masks) else None
                attention_mask = _move_value(raw_mask, device)
                moved_kwargs = self._replay_kwargs(
                    block=block,
                    layer_input=moved_inputs,
                    captured=(kwargs_list[batch_index] if batch_index < len(kwargs_list) else {}),
                    attention_mask=attention_mask,
                    position_ids=(
                        position_ids[batch_index] if batch_index < len(position_ids) else None
                    ),
                    device=device,
                )

                seq_len = moved_inputs[0].shape[1] if moved_inputs[0].ndim >= 3 else None
                keep_mask = normalize_seq_mask(attention_mask, seq_len=seq_len)
                prepare_keep_mask = getattr(self.gptq_model, "prepare_layer_replay_keep_mask", None)
                if callable(prepare_keep_mask) and not self._g2ptq_is_lm_head:
                    keep_mask = prepare_keep_mask(
                        layer=block, layer_input=moved_inputs, additional_inputs=moved_kwargs,
                        keep_mask=keep_mask, target_device=device,
                    )
                keep_mask = self.calibration_keep_mask(
                    keep_mask, batch_index=batch_index, layer_input=moved_inputs,
                )

                with torch.inference_mode(False), torch.enable_grad():
                    # Calibration tensors were produced by inference-mode
                    # forwards; autograd cannot save them for backward.
                    live_inputs = [_fresh_value(item) for item in moved_inputs]
                    live_kwargs = {
                        key: _fresh_value(value) for key, value in moved_kwargs.items()
                    }
                    captured: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
                    handles = []
                    for name, module in modules.items():
                        handles.append(
                            module.register_forward_hook(
                                self._module_recorder(name, captured)
                            )
                        )
                    try:
                        output = block(*live_inputs, **live_kwargs)
                    finally:
                        for handle in handles:
                            handle.remove()

                    produced = _first_tensor(output)
                    if produced is None or produced.shape != reference.shape:
                        raise RuntimeError("G2PTQ block replay output does not match its reference.")
                    target = _fresh_value(reference).to(device=produced.device)
                    if keep_mask is not None:
                        if keep_mask.numel() != produced.numel() // produced.shape[-1]:
                            raise RuntimeError("G2PTQ replay mask does not match the block output.")
                        keep_rows = _fresh_value(keep_mask).flatten()
                        produced = produced.flatten(0, -2)[keep_rows]
                        target = target.flatten(0, -2)[keep_rows]
                    if produced.numel() == 0:
                        continue
                    if use_kl and head_path is not None:
                        loss_g, loss_h = self._last_block_objectives(
                            produced,
                            target,
                            head_path=head_path,
                            generator=generator,
                        )
                    else:
                        loss_g, loss_h = self._block_objectives(
                            produced, target, use_kl=use_kl, generator=generator,
                        )

                    weight_grads = torch.autograd.grad(
                        loss_g,
                        [modules[name].weight for name in names],
                        retain_graph=bool(captured),
                        allow_unused=True,
                        materialize_grads=True,
                    )
                    activation_grads = ()
                    if captured:
                        activation_grads = torch.autograd.grad(
                            loss_h,
                            [output for _, _, output in captured],
                            allow_unused=True,
                            materialize_grads=True,
                        )

                for name, grad in zip(names, weight_grads):
                    grad = self._orientation(modules[name], grad).detach().to(
                        dtype=torch.float32
                    )
                    gradients[name] = (
                        grad if gradients[name] is None else gradients[name] + grad
                    )

                for (name, activation, _), grad in zip(captured, activation_grads):
                    rows = self._module_rows(activation, grad)
                    if rows is None:
                        continue
                    activations, saliency = rows
                    if keep_mask is not None and activations.shape[0] == keep_mask.numel():
                        activations = activations[keep_mask.flatten()]
                        saliency = saliency[keep_mask.flatten()]
                    if activations.shape[0] == 0:
                        continue
                    hessians[name] = self._accumulate_grouped_hessian(
                        hessians[name],
                        activations,
                        saliency,
                        num_groups=tasks[name].qcfg.g2ptq.num_groups,
                    )
                    energy = activations.float().square().sum(dim=0)
                    activation_squares[name] = (
                        energy
                        if activation_squares[name] is None
                        else activation_squares[name] + energy
                    )
        finally:
            self._g2ptq_suppress_capture = False

        statistics = {}
        for name in names:
            hessian = hessians[name]
            gradient = gradients[name]
            if hessian is None or gradient is None:
                continue
            statistics[name] = {
                "hessian": hessian,
                "gradient": gradient,
                "activation_square": activation_squares[name],
            }
        return statistics

    @classmethod
    def _differentiable_block(
        cls,
        block: torch.nn.Module,
        *,
        layer: Optional[torch.nn.Module],
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """Return a block snapshot that autograd can differentiate through.

        Two properties of the live pipeline block autograd: the looper's
        ``HookedLinear`` forward runs under ``torch.inference_mode``, and model
        loading leaves parameters as inference tensors. Both are repaired on
        the snapshot we own, never on the live modules. The repair runs inside
        ``torch.inference_mode(False)`` because the whole quantization stage is
        itself inside inference mode, and anything allocated there is again an
        inference tensor that autograd cannot save for backward.

        Ownership matters twice: the live layer is never mutated, because it is
        still being quantized while the refresh runs, and a snapshot that
        already carries grad leaves is reused instead of rebuilt, so a layer
        with several subsets pays for one float copy rather than one per
        subset.
        """

        import copy

        target = device if device is not None else _module_device(block)
        # ``block is layer`` means the caller only has the live module; every
        # other caller passes the stage's pristine clone, which we own.
        owns_block = block is not layer
        hooked = any(hasattr(module, "forward_hook") for module in block.modules())
        frozen = any(
            parameter.is_inference() or (
                (parameter.is_floating_point() or parameter.is_complex()) and not parameter.requires_grad
            )
            for parameter in block.parameters()
        )
        inference_buffers = any(buffer.is_inference() for buffer in block.buffers())
        if owns_block and not hooked and not frozen and not inference_buffers and _module_device(block) == target:
            return block

        with torch.inference_mode(False):
            snapshot = block if owns_block else copy.deepcopy(block)
            if hooked:
                from ..nn_modules.hooked_linear import HookedLinear, restore_linear_from_hooked

                for name, module in list(snapshot.named_modules()):
                    if not hasattr(module, "forward_hook"):
                        continue
                    if not isinstance(module, HookedLinear):
                        raise RuntimeError(
                            "G2PTQ cannot build a differentiable copy of "
                            f"`{type(module).__name__}`."
                        )
                    restored = restore_linear_from_hooked(module)
                    if not name:
                        # The output head is itself a ``HookedLinear``; rebuild
                        # the root instead of replacing a child.
                        snapshot = restored
                        continue
                    parent = snapshot.get_submodule(name.rpartition(".")[0])
                    setattr(parent, name.rpartition(".")[2], restored)
            if _module_device(snapshot) != target:
                snapshot = snapshot.to(target)
            cls._detach_for_autograd(snapshot)
        return snapshot

    @staticmethod
    def _detach_for_autograd(snapshot: torch.nn.Module) -> None:
        """Replace inference tensors and frozen leaves with fresh grad leaves.

        The parameter objects are rebuilt rather than having ``.data``
        assigned: assigning data leaves the parameter itself flagged as an
        inference tensor, and autograd then refuses to save it for backward.
        """

        replacements = {}
        for module in snapshot.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None:
                    continue
                differentiable = parameter.is_floating_point() or parameter.is_complex()
                if parameter.is_inference() or (differentiable and not parameter.requires_grad):
                    # All registrations of a tied weight must use the same leaf.
                    key = id(parameter)
                    if key not in replacements:
                        replacements[key] = torch.nn.Parameter(
                            parameter.detach().clone(), requires_grad=differentiable
                        )
                    module._parameters[name] = replacements[key]
            for name, buffer in list(module._buffers.items()):
                if buffer is None:
                    continue
                if buffer.is_inference():
                    module._buffers[name] = buffer.detach().clone()


    @staticmethod
    def _snapshot_module(
        block: torch.nn.Module,
        layer: Optional[torch.nn.Module],
        task: G2PTQ,
        *,
        name: str,
    ) -> torch.nn.Module:
        """Find the block-snapshot module that corresponds to one GPTQ task.

        The task holds the module instance it was built from, which is not
        necessarily the instance that ended up in the hooked block: the looper
        replaces projections while a layer is prepared, and a resumed run can
        carry tasks across that swap. The task's own relative name is stable
        across both, so prefer identity and fall back to the recorded name.
        """

        wrapped = getattr(task.module, "module", task.module)
        if layer is not None:
            path = _relative_path(layer, wrapped)
            if path is not None:
                return block.get_submodule(path)
            if block is layer and getattr(block, "weight", None) is not None:
                return block

        named = getattr(task, "_named_module", None)
        candidates = []
        if named is not None:
            candidates.append(getattr(named, "name", None))
            full_name = getattr(named, "full_name", None)
            if isinstance(full_name, str) and full_name:
                candidates.append(full_name.rsplit(".", 1)[-1])
        for candidate in candidates:
            if not candidate:
                continue
            try:
                resolved = block.get_submodule(candidate)
            except AttributeError:
                continue
            if resolved is not None:
                return resolved

        raise RuntimeError(
            f"G2PTQ could not locate `{name}` inside its block snapshot "
            f"({type(block).__name__})."
        )

    def _replay_kwargs(
        self,
        *,
        block: torch.nn.Module,
        layer_input: List[torch.Tensor],
        captured: Dict[str, Any],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Rebuild the per-batch forward kwargs the executor would pass."""

        additional: Dict[str, Any] = {
            "attention_mask": attention_mask if getattr(self.gptq_model, "support_batch_quantize", True) else None,
            "use_cache": False,
        }
        for key, value in (captured or {}).items():
            if key in _REPLAY_KWARG_BLOCKLIST or key in _REPLAY_MANAGED_KWARGS:
                continue
            additional[key] = _move_value(value, device)
        if position_ids is not None:
            additional["position_ids"] = _move_value(position_ids, device)

        prepare = getattr(self.gptq_model, "prepare_layer_replay_kwargs", None)
        if callable(prepare) and not self._g2ptq_is_lm_head:
            additional = prepare(
                layer=block,
                layer_input=layer_input,
                additional_inputs=additional,
                target_device=device,
            )
        # Model hooks consume private capture metadata before the forward.
        return {key: value for key, value in additional.items() if _accepts_keyword(block, key)}

    @staticmethod
    def _orientation(module: torch.nn.Module, grad: torch.Tensor) -> torch.Tensor:
        """Match the row/column orientation ``GPTQ`` uses for this module."""

        import transformers

        if isinstance(module, torch.nn.Embedding):
            return grad.t()
        if isinstance(module, transformers.pytorch_utils.Conv1D):
            return grad.t()
        return grad

    @staticmethod
    def _module_recorder(
        name: str,
        captured: List[Tuple[str, torch.Tensor, torch.Tensor]],
    ):
        """Record one module's input activation and differentiable output."""

        def record(module, inp, out):
            del module
            output = _first_tensor(out)
            if output is None or not output.requires_grad:
                return
            activation = _first_tensor(inp)
            if activation is None or not torch.is_tensor(activation):
                return
            captured.append((name, activation, output))

        return record

    @staticmethod
    def _module_rows(
        activation: torch.Tensor,
        output_grad: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Flatten one module's activation rows and squared output gradients.

        Eq. 8 needs, per output channel ``j`` and token ``i``,
        ``(d ell / d Z_ji)^2``; Eq. 9 then averages those channel weights
        inside each output-channel group.
        """

        if not torch.is_tensor(activation) or not torch.is_tensor(output_grad):
            return None
        if activation.ndim < 2 or output_grad.ndim < 2:
            return None
        rows = activation.reshape(-1, activation.shape[-1])
        grad_rows = output_grad.reshape(-1, output_grad.shape[-1])
        if rows.shape[0] != grad_rows.shape[0]:
            return None
        return rows.detach(), grad_rows.detach().float().square()

    @staticmethod
    def _accumulate_grouped_hessian(
        hessian: Optional[torch.Tensor],
        activations: torch.Tensor,
        saliency: torch.Tensor,
        *,
        num_groups: int,
    ) -> torch.Tensor:
        """Add ``sum_i s[g, i] x_i x_i^T`` for every output-channel group."""

        columns = activations.shape[1]
        groups = int(num_groups)
        if groups < 1 or saliency.shape[1] % groups != 0:
            groups = 1
        channels = saliency.shape[1] // groups
        if hessian is None:
            hessian = torch.zeros(
                (groups, columns, columns),
                dtype=torch.float32,
                device=activations.device,
            )
        saliency = saliency.to(device=activations.device, dtype=torch.float32)
        activations = activations.to(dtype=torch.float32)
        for group in range(groups):
            start = group * channels
            weights = saliency[:, start:start + channels].mean(dim=1)
            if not torch.any(weights):
                continue
            weighted = activations * weights.unsqueeze(1)
            hessian[group] += weighted.transpose(0, 1) @ activations
        return hessian

    @staticmethod
    def _last_block_objectives(
        quantized: torch.Tensor,
        target: torch.Tensor,
        *,
        head_path: _LastBlockHead,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return gradient surrogates for the composed last-block KL/NLL pair.

        Algorithm 2 lines 12-13 score the final block through the model's
        predictive distribution. The head is applied without autograd so the
        large vocabulary projection is never saved for backward; the returned
        scalars are linearized at the current logits, so their derivative with
        respect to the block output equals the derivative of the KL/NLL pair.
        """

        quantized = quantized.reshape(-1, quantized.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        if head_path.norm is not None:
            hidden = head_path.norm(quantized)
            with torch.no_grad():
                reference_hidden = head_path.norm(target)
        else:
            hidden = quantized
            reference_hidden = target

        rows = max(
            1, _LAST_BLOCK_LOGIT_CHUNK_ELEMENTS // max(int(head_path.vocab), 1)
        )
        loss_g = torch.zeros((), device=hidden.device, dtype=torch.float32)
        loss_h = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], rows):
            hidden_chunk = hidden[start:start + rows]
            reference_chunk = reference_hidden[start:start + rows]
            logits = head_path.logits(hidden_chunk.detach()).float()
            reference_logits = head_path.logits(reference_chunk).float()
            log_prob = F.log_softmax(logits, dim=-1)
            reference_prob = F.softmax(reference_logits, dim=-1)

            sample_generator = generator
            if generator is not None and generator.device != reference_prob.device:
                sample_generator = torch.Generator(
                    device=reference_prob.device
                ).manual_seed(generator.initial_seed())
            labels = torch.multinomial(
                reference_prob, num_samples=1, generator=sample_generator
            ).squeeze(1)

            quant_prob = log_prob.exp()
            grad_g = quant_prob - reference_prob
            grad_h = quant_prob.clone()
            grad_h[torch.arange(labels.numel(), device=grad_h.device), labels] -= 1.0
            del quant_prob

            grad_hidden_g = head_path.input_grad(grad_g).to(hidden.device)
            grad_hidden_h = head_path.input_grad(grad_h).to(hidden.device)
            loss_g = loss_g + (grad_hidden_g.detach() * hidden_chunk).sum()
            loss_h = loss_h + (grad_hidden_h.detach() * hidden_chunk).sum()
        return loss_g, loss_h

    @staticmethod
    def _block_objectives(
        quantized: torch.Tensor,
        target: torch.Tensor,
        *,
        use_kl: bool,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(ell_G, ell_H)`` for one block replay batch."""

        if quantized.dtype in (torch.float16, torch.bfloat16):
            quantized = quantized.float()
        target = target.to(dtype=quantized.dtype)
        if use_kl:
            logits = quantized.reshape(-1, quantized.shape[-1])
            reference = target.reshape(-1, target.shape[-1])
            log_prob = F.log_softmax(logits, dim=-1)
            reference_prob = F.softmax(reference, dim=-1)
            loss_g = F.kl_div(log_prob, reference_prob, reduction="sum")
            with torch.no_grad():
                labels = torch.multinomial(
                    reference_prob, num_samples=1, generator=generator
                ).squeeze(1)
            loss_h = F.nll_loss(log_prob, labels, reduction="sum")
            return loss_g, loss_h

        hidden = int(quantized.shape[-1])
        if hidden <= 0:
            raise RuntimeError("G2PTQ block replay produced an empty hidden axis.")
        difference = quantized - target
        loss_g = 0.5 * difference.square().sum() / hidden
        # Theorem 1: the Fisher proxy adds Gaussian noise to the model output
        # and keeps it as a detached target, so the residual is exactly the
        # injected noise.
        noise = torch.randn(
            quantized.shape, device=quantized.device, dtype=quantized.dtype, generator=generator
        ) * math.sqrt(hidden)
        residual = quantized - quantized.detach() - noise
        loss_h = 0.5 * residual.square().sum() / hidden
        return loss_g, loss_h


__all__ = ["G2PTQProcessor"]
