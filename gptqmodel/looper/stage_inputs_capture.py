# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Isolated stage for capturing calibration inputs prior to quantization."""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

import torch

from .. import DEVICE_THREAD_POOL
from ..looper.input_cache import InputCache
from ..nn_modules.hooked_linear import StopForward
from ..quantization.config import QuantizeEmbed
from ..utils.ctx import ctx
from ..utils.device import get_device
from ..utils.inspect import bound_forward_argument
from ..utils.linalg_warmup import await_torch_linear_prewarm
from ..utils.logger import setup_logger
from ..utils.looper_helpers import device_ctx, normalize_device_like, select_forward_devices
from ..utils.model import get_module, get_module_by_name_prefix, move_to, nested_move_to
from ..utils.torch import CPU, META


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .module_looper import ModuleLooper


_VISION_INPUT_KEYS = frozenset(("pixel_values", "images"))


def _has_vision_inputs(example: Dict[str, Any]) -> bool:
    return any(key in example for key in _VISION_INPUT_KEYS)


def _forward_dtype(model: torch.nn.Module) -> Optional[torch.dtype]:
    """Best-effort dtype the capture forward will run in, for the `linear` prewarm join."""
    try:
        return next(model.parameters()).dtype
    except (StopIteration, AttributeError, TypeError):
        return None


def _dispatch_execution_device(module: torch.nn.Module) -> Optional[torch.device]:
    """Execution device of a resident accelerate dispatch hook on ``module``.

    Input capture runs through the loaded model, so a module owned by an
    accelerate hook must stay on that hook's execution device. Moving its
    weight to the capture device while the hook moves the inputs to the
    execution device splits the forward across two devices (#6132). Offload
    hooks are left alone: their weight map owns the placement.
    """

    from accelerate.hooks import AlignDevicesHook, SequentialHook

    pending = [getattr(module, "_hf_hook", None)]
    while pending:
        hook = pending.pop()
        if hook is None:
            continue
        if isinstance(hook, SequentialHook):
            pending.extend(hook.hooks)
            continue
        if isinstance(hook, AlignDevicesHook) and not hook.offload:
            execution_device = hook.execution_device
            if execution_device is not None:
                try:
                    execution_device = torch.device(execution_device)
                except (TypeError, RuntimeError):
                    continue
                if execution_device.type != "meta":
                    return execution_device
    return None


def _branch_first_layer_indices(layer_names: Optional[Sequence[str]]) -> List[int]:
    """Return the flat index of each concurrent module-tree branch's first layer.

    ``get_layers_with_prefixes`` concatenates each tree variant's layers, so a
    prefix change marks a branch boundary.  A single-stack model returns one
    index and keeps the historical early-stop capture path.
    """

    if not layer_names:
        return []

    indices: List[int] = []
    previous_prefix: Optional[str] = None
    for index, name in enumerate(layer_names):
        if not name:
            continue
        prefix = name.rsplit(".", 1)[0] if "." in name else name
        if prefix != previous_prefix:
            indices.append(index)
            previous_prefix = prefix
    return indices


class StageInputsCapture:
    """Capture layer inputs so processors can reuse cached activations."""

    def __init__(self, looper: ModuleLooper, logger=None) -> None:
        """Binds the capture stage to a looper instance and logger."""

        self.looper = looper
        self.gptq_model = looper.gptq_model
        self.logger = logger or setup_logger()

    def _materialize_modules_with_direct_meta_tensors(self, device: torch.device) -> None:
        get_modules = getattr(self.gptq_model, "get_modules_with_direct_meta_tensors", None)
        if not callable(get_modules):
            return
        for module_name in get_modules(self.gptq_model.model):
            module = get_module(self.gptq_model.model, module_name)
            if isinstance(module, torch.nn.Module):
                self.gptq_model.shell_direct_meta_materialize(
                    target_submodule=module,
                    device=device,
                    module_path=module_name,
                )

    def _resolve_forward_device(
        self,
        example: Dict[str, Any],
        fallback: torch.device,
    ) -> torch.device:
        """Resolve where model inputs must live for the pre-layer forward."""

        if not torch.is_tensor(example.get("input_ids")):
            return fallback

        try:
            embedding = self.gptq_model.get_input_embeddings()
        except Exception:
            return fallback

        if not isinstance(embedding, torch.nn.Module):
            return fallback

        embedding_device = get_device(embedding)
        return fallback if embedding_device == META else embedding_device

    def _first_layer_has_deferred_floatx_source(
        self,
        layer: torch.nn.Module,
        *,
        module_path: Optional[str] = None,
    ) -> bool:
        """Whether input capture must not materialize a packed floatx layer.

        The first decoder layer is only used as a pre-forward hook during input
        capture: the hook raises before that layer executes.  Materializing its
        packed leaf tensors first bypasses the module-local decoder and fails
        for layouts such as NVFP4, whose storage shape deliberately differs
        from the dense ``nn.Linear`` shell shape.
        """

        active_decoder = getattr(self.gptq_model, "_active_auto_module_decoder_config", None)
        if not callable(active_decoder) or active_decoder() is None:
            return False

        turtle_model = getattr(self.gptq_model, "turtle_model", None)
        get_tensors = getattr(turtle_model, "checkpoint_tensors_for_submodule", None)
        decoder_format = getattr(self.gptq_model, "_decoder_weight_format", None)
        if not callable(get_tensors) or not callable(decoder_format):
            return False

        for submodule_name, submodule in layer.named_modules():
            if not hasattr(submodule, "weight"):
                continue
            checkpoint_tensors = get_tensors(
                target_model=self.gptq_model.model,
                target_submodule=submodule,
                recurse=False,
            )
            weight = checkpoint_tensors.get("weight") if isinstance(checkpoint_tensors, dict) else None
            module_name = None
            if module_path is not None:
                module_name = ".".join(part for part in (module_path, submodule_name) if part)
            if isinstance(weight, torch.Tensor) and decoder_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
                module_name=module_name,
            ) is not None:
                return True
        return False

    def cache_inputs(
        self,
        layers: Sequence[torch.nn.Module],
        calibration_data: Iterable[Dict[str, torch.Tensor]],
        use_cache: bool,
        embed_quant_mode: Optional[QuantizeEmbed] = None,
        layer_names: Optional[List[str]] = None,
    ) -> InputCache:
        """Runs a short forward over calibration data and caches first-layer inputs."""

        src_inputs: List[List[torch.Tensor]] = []
        layer_inputs: List[List[torch.Tensor]] = []
        attention_masks: List[torch.Tensor | None] = []
        position_ids: List[torch.Tensor] = []
        layer_input_kwargs: List[Dict[str, Any]] = []
        branch_first_indices = _branch_first_layer_indices(layer_names)
        multi_branch = len(branch_first_indices) > 1

        timer = getattr(self.gptq_model, "quant_region_timer", None)
        layer_label = None
        if layers:
            first_layer = layers[0]
            # The caller-supplied dotted path is the single source of truth for
            # materializing the first layer from a lazy checkpoint source. An
            # explicit `full_name` attribute is a legacy fallback; warn if the two
            # disagree so the override is not silent.
            full_name = getattr(first_layer, "full_name", None)
            module_path = None
            if layer_names and layer_names[0]:
                module_path = layer_names[0]
                if full_name and full_name != module_path:
                    self.logger.warn(
                        f"cache_inputs: using caller-supplied layer_name {module_path!r} "
                        f"instead of layer.full_name {full_name!r} for materialization"
                    )
            elif full_name:
                module_path = full_name
            if not module_path:
                # Fallback: discover the dotted path from the model tree when the
                # caller does not supply names.
                for name, mod in self.gptq_model.model.named_modules():
                    if mod is first_layer:
                        module_path = name
                        break

            # Keep the class-name fallback as a display label only; do not pass it
            # as a checkpoint module path because it is not a valid dotted prefix.
            layer_label = module_path or full_name or getattr(
                getattr(first_layer, "__class__", None), "__name__", None
            ) or type(first_layer).__name__
            capture_source = f"cache_inputs:{layer_label}"
        else:
            module_path = None
            capture_source = "cache_inputs"
        start_time = time.perf_counter() if timer else None

        try:
            calibration_batches = len(calibration_data)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            calibration_batches = None

        if calibration_batches is None:
            self.logger.info("ModuleLooper: capturing layer inputs (batch count unknown)")
        else:
            self.logger.info(
                "ModuleLooper: capturing layer inputs from %s calibration batches",
                calibration_batches,
            )

        # materialize / move.to CPU for initial input capture and for first layer to minimize VRAM usage, inputs will be stored on CPU
        # and to mimic behavior of offload_to_disk=False for offload_to_disk=True
        # Use calibration_data_device to specify device for calibration data (or "balanced" for round-robin across GPUs)
        def _materialize_capture_layer(index: int, path: Optional[str]) -> torch.nn.Module:
            layer = layers[index]
            layer_device = _dispatch_execution_device(layer) or CPU
            if index == 0 and self._first_layer_has_deferred_floatx_source(layer, module_path=path):
                self.logger.info(
                    "Floatx input capture: keeping first decoder layer lazy; its pre-hook runs before packed weights are used."
                )
                return layer
            prepare_capture_layer = getattr(self.gptq_model, "prepare_input_capture_layer", None)
            if callable(prepare_capture_layer) and index == 0:
                return prepare_capture_layer(layer, module_path=path, device=layer_device)
            return self.gptq_model.shell_module_materialize(
                target_submodule=layer, module_path=path, device=layer_device,
            )

        if multi_branch:
            # Reaching a later sibling stack requires running the stacks before
            # it.  The single-stack early-stop optimization cannot do that, so
            # materialize every decoder layer for the full capture forward.
            self.logger.info(
                "ModuleLooper: capturing %s concurrent module-tree branches; "
                "materializing all decoder layers for the calibration forward.",
                len(branch_first_indices),
            )
            for branch_index in range(len(layers)):
                branch_path = (
                    layer_names[branch_index]
                    if layer_names and branch_index < len(layer_names)
                    else None
                )
                layers[branch_index] = _materialize_capture_layer(branch_index, branch_path)
        else:
            layers[0] = _materialize_capture_layer(0, module_path)
        cur_layer_device = CPU

        # Use calibration_data_device if specified, otherwise use cur_layer_device
        calib_device_cfg = self.gptq_model.quantize_config.calibration_data_device

        # Prepare devices for balanced mode
        balanced_devices: List[torch.device] = []
        balanced_mode = False
        if calib_device_cfg == "balanced":
            balanced_mode = True
            # Get all available devices of same type as the quantization device
            all_devices = select_forward_devices(self.gptq_model.quantize_config.device)
            # Apply compute_device_filter if set
            compute_device_filter = self.gptq_model.quantize_config.compute_device_filter
            if compute_device_filter is not None:
                balanced_devices = compute_device_filter(all_devices)
                if not balanced_devices:
                    balanced_devices = all_devices
            else:
                balanced_devices = all_devices
            data_device = balanced_devices[0] if balanced_devices else cur_layer_device
        elif calib_device_cfg is not None:
            data_device = calib_device_cfg
        else:
            data_device = cur_layer_device

        # Round-robin counter for balanced mode
        balanced_rr_counter = [0]  # Use list to allow modification in nested function

        cache_forward_pb = None
        processed_rows = 0
        cache_total_batches = None
        if calibration_batches is not None and calibration_batches > 0:
            cache_total_batches = int(calibration_batches)
            cache_forward_pb = (
                self.logger.pb(range(max(cache_total_batches, 1)))
                .manual()
                .set(show_left_steps=False)
            )
            cache_title = (
                f"Forward cached inputs (Pre {layer_label})"
                if layer_label
                else "Forward cached inputs"
            )
            cache_forward_pb.title(cache_title).subtitle(
                f"Batch 0/{cache_total_batches}"
            ).draw()

        def _capture_layer_inputs(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: Dict[str, Any],
            batch_device: torch.device,
        ):
            layer_input = self.gptq_model.capture_first_layer_positional_inputs(
                args=args,
                kwargs=kwargs,
                batch_device=batch_device,
            )
            # `transformers` 5 hands some blocks their causal mask positionally:
            # `GPT2Model.forward` calls every block as
            # `block(hidden_states, past_key_values, causal_mask, ...)`. Reading
            # `kwargs` alone therefore captured `attention_mask=None` for GPT-2,
            # and replay ran every layer without causality under eager attention
            # and without padding under SDPA. Bind the block's own signature and
            # read the parameter by name; it is the same object the model passed.
            attention_mask = bound_forward_argument(
                module, args, kwargs, "attention_mask"
            )
            if torch.is_tensor(attention_mask):
                attention_mask = move_to(attention_mask, device=batch_device)
            pos_ids = bound_forward_argument(module, args, kwargs, "position_ids")
            if torch.is_tensor(pos_ids):
                pos_ids = move_to(pos_ids, device=batch_device)
            else:
                pos_ids = None
            one_kwargs: Dict[str, Any] = {}
            for (k, v) in kwargs.items():
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=batch_device)
            one_kwargs = self.gptq_model.capture_first_layer_input_kwargs(
                args=args,
                kwargs=kwargs,
                batch_device=batch_device,
                layer_input_kwargs=one_kwargs,
            )
            # `transformers` 5 builds one mask (and, for Gemma3, one rotary tuple)
            # per layer *type*. The hook fires for the first decoder layer of a
            # branch, so capture the remaining declared types here; replay
            # selects the entry that belongs to the layer it is running.
            capture_layer_type_kwargs = getattr(
                self.gptq_model, "capture_layer_type_replay_kwargs", None
            )
            if callable(capture_layer_type_kwargs):
                one_kwargs = capture_layer_type_kwargs(
                    args=args,
                    kwargs=kwargs,
                    batch_device=batch_device,
                    layer_input_kwargs=one_kwargs,
                )
            return layer_input, attention_mask, pos_ids, one_kwargs

        def _batch_device() -> torch.device:
            # Select device for this batch (round-robin for balanced mode)
            if balanced_mode and balanced_devices:
                device = balanced_devices[balanced_rr_counter[0] % len(balanced_devices)]
                balanced_rr_counter[0] += 1
                return device
            return data_device

        def store_input_hook(module, args, kwargs):
            """Captures the incoming batch for the first layer and aborts the forward."""

            layer_input, attention_mask, pos_ids, one_kwargs = _capture_layer_inputs(
                module, args, kwargs, _batch_device(),
            )
            layer_inputs.append(layer_input)
            attention_masks.append(attention_mask)
            if pos_ids is not None:
                position_ids.append(pos_ids)
            layer_input_kwargs.append(one_kwargs)

            # In normal repeating layer/subset early stop happens on the last module forward
            # but the first model input embedding call we use a simple model register forward hook
            # and wait for the first instance this callback is called
            raise StopForward("Forwarding stopped")

        # Parameters attached to the shell root must be ready before embedding forward.
        self._materialize_modules_with_direct_meta_tensors(cur_layer_device)

        def _resolve_module_name(mod: torch.nn.Module) -> Optional[str]:
            for name, m in self.gptq_model.model.named_modules():
                if m is mod:
                    return name
            return None

        get_input_embeddings = getattr(self.gptq_model, "get_input_embeddings", None)
        input_embeddings = get_input_embeddings() if callable(get_input_embeddings) else None
        get_input_embeddings_name = getattr(self.gptq_model, "get_input_embeddings_name", None)
        input_embeddings_name = get_input_embeddings_name() if callable(get_input_embeddings_name) else None
        ori_outside_layer_module_devices: Dict[str, torch.device] = {}
        for module_name in self.gptq_model.get_base_modules(self.gptq_model.model):
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue
            if (
                embed_quant_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
                and module_name == input_embeddings_name
            ):
                continue

            resolved_name = _resolve_module_name(module)
            m_device = get_device(module)
            module_device = _dispatch_execution_device(module) or cur_layer_device
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device
            self.gptq_model.shell_module_materialize(
                target_submodule=module,
                device=module_device,
                module_path=resolved_name,
            )

        branch_captures: Dict[int, Dict[str, List[Any]]] = {}
        batch_captured: set[int] = set()
        handles = []

        if multi_branch:
            for branch_index in branch_first_indices:
                branch_captures[branch_index] = {
                    "layer_inputs": [],
                    "layer_input_kwargs": [],
                    "position_ids": [],
                    "attention_masks": [],
                }

            def _make_branch_hook(branch_index: int):
                def hook(module, args, kwargs):
                    if branch_index in batch_captured:
                        # A recurrent branch may invoke its first layer more
                        # than once per sample; only the first invocation is
                        # the branch-boundary input.
                        return
                    batch_captured.add(branch_index)
                    capture = branch_captures[branch_index]
                    layer_input, attention_mask, pos_ids, one_kwargs = _capture_layer_inputs(
                        module, args, kwargs, _batch_device(),
                    )
                    capture["layer_inputs"].append(layer_input)
                    capture["attention_masks"].append(attention_mask)
                    if pos_ids is not None:
                        capture["position_ids"].append(pos_ids)
                    capture["layer_input_kwargs"].append(one_kwargs)
                    if len(batch_captured) == len(branch_first_indices):
                        raise StopForward("Forwarding stopped")
                return hook

            handles = [
                layers[index].register_forward_pre_hook(
                    _make_branch_hook(index), with_kwargs=True
                )
                for index in branch_first_indices
            ]
        else:
            handles = [
                layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
            ]

        # `pre_quantize_generate_hook_start` materializes the multimodal side
        # modules onto the forward device, and the pre-hooks above abort the
        # forward once every branch's first layer has been captured. Both mutate
        # the model for a bounded window, so both are acquired inside the `try`
        # and released in its `finally`: a failing start (for example an OOM
        # while moving a vision tower) must not leave a half-materialized model
        # or registered hooks behind for the rest of the quantization run.
        #
        # `capture_failed` tracks *this call's* exit state. Sampling
        # `sys.exc_info()` in the `finally` instead reads the thread's active
        # exception, which can belong to an enclosing handler in the caller
        # (a retry loop, `requantize()`, an application `except` block). That
        # would classify a successful capture as failed and silently swallow a
        # genuine release error.
        capture_failed = False
        try:
            self.gptq_model.pre_quantize_generate_hook_start()

            # The first `linear` of the capture forward pays a one-time accelerator-library
            # init unless the background prewarm kicked at load time already completed it.
            # All CPU-bound setup is done here, so joining now costs only the prewarm's
            # remaining tail; racing it instead would pay the init a second time inline.
            await_torch_linear_prewarm(
                self.gptq_model.quantize_config.device,
                _forward_dtype(self.gptq_model.model),
            )
            for batch_index, example in enumerate(calibration_data, start=1):
                batch_captured.clear()
                if self.gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                    forward_device = self.gptq_model.quantize_config.device
                else:
                    forward_device = (
                        self.gptq_model.quantize_config.device
                        if _has_vision_inputs(example)
                        else cur_layer_device
                    )
                forward_device = self._resolve_forward_device(example, forward_device)
                forward_device = normalize_device_like(forward_device) or CPU
                if (
                    embed_quant_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
                    and "input_ids" in example
                    and input_embeddings is not None
                ):
                    embedding_device = get_device(input_embeddings)
                    if embedding_device != META:
                        forward_device = embedding_device
                example = self.gptq_model.move_input_capture_example(example, forward_device)
                if (
                    embed_quant_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
                    and "input_ids" in example
                ):
                    src_inputs.append([move_to(example["input_ids"], device=CPU)])
                begin_example = getattr(
                    self.gptq_model, "begin_input_capture_example", None
                )
                end_example = getattr(
                    self.gptq_model, "end_input_capture_example", None
                )
                if callable(begin_example):
                    begin_example(example=example, batch_device=forward_device)
                try:
                    with ctx(
                        DEVICE_THREAD_POOL.read_lock(self.gptq_model.quantize_config.device),
                        device_ctx(self.gptq_model.quantize_config.device),
                    ):
                        self.gptq_model.run_input_capture(
                            example,
                            use_cache=use_cache,
                            data_device=forward_device,
                        )
                except StopForward:
                    pass
                finally:
                    if callable(end_example):
                        end_example()
                    processed_batches = batch_index
                    if cache_forward_pb is not None:
                        rows_for_batch = 0
                        progress_layer_inputs = (
                            branch_captures[branch_first_indices[0]]["layer_inputs"]
                            if multi_branch
                            else layer_inputs
                        )
                        if batch_index <= len(progress_layer_inputs):
                            rows_for_batch = self.looper._batch_row_count(
                                progress_layer_inputs[batch_index - 1]
                            )
                            if rows_for_batch <= 0:
                                rows_for_batch = 1
                        processed_rows += rows_for_batch
                        cache_forward_pb.current_iter_step = processed_batches
                        subtitle = f"Batch {processed_batches}/{cache_total_batches}"
                        if processed_rows > 0:
                            subtitle += f" rows {processed_rows}"
                        cache_forward_pb.subtitle(subtitle).draw()
                if multi_branch:
                    missing = [
                        index
                        for index in branch_first_indices
                        if index not in batch_captured
                    ]
                    if missing:
                        missing_names = [
                            layer_names[index]
                            if layer_names and index < len(layer_names)
                            else str(index)
                            for index in missing
                        ]
                        raise ValueError(
                            "ModuleLooper: the calibration forward never reached the first layer "
                            f"of module-tree branch(es) {missing_names} for calibration batch "
                            f"{batch_index}. Refusing to calibrate a branch on another branch's "
                            "activations. Provide calibration data that exercises every declared "
                            "branch, or remove the unreachable branch from module_tree."
                        )
        except BaseException:
            # Record the failure before unwinding so the cleanup below can
            # distinguish "the capture raised" from "only the release raised".
            capture_failed = True
            raise
        finally:
            # Closing the progress bar is cosmetic; releasing the model state is
            # not. Do not let a failing renderer skip the cleanup below, which is
            # the only thing that un-mutates the model and detaches the pre-hook.
            # A renderer error also follows the same precedence as a release
            # error: it propagates when nothing else is in flight, but is logged
            # instead of masking a capture error that is already unwinding.
            try:
                if cache_forward_pb is not None:
                    cache_forward_pb.close()
            except Exception:
                # Deliberately not `BaseException`: an interrupt must stay
                # interruptible even while a capture error is unwinding.
                if not capture_failed:
                    raise
                self.logger.warn(
                    "cache_inputs() progress-bar close failed during capture cleanup; "
                    "re-raising the original capture error:\n" + traceback.format_exc()
                )
            finally:
                # Cleanup must run on every exit path, not only a successful
                # capture: a calibration forward can raise for ordinary reasons.
                # The handle removal is nested so a failing model-specific release
                # still detaches the pre-hook. When another exception is already
                # propagating, a failure from `pre_quantize_generate_hook_end` is
                # logged and the original calibration error is re-raised, so the
                # user sees the cause instead of a cleanup symptom.
                try:
                    self.gptq_model.pre_quantize_generate_hook_end()
                except Exception:
                    # Deliberately not `BaseException`: an interrupt must stay
                    # interruptible even while a capture error is unwinding.
                    if not capture_failed:
                        raise
                    self.logger.warn(
                        "pre_quantize_generate_hook_end() failed during capture cleanup; "
                        "re-raising the original capture error:\n" + traceback.format_exc()
                    )
                finally:
                    for handle in handles:
                        handle.remove()

        branch_first_layer_inputs = None
        if multi_branch:
            branch_first_layer_inputs = {}
            for branch_index in branch_first_indices:
                capture = branch_captures[branch_index]
                branch_first_layer_inputs[branch_index] = InputCache(
                    layer_inputs=capture["layer_inputs"],
                    layer_input_kwargs=capture["layer_input_kwargs"],
                    position_ids=capture["position_ids"],
                    attention_masks=capture["attention_masks"],
                    src_inputs=src_inputs,
                )
            # Keep the top-level cache fields as the first branch's inputs so
            # existing consumers that read the cache directly still see a
            # complete replay boundary.
            first_branch = branch_first_layer_inputs[branch_first_indices[0]]
            layer_inputs = first_branch.layer_inputs
            layer_input_kwargs = first_branch.layer_input_kwargs
            position_ids = first_branch.position_ids
            attention_masks = first_branch.attention_masks

        result = InputCache(
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
            src_inputs=src_inputs,
            branch_first_layer_inputs=branch_first_layer_inputs,
        )

        if timer is not None and start_time is not None:
            timer.record(
                "capture_inputs",
                time.perf_counter() - start_time,
                source=capture_source,
            )

        return result
