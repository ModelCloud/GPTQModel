# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Isolated stage for capturing calibration inputs prior to quantization."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

import torch

from .. import DEVICE_THREAD_POOL
from ..looper.input_cache import InputCache
from ..nn_modules.hooked_linear import STOP_FORWARD_EXCEPTION, StopForward
from ..quantization.config import QuantizeEmbed
from ..utils.ctx import ctx
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.looper_helpers import device_ctx, normalize_device_like, select_forward_devices
from ..utils.model import get_module, get_module_by_name_prefix, move_to, nested_move_to
from ..utils.offload import offload_to_disk
from ..utils.torch import CPU, META


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .module_looper import ModuleLooper


_VISION_INPUT_KEYS = frozenset(("pixel_values", "images"))


def _has_vision_inputs(example: Dict[str, Any]) -> bool:
    return any(key in example for key in _VISION_INPUT_KEYS)


def _resolve_input_capture_device(
    *,
    example: Dict[str, Any],
    embed_quant_mode: Optional[QuantizeEmbed],
    input_embeddings: Optional[torch.nn.Module],
    fallback_device: torch.device,
) -> torch.device:
    """Resolve where the model input must execute during calibration capture."""

    if (
        embed_quant_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
        and "input_ids" in example
        and input_embeddings is not None
    ):
        embedding_device = get_device(input_embeddings)
        if embedding_device != META:
            return embedding_device
    return fallback_device


class StageInputsCapture:
    """Capture layer inputs so processors can reuse cached activations."""

    def __init__(self, looper: ModuleLooper, logger=None) -> None:
        """Binds the capture stage to a looper instance and logger."""

        self.looper = looper
        self.gptq_model = looper.gptq_model
        self.logger = logger or setup_logger()

    def _materialize_modules_with_direct_meta_tensors(self, device: torch.device) -> None:
        for module_name in self.gptq_model.get_modules_with_direct_meta_tensors(self.gptq_model.model):
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
        """Resolve where token inputs must execute for the materialized embedding."""

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
        position_ids: List[Optional[torch.Tensor]] = []
        layer_input_kwargs: List[Dict[str, Any]] = []

        timer = getattr(self.gptq_model, "quant_region_timer", None)
        layer_label = None
        if layers:
            first_layer = layers[0]
            # `LazyTurtle` resolves checkpoint keys by prefixing `module_path` to
            # relative tensor names (e.g. `model.layers.0` + `.mlp.gate.tid2eid`).
            # The caller-supplied dotted path is the single source of truth for
            # materialization; keep the display label separate so a class-name
            # fallback is never passed as a checkpoint prefix.
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
                # Fallback: discover the dotted path from the model tree. This keeps
                # `LazyTurtle` materialization correct when callers do not pass names.
                for name, mod in self.gptq_model.model.named_modules():
                    if mod is first_layer:
                        module_path = name
                        break

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

        # The first-layer pre-hook raises before the layer body executes, so its
        # parameters are deliberately left as checkpoint-backed shells here.
        # Base modules (embeddings, model-level norms, and other pre-layer
        # components) are materialized below on the capture device. The layer's
        # leaves are then loaded once by the quantization stage, directly onto
        # their module-tree-planned devices.
        capture_device = normalize_device_like(self.gptq_model.quantize_config.device) or CPU
        cur_layer_device = capture_device

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

        def store_input_hook(module, args, kwargs):
            """Captures the incoming batch for the first layer and aborts the forward."""

            # Select device for this batch (round-robin for balanced mode)
            if balanced_mode and balanced_devices:
                batch_device = balanced_devices[balanced_rr_counter[0] % len(balanced_devices)]
                balanced_rr_counter[0] += 1
            else:
                batch_device = data_device

            layer_input = self.gptq_model.capture_first_layer_positional_inputs(
                args=args,
                kwargs=kwargs,
                batch_device=batch_device,
            )

            layer_inputs.append(layer_input)

            if kwargs.get("attention_mask") is not None:
                attention_masks.append(kwargs["attention_mask"].to(device=batch_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=batch_device))
            else:
                # Preserve one metadata slot per captured batch. Omitting None
                # entries shifts every later position_ids tensor to the wrong
                # replay batch.
                position_ids.append(None)
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
            layer_input_kwargs.append(one_kwargs)

            # In normal repeating layer/sbuset early stop happens on the last module forward
            # but the first model input embedding call we use a simple model register forwar hook
            # and wait for the first instance this callback is called
            raise STOP_FORWARD_EXCEPTION

        # Parameters attached to the shell root must be ready before embedding forward.
        self._materialize_modules_with_direct_meta_tensors(cur_layer_device)

        def _resolve_module_name(mod: torch.nn.Module) -> Optional[str]:
            for name, m in self.gptq_model.model.named_modules():
                if m is mod:
                    return name
            return None

        input_embeddings = self.gptq_model.get_input_embeddings()
        input_embeddings_name = self.gptq_model.get_input_embeddings_name()

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

            if (
                embed_quant_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
                and module_name == input_embeddings_name
            ):
                # Do not move embeddings to the CPU when quantizing them.
                continue

            resolved_name = _resolve_module_name(module)
            m_device = get_device(module)
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device
            self.gptq_model.shell_module_materialize(
                target_submodule=module,
                device=cur_layer_device,
                module_path=resolved_name,
            )

        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        generate_hook_started = False
        try:
            self.gptq_model.pre_quantize_generate_hook_start()
            generate_hook_started = True
            for batch_index, example in enumerate(calibration_data, start=1):
                if self.gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                    forward_device = self.gptq_model.quantize_config.device
                else:
                    forward_device = (
                        self.gptq_model.quantize_config.device
                        if _has_vision_inputs(example)
                        else cur_layer_device
                    )
                data_device = _resolve_input_capture_device(
                    example=example,
                    embed_quant_mode=embed_quant_mode,
                    input_embeddings=input_embeddings,
                    fallback_device=data_device,
                )
                data_device = self._resolve_forward_device(example, data_device)
                example = self.gptq_model.move_input_capture_example(example, data_device)
                if (
                        embed_quant_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
                        and "input_ids" in example
                ):
                    src_inputs.append([move_to(example["input_ids"], device=self.gptq_model.quantize_config.device)])
                captured_before = len(layer_inputs)
                capture_completed = False
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
                    capture_completed = True
                except StopForward:
                    capture_completed = True
                finally:
                    if capture_completed and len(layer_inputs) != captured_before + 1:
                        raise RuntimeError(
                            "Input capture did not reach the first-layer pre-hook exactly once; "
                            "refusing to continue with an unverified shell layer."
                        )
                    processed_batches = batch_index
                    if cache_forward_pb is not None:
                        rows_for_batch = 0
                        if batch_index <= len(layer_inputs):
                            rows_for_batch = self.looper._batch_row_count(
                                layer_inputs[batch_index - 1]
                            )
                            if rows_for_batch <= 0:
                                rows_for_batch = 1
                        processed_rows += rows_for_batch
                        cache_forward_pb.current_iter_step = processed_batches
                        subtitle = f"Batch {processed_batches}/{cache_total_batches}"
                        if processed_rows > 0:
                            subtitle += f" rows {processed_rows}"
                        cache_forward_pb.subtitle(subtitle).draw()
        finally:
            try:
                if cache_forward_pb is not None:
                    cache_forward_pb.close()
            finally:
                try:
                    if generate_hook_started:
                        self.gptq_model.pre_quantize_generate_hook_end()
                finally:
                    handle.remove()

        # In offload_to_disk mode the input embedding is no longer needed once
        # hidden-state inputs are cached. Move it to disk (unless it is tied
        # with lm_head, which must remain on the accelerator for its own
        # quantization pass) so later layers do not keep the embedding weight
        # resident in device memory.
        if (
            self.gptq_model.quantize_config.offload_to_disk
            and input_embeddings is not None
            and embed_quant_mode not in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
            and not getattr(self.gptq_model.model.config, "tie_word_embeddings", False)
        ):
            offload_path = self.gptq_model.quantize_config.offload_to_disk_path
            if offload_path:
                offload_to_disk(
                    module=input_embeddings,
                    model=self.gptq_model.model,
                    disk_path=offload_path,
                )

        result = InputCache(
            src_inputs=src_inputs,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
        )

        if timer is not None and start_time is not None:
            timer.record(
                "capture_inputs",
                time.perf_counter() - start_time,
                source=capture_source,
            )

        return result
