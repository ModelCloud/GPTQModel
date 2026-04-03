# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Isolated stage for capturing calibration inputs prior to quantization."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

import torch

from .. import DEVICE_THREAD_POOL
from ..looper.input_cache import InputCache
from ..nn_modules.hooked_linear import STOP_FORWARD_EXCEPTION, StopForward
from ..utils.ctx import ctx
from ..utils.device import get_device
from ..utils.looper_helpers import device_ctx, select_forward_devices
from ..utils.logger import setup_logger
from ..utils.model import get_module_by_name_prefix, move_to, nested_move_to
from ..utils.torch import CPU, META

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .module_looper import ModuleLooper


class StageInputsCapture:
    """Capture layer inputs so processors can reuse cached activations."""

    def __init__(self, looper: ModuleLooper, logger=None) -> None:
        """Binds the capture stage to a looper instance and logger."""

        self.looper = looper
        self.gptq_model = looper.gptq_model
        self.logger = logger or setup_logger()

    def cache_inputs(
        self,
        layers: Sequence[torch.nn.Module],
        calibration_data: Iterable[Dict[str, torch.Tensor]],
        use_cache: bool,
    ) -> InputCache:
        """Runs a short forward over calibration data and caches first-layer inputs."""

        layer_inputs: List[List[torch.Tensor]] = []
        attention_masks: List[torch.Tensor | None] = []
        position_ids: List[torch.Tensor] = []
        layer_input_kwargs: List[Dict[str, Any]] = []

        timer = getattr(self.gptq_model, "quant_region_timer", None)
        layer_label = None
        if layers:
            first_layer = layers[0]
            layer_label = getattr(first_layer, "full_name", None)
            if layer_label is None:
                layer_label = getattr(getattr(first_layer, "__class__", None), "__name__", None)
            if layer_label is None:
                layer_label = type(first_layer).__name__
            capture_source = f"cache_inputs:{layer_label}"
        else:
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
        layers[0] = self.gptq_model.shell_module_materialize(
            target_submodule=layers[0],
            device=CPU,
        )
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

        ori_outside_layer_module_devices: Dict[str, torch.device] = {}
        for module_name in self.gptq_model.get_base_modules(self.gptq_model.model):
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue

            m_device = get_device(module)
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device
            self.gptq_model.shell_module_materialize(
                target_submodule=module,
                device=cur_layer_device,
            )

        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)

        self.gptq_model.pre_quantize_generate_hook_start()

        # TODO: why data_device sometimes set to cuda (self.gptq_model.quantize_config.device) and sometimes to CPU (cur_layer_device)?
        try:
            for batch_index, example in enumerate(calibration_data, start=1):
                if self.gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                    data_device = self.gptq_model.quantize_config.device
                else:
                    data_device = (
                        self.gptq_model.quantize_config.device
                        if "pixel_values" in example.keys()
                        else cur_layer_device
                    )
                example = self.gptq_model.move_input_capture_example(example, data_device)
                try:
                    with ctx(
                        DEVICE_THREAD_POOL.read_lock(self.gptq_model.quantize_config.device),
                        device_ctx(self.gptq_model.quantize_config.device),
                    ):
                        self.gptq_model.run_input_capture(
                            example,
                            use_cache=use_cache,
                            data_device=data_device,
                        )
                except StopForward:
                    pass
                finally:
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
            if cache_forward_pb is not None:
                cache_forward_pb.close()

        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()

        result = InputCache(
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
