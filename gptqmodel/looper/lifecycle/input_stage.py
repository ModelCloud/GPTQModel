from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from ... import DEVICE_THREAD_POOL
from ...nn_modules.hooked_linear import STOP_FORWARD_EXCEPTION, StopForward
from ...utils.ctx import ctx
from ...utils.looper_helpers import device_ctx
from ...utils.model import move_to, nested_move_to, get_module_by_name_prefix
from ...utils.device import get_device
from ...utils.torch import CPU, META
from ...utils.logger import setup_logger
from ..input_cache import InputCache

if TYPE_CHECKING:  # pragma: no cover
    from logbar.progress import ProgressBar
    from ..module_looper import ModuleLooper


class InputCaptureStage:
    """Capture calibration inputs before the layer loop begins."""

    def __init__(self, looper: "ModuleLooper", logger: Optional[logging.Logger] = None) -> None:
        self.looper = looper
        self.log = logger or setup_logger()

    def capture(self, layers, calibration_data, use_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        gptq_model = self.looper.gptq_model
        timer = getattr(gptq_model, "quant_region_timer", None)
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
            calibration_batches = len(calibration_data)
        except (TypeError, AttributeError):
            calibration_batches = None

        if calibration_batches is None:
            self.log.info("ModuleLooper: capturing layer inputs (batch count unknown)")
        else:
            self.log.info(
                "ModuleLooper: capturing layer inputs from %s calibration batches",
                calibration_batches,
            )

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device

        cache_forward_pb: ProgressBar = None
        processed_rows = 0
        cache_total_batches = None
        if calibration_batches is not None and calibration_batches > 0:
            cache_total_batches = int(calibration_batches)
            cache_forward_pb = (
                self.log.pb(range(max(cache_total_batches, 1)))
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
            layer_input = []
            if kwargs.get("hidden_states") is not None:
                layer_input.append(move_to(kwargs["hidden_states"], device=data_device))
            else:
                layer_input.append(move_to(args[0], device=data_device))

            layer_inputs.append(layer_input)

            if kwargs.get("attention_mask") is not None:
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs: Dict[str, torch.Tensor] = {}
            for (k, v) in kwargs.items():
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            raise STOP_FORWARD_EXCEPTION

        if cur_layer_device == META:
            layers[0] = gptq_model.shell_module_materialize(
                target_submodule=layers[0],
                device=gptq_model.quantize_config.device,
            )
            cur_layer_device = gptq_model.quantize_config.device
        else:
            layers[0] = layers[0].to(gptq_model.quantize_config.device)

        ori_outside_layer_module_devices = {}
        for module_name in gptq_model.get_base_modules(gptq_model.model):
            module, _ = get_module_by_name_prefix(gptq_model.model, [module_name])

            if module is None:
                continue

            m_device = get_device(module)
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device
            if module is not None:
                gptq_model.shell_module_materialize(
                    target_submodule=module,
                    device=cur_layer_device,
                )

        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)

        is_ovis = gptq_model.__class__.__name__ == "OvisGPTQ"

        gptq_model.pre_quantize_generate_hook_start()

        try:
            for batch_index, example in enumerate(calibration_data, start=1):
                for k, v in example.items():
                    if gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                        data_device = gptq_model.quantize_config.device
                    else:
                        data_device = (
                            gptq_model.quantize_config.device
                            if k == "pixel_values"
                            else cur_layer_device
                        )
                    if isinstance(v, list):
                        for idx in range(len(v)):
                            if len(v[idx].shape) == 1:
                                v[idx] = v[idx].unsqueeze(0)
                            v[idx] = move_to(
                                v[idx].to(gptq_model.model.visual_tokenizer.dtype)
                                if is_ovis
                                else v[idx],
                                device=data_device,
                            )
                    else:
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        example[k] = move_to(v, device=data_device)
                try:
                    if gptq_model.ATTENTION_MASKS_DTYPE is torch.long:
                        example["attention_mask"] = example["attention_mask"].long()

                    with ctx(
                        DEVICE_THREAD_POOL.read_lock(gptq_model.quantize_config.device),
                        device_ctx(gptq_model.quantize_config.device),
                    ):
                        if gptq_model.INPUT_EMBEDDING_EXTRA_ARGS:
                            gptq_model.model.generate(
                                **example,
                                **gptq_model.INPUT_EMBEDDING_EXTRA_ARGS,
                            )
                        else:
                            gptq_model.model(**example, use_cache=use_cache)
                except StopForward:
                    pass
                finally:
                    processed_batches = batch_index
                    if cache_forward_pb is not None:
                        rows_for_batch = 0
                        if batch_index <= len(layer_inputs):
                            rows_for_batch = self.looper._batch_row_count(layer_inputs[batch_index - 1])
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

        gptq_model.pre_quantize_generate_hook_end()
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
