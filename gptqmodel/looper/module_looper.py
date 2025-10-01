# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utilities for orchestrating the quantisation loop across multiple devices.

ModuleLooper is the high-level coordinator that fans calibration batches across
the available accelerators, runs each processing stage, and keeps the shell and
turtle model state coherent. The implementation mixes synchronous orchestration
with asynchronous workers, so the helpers below focus on keeping device context
consistent and ensuring data dependencies survive the roundtrips through the
thread pool.
"""

from __future__ import annotations

import copy
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
from torch.nn.parallel import replicate as torch_replicate

from ..looper.dequantize_processor import DequantizeProcessor
from ..looper.eora_processor import EoraProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.input_cache import InputCache
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import SUPPORTS_MODULE_TYPES, DEVICE
from ..nn_modules.hooked_linear import (STOP_FORWARD_EXCEPTION, HookedLinear,
                                        StopForward, replace_module_with_hooked_legacy)
from .. import DEBUG_ON
from ..utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask
from ..utils.device import get_device, get_device_new
from ..utils.logger import setup_logger
from ..utils.model import find_modules, get_module, get_module_by_name_prefix, move_to, nested_move_to
from ..utils.offload import offload_to_disk
from ..utils.threadx import DeviceThreadPool
from ..utils.torch import (ALL_DEVICES, CPU, META, device_next, device_next_reset, torch_sync)
from .awq_processor import AWQProcessor
from .qqq_processor import QQQProcessor

log = setup_logger()


# -------------------- Device helpers (local) --------------------

@contextmanager
def _device_ctx(dev: Optional[torch.device|DEVICE]):
    """
    Ensure the caller thread’s current device matches `dev` for the duration of the
    context (CUDA/XPU). Prevents cuBLAS/cuDNN handle/device mismatches in multi-GPU.
    """
    if dev is None:
        yield
    else:
        if dev.type == "cuda":
            with torch.cuda.device(dev.index):
                yield
        elif dev.type == "xpu" and hasattr(torch, "xpu"):
            with torch.xpu.device(dev.index):  # type: ignore[attr-defined]
                yield
        else:
            # cpu/mps/meta -> nothing special needed
            yield


@torch.inference_mode()
def _rehome_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    *,
    move_parameters: bool = False,
    move_buffers: bool = True,
    include_non_persistent_buffers: bool = True,
    only_mismatched: bool = True,
):
    """
    Move a module's **registered** tensors to `device`.
    Defaults to buffers-only (fast; fixes RoPE cos/sin caches and other internal state).
    Parameters can be moved too, but that risks breaking weight tying and increases VRAM churn.
    """
    for sub in module.modules():
        # Buffers (covers most cached internal tensors if properly registered)
        if move_buffers:
            np_set = getattr(sub, "_non_persistent_buffers_set", set())
            for name, buf in list(getattr(sub, "_buffers", {}).items()):
                if buf is None or not isinstance(buf, torch.Tensor):
                    continue
                if not include_non_persistent_buffers and name in np_set:
                    continue
                if only_mismatched and buf.device == device:
                    continue
                try:
                    sub._buffers[name] = buf.to(device, non_blocking=True)
                except Exception:
                    try:
                        sub._buffers[name] = buf.to(device)
                    except Exception:
                        pass

        # Parameters (rarely needed; default False)
        if move_parameters:
            for pname, p in list(getattr(sub, "_parameters", {}).items()):
                if p is None or not isinstance(p, torch.nn.Parameter):
                    continue
                if only_mismatched and p.device == device:
                    continue
                try:
                    with torch.no_grad():
                        new_p = torch.nn.Parameter(
                            p.data.to(device, non_blocking=True),
                            requires_grad=p.requires_grad
                        )
                    sub._parameters[pname] = new_p
                except Exception:
                    try:
                        with torch.no_grad():
                            new_p = torch.nn.Parameter(
                                p.data.to(device),
                                requires_grad=p.requires_grad
                            )
                        sub._parameters[pname] = new_p
                    except Exception:
                        pass


class ModuleLooper():
    """Drive the per-layer quantisation workflow over one or more devices.

    The looper owns a :class:`DeviceThreadPool` that executes CPU and accelerator
    work. Forward passes can be replicated across devices, processors can enqueue
    asynchronous tasks, and the class handles the bookkeeping required to stitch
    the results back into a sequential quantisation order.
    """
    def __init__(self, model: BaseQModel, processors: List[LoopProcessor]):
        self.processors = processors
        self.gptq_model = model
        self.support_batch_quantize = model.support_batch_quantize
        self.lock = threading.Lock()

        # The looper shares one pool for its lifetime so tasks such as module
        # reloading, forward passes and finalisation reuse the same worker
        # threads. The first worker per device is treated as the serial lane for
        # forward execution; additional workers handle background jobs.
        self.pool = DeviceThreadPool(
            inference_mode=True,
            workers={
                "cuda:per": 4, # unique memory per instance
                "xpu:per": 1, # unique memory per instance
                "mps": 8, # unified memory
                "cpu": 8, # unified memory
            },
            empty_cache_every_n=1024,  # disable auto gc based gpu work rate; enable if you want
        )

        self.gptq_model.register_background_pool(self.pool)

        for processor in self.processors:
            self._processor_mask_tls(processor)

    # Processors capture activations through hooks that need thread-local state
    # so masks survive the roundtrip to worker threads.
    def _processor_mask_tls(self, processor: LoopProcessor) -> threading.local:
        tls = getattr(processor, "_mask_tls", None)
        if tls is None:
            tls = threading.local()
            setattr(processor, "_mask_tls", tls)
        return tls

    def _set_processor_mask(self, processor: LoopProcessor, mask):
        tls = self._processor_mask_tls(processor)
        tls.value = mask

    def _get_processor_mask(self, processor: LoopProcessor):
        tls = getattr(processor, "_mask_tls", None)
        return getattr(tls, "value", None) if tls else None

    def _select_forward_devices(self, base_device: Optional[torch.device]) -> List[torch.device]:
        if base_device is None:
            return [CPU]

        devices = [base_device]
        base_type = base_device.type
        if base_type in ("cuda", "xpu", "mps"):
            for dev in ALL_DEVICES:
                if dev.type == base_type and dev not in devices:
                    devices.append(dev)
        return devices

    def _clone_module_for_devices(self, module: torch.nn.Module, devices: List[torch.device]) -> Dict[torch.device, torch.nn.Module]:
        clones: Dict[torch.device, torch.nn.Module] = {}
        module_label = getattr(module, "full_name", module.__class__.__name__)
        clone_timings = [] if DEBUG_ON else None

        if len(devices) == 1:
            target_device = devices[0]
            module = module.to(target_device)
            module.eval()
            _rehome_module_to_device(module, target_device, move_parameters=True, move_buffers=True)
            self._clear_non_picklable_state(module)
            setattr(module, "_gptqmodule_device_hint", target_device)
            clones[target_device] = module
            return clones

        self._clear_non_picklable_state(module)
        primary_device = devices[0]
        replicate_devices = [dev for dev in devices if dev.type == primary_device.type]
        use_replicate = (
            len(replicate_devices) == len(devices)
            and primary_device.type == "cuda"
            and torch.cuda.is_available()
        )

        if use_replicate:
            try:
                module = module.to(primary_device)
                module.eval()
                _rehome_module_to_device(module, primary_device, move_parameters=True, move_buffers=True)
                setattr(module, "_gptqmodule_device_hint", primary_device)
                replicate_start = time.perf_counter() if DEBUG_ON else None
                replicas = torch_replicate(module, devices)
                if clone_timings is not None and replicate_start is not None:
                    clone_timings.append(("replicate", time.perf_counter() - replicate_start))

                for dev, replica in zip(devices, replicas):
                    replica.eval()
                    _rehome_module_to_device(replica, dev, move_parameters=True, move_buffers=True)
                    self._clear_non_picklable_state(replica)
                    setattr(replica, "_gptqmodule_device_hint", dev)
                    clones[dev] = replica

                if clone_timings:
                    timing_str = ", ".join(
                        f"{str(dev)}={duration * 1000:.2f}ms" for dev, duration in clone_timings
                    )
                    log.debug(f"ModuleLooper: replicate {module_label} -> {timing_str}")
                return clones
            except Exception:
                log.info("Clone: fast clone failed")
                # Fall back to deepcopy path if replicate is unsupported for this module.
                if clone_timings is not None:
                    clone_timings.append(("replicate_failed", 0.0))

        for dev in devices:
            start_ts = time.perf_counter() if DEBUG_ON else None
            replica = copy.deepcopy(module)
            replica = replica.to(dev)
            replica.eval()
            _rehome_module_to_device(replica, dev, move_parameters=True, move_buffers=True)
            self._clear_non_picklable_state(replica)
            setattr(replica, "_gptqmodule_device_hint", dev)
            clones[dev] = replica
            if clone_timings is not None and start_ts is not None:
                clone_timings.append((dev, time.perf_counter() - start_ts))

        if clone_timings:
            timing_str = ", ".join(f"{str(dev)}={duration * 1000:.2f}ms" for dev, duration in clone_timings)
            log.debug(f"ModuleLooper: deepcopy {module_label} -> {timing_str}")
        return clones

    def _clear_non_picklable_state(self, module: torch.nn.Module):
        cleared = []
        seen = set()

        def maybe_clear(obj: torch.nn.Module):
            if id(obj) in seen:
                return
            seen.add(id(obj))

        if isinstance(module, torch.nn.Module):
            for sub in module.modules():
                maybe_clear(sub)
        else:
            maybe_clear(module)

        return cleared

    @staticmethod
    def _forward_batch_worker(
        module: torch.nn.Module,
        processor: LoopProcessor,
        batch_index: int,
        layer_input: List[torch.Tensor],
        layer_input_kwargs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        *,
        support_batch_quantize: bool,
        is_lm_head_module: bool,
        need_output: bool,
        reuse_kv: bool,
        prev_kv,
    ):
        """Run one forward micro-batch on a pool worker and return its output.

        The worker receives pre-moved inputs, executes the module on its bound
        device and ships back both the model outputs and any next-layer KV cache
        update. The thin signature keeps the function pickleable for the worker
        queue.
        """
        module_device = getattr(module, "_gptqmodule_device_hint", None) or get_device(module)
        _rehome_module_to_device(module, module_device, move_parameters=True, move_buffers=True)

        inputs = [move_to(inp, device=module_device, stream=False) for inp in layer_input]

        attn_tensor = None
        if attention_mask is not None:
            attn_tensor = move_to(attention_mask, device=module_device, stream=False)

        additional_inputs: Dict[str, torch.Tensor] = {}
        if support_batch_quantize and attn_tensor is not None:
            additional_inputs["attention_mask"] = attn_tensor

        if position_ids is not None:
            additional_inputs["position_ids"] = move_to(position_ids, device=module_device, stream=False)

        for key, value in layer_input_kwargs.items():
            additional_inputs[key] = nested_move_to(value, device=module_device, stream=False)

        keep_mask = None
        if attn_tensor is not None:
            seq_len = inputs[0].shape[1] if (len(inputs) > 0 and inputs[0].dim() >= 2) else None
            keep_mask = normalize_seq_mask(attn_tensor, seq_len=seq_len)

        mask_tls = getattr(processor, "_mask_tls", None)
        if mask_tls is not None:
            mask_tls.value = keep_mask

        if reuse_kv and prev_kv is not None:
            additional_inputs["kv_last_layer"] = nested_move_to(prev_kv, device=module_device, stream=False)

        module_output = None
        kv_next = None
        try:
            if is_lm_head_module:
                module_output = module(*inputs)
            else:
                module_output = module(*inputs, **additional_inputs)
        except StopForward:
            module_output = None
        finally:
            if mask_tls is not None:
                mask_tls.value = None

        if reuse_kv and module_output is not None and isinstance(module_output, tuple) and len(module_output) > 0:
            kv_next = module_output[-1]

        result_output = module_output if need_output else None
        return batch_index, result_output, kv_next

    def _run_forward_batches(
        self,
        *,
        module: torch.nn.Module,
        processor: LoopProcessor,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        position_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        cur_layer_device: torch.device,
        is_lm_head_module: bool,
        shared_kv_cache_dict: Dict[int, torch.Tensor],
        layer_index: int,
        need_outputs: bool,
        reuse_kv: bool,
    ) -> List[List[torch.Tensor]]:
        """Dispatch the captured layer inputs through the module.

        When multiple accelerators of the same type are available we clone the
        module and execute batches in parallel, otherwise we fall back to a
        single threaded path. The helper returns the ordered outputs that feed
        the next processor stage when ``need_outputs`` is set.
        """
        devices = self._select_forward_devices(cur_layer_device)

        if len(devices) <= 1:
            return self._run_forward_batches_single(
                module=module,
                processor=processor,
                layer_inputs=layer_inputs,
                layer_input_kwargs=layer_input_kwargs,
                position_ids=position_ids,
                attention_masks=attention_masks,
                cur_layer_device=cur_layer_device,
                is_lm_head_module=is_lm_head_module,
                shared_kv_cache_dict=shared_kv_cache_dict,
                layer_index=layer_index,
                need_outputs=need_outputs,
                reuse_kv=reuse_kv,
            )

        return self._run_forward_batches_parallel(
            module=module,
            processor=processor,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
            cur_layer_device=cur_layer_device,
            is_lm_head_module=is_lm_head_module,
            shared_kv_cache_dict=shared_kv_cache_dict,
            layer_index=layer_index,
            need_outputs=need_outputs,
            reuse_kv=reuse_kv,
            devices=devices,
        )

    def _run_forward_batches_single(
        self,
        *,
        module: torch.nn.Module,
        processor: LoopProcessor,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        position_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        cur_layer_device: torch.device,
        is_lm_head_module: bool,
        shared_kv_cache_dict: Dict[int, torch.Tensor],
        layer_index: int,
        need_outputs: bool,
        reuse_kv: bool,
    ) -> List[List[torch.Tensor]]:
        """Sequential fallback when only one forward device is in use."""
        outputs: List[List[torch.Tensor]] = []
        prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None

        for batch_idx in range(processor.num_batches):
            layer_input = [move_to(inp, device=cur_layer_device, stream=False) for inp in layer_inputs[batch_idx]]

            raw_mask = attention_masks[batch_idx]
            attn_tensor = raw_mask if raw_mask is None else move_to(raw_mask, device=cur_layer_device, stream=False)

            keep_mask = None
            if attn_tensor is not None:
                seq_len = layer_input[0].shape[1] if (len(layer_input) > 0 and layer_input[0].dim() >= 2) else None
                keep_mask = normalize_seq_mask(attn_tensor, seq_len=seq_len)
            self._set_processor_mask(processor, keep_mask)

            additional_inputs: Dict[str, torch.Tensor] = {}
            if self.support_batch_quantize and attn_tensor is not None:
                additional_inputs["attention_mask"] = attn_tensor

            if position_ids:
                pos = position_ids[batch_idx]
                if pos is not None:
                    additional_inputs["position_ids"] = move_to(pos, device=cur_layer_device, stream=False)

            for key, value in layer_input_kwargs[batch_idx].items():
                additional_inputs[key] = nested_move_to(value, device=cur_layer_device, stream=False)

            if reuse_kv and prev_kv is not None:
                additional_inputs["kv_last_layer"] = nested_move_to(prev_kv, device=cur_layer_device, stream=False)

            _rehome_module_to_device(module, cur_layer_device, move_parameters=True, move_buffers=True)

            module_output = None
            try:
                if is_lm_head_module:
                    module_output = module(*layer_input)
                else:
                    module_output = module(*layer_input, **additional_inputs)
            except StopForward:
                module_output = None
            finally:
                self._set_processor_mask(processor, None)

            if (
                reuse_kv
                and module_output is not None
                and isinstance(module_output, tuple)
                and len(module_output) > 0
                and shared_kv_cache_dict.get(layer_index) is None
            ):
                shared_kv_cache_dict[layer_index] = module_output[-1]

            if need_outputs and module_output is not None:
                primary = module_output[0] if isinstance(module_output, tuple) else module_output
                primary = move_to(primary, device=cur_layer_device, stream=False)
                outputs.append([primary])

        return outputs

    def _run_forward_batches_parallel(
        self,
        *,
        module: torch.nn.Module,
        processor: LoopProcessor,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        position_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        cur_layer_device: torch.device,
        is_lm_head_module: bool,
        shared_kv_cache_dict: Dict[int, torch.Tensor],
        layer_index: int,
        need_outputs: bool,
        reuse_kv: bool,
        devices: List[torch.device],
    ) -> List[List[torch.Tensor]]:
        """Fan batches across device clones and preserve result ordering."""
        module_replicas = self._clone_module_for_devices(module, devices)

        prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None

        results: Dict[int, torch.Tensor | tuple | None] = {}

        chunk = len(devices)
        total_batches = processor.num_batches
        for start in range(0, total_batches, chunk):
            futures = []
            end = min(start + chunk, total_batches)
            for offset, batch_idx in enumerate(range(start, end)):
                device = devices[offset]
                replica = module_replicas[device]
                submitter = self.pool.submit_serial if device.type in ("cuda", "xpu", "mps") else self.pool.submit

                futures.append(
                    submitter(
                        device,
                        self._forward_batch_worker,
                        replica,
                        processor,
                        batch_idx,
                        layer_inputs[batch_idx],
                        layer_input_kwargs[batch_idx],
                        attention_masks[batch_idx],
                        position_ids[batch_idx] if position_ids else None,
                        support_batch_quantize=self.support_batch_quantize,
                        is_lm_head_module=is_lm_head_module,
                        need_output=need_outputs,
                        reuse_kv=reuse_kv,
                        prev_kv=prev_kv,
                    )
                )

            for fut in futures:
                batch_idx, module_output, kv_next = fut.result()
                if need_outputs and module_output is not None:
                    results[batch_idx] = module_output
                if reuse_kv and kv_next is not None and shared_kv_cache_dict.get(layer_index) is None:
                    shared_kv_cache_dict[layer_index] = nested_move_to(kv_next, device=cur_layer_device, stream=False)

        # ensure replicas that are clones release promptly
        for dev in list(module_replicas.keys()):
            if dev != cur_layer_device:
                del module_replicas[dev]

        if not need_outputs:
            return []

        ordered_outputs: List[List[torch.Tensor]] = []
        for idx in range(total_batches):
            module_output = results.get(idx)
            if module_output is None:
                raise RuntimeError("Forward batch returned no output; data-parallel execution produced empty result.")
            if isinstance(module_output, tuple):
                primary = module_output[0]
            else:
                primary = module_output
            primary = move_to(primary, device=cur_layer_device, stream=False)
            ordered_outputs.append([primary])

        return ordered_outputs

    def _masked_hook_wrapper(self, processor: LoopProcessor, inner_hook):
        def hook(module, inputs, output):
            keep = self._get_processor_mask(processor)

            # Mask first tensor-like input if it's [B, S, ...]
            new_inputs = inputs
            try:
                if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                    x = inputs[0]
                    if keep is not None and x.dim() >= 3:
                        xk = apply_keep_mask_bt(x, keep)
                        if isinstance(inputs, tuple):
                            new_inputs = (xk,) + tuple(inputs[1:])
                        else:
                            new_inputs = [xk] + list(inputs[1:])
            except Exception:
                # Never break the forward due to masking; fall back to original
                new_inputs = inputs

            # Mask primary tensor output if it's [B, S, ...]
            new_output = output
            try:
                if isinstance(output, (tuple, list)) and len(output) > 0:
                    y0 = output[0]
                    if torch.is_tensor(y0) and keep is not None and y0.dim() >= 3:
                        yk = apply_keep_mask_bt(y0, keep)
                        if isinstance(output, tuple):
                            new_output = (yk,) + tuple(output[1:])
                        else:
                            new_output = [yk] + list(output[1:] )
                elif torch.is_tensor(output) and keep is not None and output.dim() >= 3:
                    new_output = apply_keep_mask_bt(output, keep)
            except Exception:
                new_output = output

            return inner_hook(module, new_inputs, new_output)
        return hook

    def cache_inputs(self, layers, calibration_data, use_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device

        # make sure turtle is ready for lias
        self.gptq_model.wait_for_turtle_reload()

        # TODO HookLinear add register_forward_pre_hook()
        def store_input_hook(module, args, kwargs):
            # Positional arguments.
            layer_input = []
            if kwargs.get("hidden_states") is not None:
                layer_input.append(move_to(kwargs["hidden_states"], device=data_device))
            else:
                # If hidden_states is not in kwargs, get it from the first positional argument
                # If error occurs here, check the model's modeling code
                layer_input.append(move_to(args[0], device=data_device))

            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs.get("attention_mask") is not None and self.gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            raise STOP_FORWARD_EXCEPTION

        # move layer to target device
        if cur_layer_device == META:
            layers[0] = self.gptq_model.shell_module_materialize(
                target_submodule=layers[0],
                device=self.gptq_model.quantize_config.device,
            )
            cur_layer_device = self.gptq_model.quantize_config.device
        else:
            layers[0] = layers[0].to(self.gptq_model.quantize_config.device)

        ori_outside_layer_module_devices = {}
        for module_name in self.gptq_model.get_base_modules(self.gptq_model.model):
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue

            m_device = get_device(module)
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device
            if module is not None:
                self.gptq_model.shell_module_materialize(
                    target_submodule=module,
                    device=cur_layer_device,
                )

        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)

        # TODO FIX ME.. remove hard coded Ovis code
        is_ovis = self.gptq_model.__class__.__name__ == "OvisGPTQ"

        # LifeCycle: start pre-first layer embedding hook
        self.gptq_model.pre_quantize_generate_hook_start()

        for example in calibration_data:
            for k, v in example.items():
                if self.gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                    data_device = self.gptq_model.quantize_config.device
                else:
                    data_device = self.gptq_model.quantize_config.device if k == "pixel_values" else cur_layer_device
                if isinstance(v, list):
                    for index in range(len(v)):
                        if len(v[index].shape) == 1:
                            v[index] = v[index].unsqueeze(0)
                        v[index] = move_to(v[index].to(self.gptq_model.model.visual_tokenizer.dtype) if is_ovis else v[index],
                                                  device=data_device)
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, device=data_device)
            try:
                if self.gptq_model.ATTENTION_MASKS_DTYPE is torch.long:
                    example["attention_mask"] = example["attention_mask"].long()

                # Ensure initial caches (like RoPE) are created on the quant device
                with self.pool.read_lock(self.gptq_model.quantize_config.device):
                    with _device_ctx(self.gptq_model.quantize_config.device):
                        if self.gptq_model.INPUT_EMBEDDING_EXTRA_ARGS:
                            self.gptq_model.model.generate(**example, **self.gptq_model.INPUT_EMBEDDING_EXTRA_ARGS)
                        else:
                            self.gptq_model.model(**example, use_cache=use_cache)
            except StopForward:
                pass

        # LifeCycle: pre-first layer embedding hook
        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()

        return InputCache(layer_inputs=layer_inputs, layer_input_kwargs=layer_input_kwargs, position_ids=position_ids,
                          attention_masks=attention_masks)

    @torch.inference_mode()
    def loop(self, fail_safe: bool = False, **kwargs):
        if self.gptq_model.quantize_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError("quantization of `lm_head` layer with `tied_weights=True` model state is not supported. Please check model has `tied_weights=False`.")

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if get_module(self.gptq_model.model, key=self.gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = self.gptq_model.model.config.use_cache if hasattr(self.gptq_model.model.config, "use_cache") else False
        self.gptq_model.model.config.use_cache = False
        layers, layers_prefix = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.extract_layers_node())

        for p_index, processor in enumerate(self.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, EoraProcessor) or\
                        (isinstance(processor, GPTQProcessor) and self.gptq_model.quantize_config.v2):
                    prev_processor = self.processors[p_index - 1]
                    processor.set_calibration_dataset(prev_processor.calibration_dataset)
                    # If calibration_dataset is None or Empty, the input_cache of the previous processor is used.
                    processor.receive_input_cache(prev_processor.inputs_cache)
                elif isinstance(processor, DequantizeProcessor):
                    # DequantizeProcessor does not perform any operations on dataset.
                    processor.set_calibration_dataset([])
                    processor.receive_input_cache(InputCache([], [], [], []))

                continue

            input_cache = self.cache_inputs(layers=layers,
                                            calibration_data=processor.calibration_dataset,
                                            use_cache=False)
            processor.receive_input_cache(input_cache)

        # release calibration_dataset
        for processor in self.processors:
            processor.release_calibration_dataset()

        layer_modules = self.gptq_model.simple_layer_modules(model_config=self.gptq_model.model.config, quantize_config=self.gptq_model.quantize_config)

        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        layer_count = len(layers)
        quant_modules_pb = (log.pb(layer_count + 1 if self.gptq_model.quantize_config.lm_head else layer_count)
                            .manual()
                            .set(left_steps_offset=1))

        for processor in self.processors:
            processor.layer_count = layer_count
            processor.pb = quant_modules_pb

        shared_kv_cache_dict = {}

        replace_module_with_hooked_legacy(self.gptq_model.model, quant_lm_head=self.gptq_model.quantize_config.lm_head)

        if self.gptq_model.quantize_config.lm_head:
            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module and isinstance(lm_head_module, torch.nn.Linear):
                hooked_lm_head = HookedLinear.from_linear(lm_head_module)
                module_path = self.gptq_model.lm_head.split('.')
                parent = self.gptq_model.model
                for part in module_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, module_path[-1], hooked_lm_head)

        for layer_index in quant_modules_pb:
            is_lm_head_module = layer_index >= layer_count

            if is_lm_head_module:
                quant_modules_pb.title("Quantizing lm_head").draw()
                module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            else:
                quant_modules_pb.title(f"Quantizing layer {layer_index} of {layer_count - 1}").draw()
                module = layers[layer_index]

            self.gptq_model.wait_for_turtle_reload()

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue

            module = self.gptq_model.pre_quantize(module)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")

            for p_index, processor in enumerate(self.processors):
                processor.log_call_count = 0  # reset
                processor.collect_memory_info(layer_index)

                modules = [[self.gptq_model.lm_head]] if is_lm_head_module else layer_modules

                # for NativeProcessor we process one time forward on all grouped module subsets
                if processor.fwd_all_modules_in_single_pass:
                    # merge all subsets into one
                    modules = [sum(modules, [])]

                # AWQ does per-layer itself; skip here
                if isinstance(processor, AWQProcessor):
                    named_childs = dict()
                    for index, names in enumerate(modules):
                        named_modules = self.crate_named_modules(full=full,
                                                                 is_lm_head_module=is_lm_head_module,
                                                                 layer_index=layer_index, layers_prefix=layers_prefix,
                                                                 names=names,
                                                                 processor=processor,
                                                                 fail_safe=fail_safe)
                        named_childs.update(named_modules)
                    processor.layer_quantize(module, cur_layer_device, named_childs)
                    continue

                layer_inputs = processor.inputs_cache.layer_inputs
                if is_lm_head_module:
                    layer_inputs = self.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                processed_subset = {}

                for index, names in enumerate(modules):
                    subset = self.crate_named_modules(full=full, is_lm_head_module=is_lm_head_module,
                                                      layer_index=layer_index, layers_prefix=layers_prefix,
                                                      names=names,
                                                      processor=processor,
                                                      fail_safe=fail_safe)

                    if len(subset) == 0:
                        continue

                    handle = []
                    device_next_reset()

                    subset_size = len(subset)
                    for idx, (name, m) in enumerate(subset.items()):
                        is_last = (idx == subset_size - 1)

                        target_device = device_next()
                        m.target_device = target_device
                        m.module.target_device = target_device

                        # Wrap the processor hook with masking
                        if hasattr(subset[name], 'forward_hook'):
                            original_hook = processor.pre_process_fwd_hook(name)
                            subset[name].forward_hook = self._masked_hook_wrapper(processor, original_hook)
                            if is_last and processor.fwd_after_process:
                                subset[name].forward_hook_last = True
                        else:
                            # Older registration path
                            original_hook = processor.pre_process_fwd_hook(name)
                            handle.append(subset[name].register_forward_hook(
                                self._masked_hook_wrapper(processor, original_hook)
                            ))

                    # ---- Start Pre-Quantized Forward ----
                    fwd_start = time.time()

                    need_outputs = not processor.fwd_after_process
                    reuse_kv = bool(getattr(module, "reuse_kv", False))
                    forward_outputs = self._run_forward_batches(
                        module=module,
                        processor=processor,
                        layer_inputs=layer_inputs,
                        layer_input_kwargs=layer_input_kwargs,
                        position_ids=position_ids,
                        attention_masks=attention_masks,
                        cur_layer_device=cur_layer_device,
                        is_lm_head_module=is_lm_head_module,
                        shared_kv_cache_dict=shared_kv_cache_dict,
                        layer_index=layer_index,
                        need_outputs=need_outputs,
                        reuse_kv=reuse_kv,
                    )
                    if need_outputs:
                        processor.receive_layer_inputs(forward_outputs)
                        layer_inputs = processor.inputs_cache.layer_inputs
                        del forward_outputs

                    fwd_time = time.time() - fwd_start
                    processor.set_fwd_time(fwd_time)

                    for h in handle:
                        h.remove()

                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None
                            subset[name].forward_hook_last = False

                    # MoE coverage check for GPTQ
                    moe_skip_modules = []
                    if isinstance(processor, GPTQProcessor):
                        for name in subset:
                            if processor.tasks[name].fwd_counter == 0:
                                log.error(f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it.")
                                moe_skip_modules.append(name)

                        if not fail_safe:
                            for name in moe_skip_modules:
                                subset.pop(name)

                    # ---- Start Process Hook (via DeviceThreadPool) ----
                    futures = []

                    @torch.inference_mode()
                    def _process_on_worker(proc: LoopProcessor, nm: NamedModule):
                        # Run processor.process for this NamedModule
                        proc.process(module=nm)
                        return nm.name, nm

                    for name in subset:
                        m = subset[name]
                        # Prefer the planned target_device; fallback to module's own device
                        tgt_dev = getattr(m.module, "target_device", None) or get_device(m) or CPU
                        futures.append(self.pool.submit(tgt_dev, _process_on_worker, processor, m))

                    for fut in futures:
                        name, m = fut.result()
                        processed_subset[name] = m
                    torch_sync()
                    # ---- End Process Hook ----

                is_last_module = layer_index == len(quant_modules_pb) - 1
                layer_outputs: List[List[torch.Tensor]] = []
                # second forward after process()
                if not is_last_module and processor.fwd_after_process:
                    layer_outputs = self._run_forward_batches(
                        module=module,
                        processor=processor,
                        layer_inputs=layer_inputs,
                        layer_input_kwargs=layer_input_kwargs,
                        position_ids=position_ids,
                        attention_masks=attention_masks,
                        cur_layer_device=cur_layer_device,
                        is_lm_head_module=is_lm_head_module,
                        shared_kv_cache_dict=shared_kv_cache_dict,
                        layer_index=layer_index,
                        need_outputs=True,
                        reuse_kv=False,
                    )

                # Finalize module after last processor
                if p_index == len(self.processors) - 1:
                    torch_sync()

                    if not is_lm_head_module:
                        layers[layer_index] = self.gptq_model.post_quantize(module)
                    else:
                        self.gptq_model.post_quantize(module)

                if processor.fwd_after_process:
                    processor.clear_cache_data()
                    processor.receive_layer_inputs(layer_outputs)
                    layer_inputs = processor.inputs_cache.layer_inputs

                if p_index == len(self.processors) - 1:
                    torch_sync()

                    # Gather finalize tasks (can offload to disk); run them via the pool
                    finalize_futures = []

                    for reverse_p in reversed(self.processors):
                        for name in processed_subset:
                            @torch.inference_mode()
                            def finalize_module(process, module):
                                process.submodule_finalize(module, self.gptq_model)

                                # Disk offload (lifecycle TODO note preserved)
                                if isinstance(process, (GPTQProcessor, QQQProcessor, AWQProcessor)):
                                    offload_to_disk(
                                        model=self.gptq_model.model,
                                        module=self.gptq_model.model.get_submodule(module.full_name),
                                        disk_path=self.gptq_model.quantize_config.offload_to_disk_path,
                                    )

                            module = processed_subset[name]

                            target_dev = get_device_new(module, recursive=True, assert_mode=True, expected="cpu")

                            # Submit on the module's device thread (safe & deterministic)
                            finalize_futures.append(
                                self.pool.submit(target_dev, finalize_module, reverse_p, module)
                            )

                    # If any finalize tasks were queued, wait for them
                    for fut in finalize_futures:
                        fut.result()

        # LifeCycle: All sub-modules have finalized meaning quantization work is complete
        # Ensure ANY remaining tasks the looper submitted have drained
        self.pool.wait()  # same as wait('all')

        self.gptq_model.wait_for_turtle_reload()

        # paranoid safety check
        # torch_sync()
        # torch_sync(device=CPU)

        total_log = {}

        for reverse_p in reversed(self.processors):
            if isinstance(reverse_p, GPTQProcessor):
                pass
            elif isinstance(reverse_p, EoraProcessor):
                pass
            elif isinstance(reverse_p, DequantizeProcessor):
                pass
            else:
                log.info(f"{reverse_p.name()} summary:\n{reverse_p.log}")

            processor_name = reverse_p.name()
            total_log[processor_name] = reverse_p.log
            if processor_name in ["gptq", "gptq v2"]:
                self.gptq_model.quant_log = reverse_p.log

            for module_log in reverse_p.log:
                log.info(module_log)
            reverse_p.log_plotly()

            reverse_p.finalize(model=self.gptq_model, **kwargs)

        self.gptq_model.model.config.use_cache = forward_pass_use_cache

        return total_log

    def crate_named_modules(self, full, is_lm_head_module, layer_index, layers_prefix, names, processor, fail_safe) -> Dict[str, NamedModule]:
        is_awq_quant = isinstance(processor, AWQProcessor)
        subset = {}
        for n in names:
            if n in full:
                subset[n] = full[n]
            # some modules have layer_modules that are dynamic based on config
            # ref: deepseek v2/v3/r1
            elif self.gptq_model.layer_modules_strict:
                raise ValueError(f"layer module item `{n}` not found in model, please check your model config.")
        skipped_modules = []
        for name in subset:
            layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layers_prefix}.{layer_index}.{name}"

            # gptq task is created and stored inside processor
            if not isinstance(subset[name], NamedModule):
                named_module = NamedModule(subset[name], name=name, full_name=layer_name,
                                           layer_index=layer_index)
                if isinstance(processor, EoraProcessor):
                    named_module.state.update({
                        "wq": processor.quantized_weights[layer_name],
                    })

                subset[name] = named_module
                full[name] = named_module

            if not is_awq_quant:
                if isinstance(processor, GPTQProcessor):
                    processor.preprocess(subset[name], fail_safe=fail_safe)
                else:
                    processor.preprocess(subset[name])
                # some modules are skipped
                if processor.is_skipped(subset[name]):
                    skipped_modules.append(name)

        if not is_awq_quant:
            for name in skipped_modules:
                subset.pop(name)
        return subset
