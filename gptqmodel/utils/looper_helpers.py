# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from __future__ import annotations

import copy
import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import parallel as torch_parallel

from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..nn_modules.hooked_linear import StopForward
from ..utils.attn_mask import normalize_seq_mask
from ..utils.device import get_device
from ..utils.env import env_flag
from ..utils.logger import setup_logger
from ..utils.model import move_to, nested_move_to
from ..utils.safe import ThreadSafe
from ..utils.torch import ALL_DEVICES, CPU, torch_sync


USE_TORCH_REPLICATE = env_flag("GPTQMODEL_USE_TORCH_REPLICATE", True)


_THREAD_SAFE_PARALLEL = ThreadSafe(torch_parallel)
_DEEPCOPY_LOCK = threading.Lock()

def torch_replicate(
    module: torch.nn.Module,
    devices: Sequence[torch.device | str | int],
    detach: bool = True,
):
    """Replicate a module across ``devices`` while coordinating device locks and syncs.

    Clones default to ``detach=True`` because quantization workflows operate in inference-only
    mode and do not need autograd edges.
    """
    normalized_devices: List[torch.device] = []
    for candidate in devices:
        normalized = normalize_device_like(candidate)
        if normalized is None:
            raise ValueError(f"Unsupported device spec: {candidate!r}")
        normalized_devices.append(normalized)

    lock_scope = normalized_devices or None

    with DEVICE_THREAD_POOL.lock(lock_scope):
        for dev in normalized_devices:
            if dev.type in {"cuda", "xpu", "mps", "npu"}:
                try:
                    torch_sync(dev)
                except BaseException:
                    pass
        return _THREAD_SAFE_PARALLEL.replicate(module, normalized_devices, detach=detach)

log = setup_logger()


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..looper.loop_processor import LoopProcessor
    from ..models._const import DEVICE


__all__ = [
    "device_ctx",
    "rehome_module_to_device",
    "clear_non_picklable_state",
    "select_forward_devices",
    "normalize_device_like",
    "clone_module_for_devices",
    "forward_batch_worker",
]


@contextmanager
def device_ctx(dev: Optional[torch.device | "DEVICE"]):
    """Temporarily set the thread-local device for CUDA/XPU backends."""
    if dev is None:
        yield
        return

    if dev.type == "cuda":
        with torch.cuda.device(dev.index):
            yield
        return
    if dev.type == "xpu" and hasattr(torch, "xpu"):
        with torch.xpu.device(dev.index):  # type: ignore[attr-defined]
            yield
        return

    # cpu/mps/meta -> nothing special needed
    yield

_rehome_lock = threading.Lock()

@torch.inference_mode()
def rehome_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    *,
    move_parameters: bool = False,
    move_buffers: bool = True,
    include_non_persistent_buffers: bool = True,
    only_mismatched: bool = True,
) -> None:
    """Move registered tensors on ``module`` to ``device`` with defensive fallbacks."""
    with _rehome_lock:
        for sub in module.modules():
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
                                requires_grad=p.requires_grad,
                            )
                        sub._parameters[pname] = new_p
                    except Exception:
                        try:
                            with torch.no_grad():
                                new_p = torch.nn.Parameter(
                                    p.data.to(device),
                                    requires_grad=p.requires_grad,
                                )
                            sub._parameters[pname] = new_p
                        except Exception:
                            pass


def clear_non_picklable_state(module: torch.nn.Module) -> List[Tuple[str, int]]:
    """Placeholder hook that tracks visited modules; extended elsewhere as needed."""
    cleared: List[Tuple[str, int]] = []
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


def _canonical_device(device: torch.device) -> torch.device:
    """Return a canonical form so indexless accelerators collapse to device:0."""
    if device.type in {"cuda", "xpu", "npu"}:
        index = device.index if device.index is not None else 0
        return torch.device(f"{device.type}:{index}")
    return device


def select_forward_devices(base_device: Optional[torch.device]) -> List[torch.device]:
    if base_device is None:
        return [CPU]

    devices: List[torch.device] = []
    seen: set[tuple[str, int | None]] = set()

    def _add(device: torch.device) -> None:
        canonical = _canonical_device(device)
        key = (canonical.type, canonical.index)
        if key in seen:
            return
        seen.add(key)
        devices.append(canonical)

    _add(base_device)
    base_type = devices[0].type
    if base_type in {"cuda", "xpu", "mps", "npu"}:
        for dev in ALL_DEVICES:
            if dev.type == base_type:
                _add(dev)
    return devices


def normalize_device_like(device_like) -> Optional[torch.device]:
    if device_like is None:
        return None
    if isinstance(device_like, torch.device):
        device = device_like
    elif hasattr(device_like, "to_torch_device"):
        device = device_like.to_torch_device()
    else:
        device = torch.device(str(device_like))

    return _canonical_device(device)


def clone_module_for_devices(
    module: torch.nn.Module,
    devices: List[torch.device],
    *,
    clear_state_fn=clear_non_picklable_state,
    progress_callback: Optional[Callable[[int, int, torch.device, str], None]] = None,
) -> Dict[torch.device, torch.nn.Module]:
    clones: Dict[torch.device, torch.nn.Module] = {}
    if not devices:
        return clones

    module_label = getattr(module, "full_name", module.__class__.__name__)
    clone_timings: List[Tuple[str, float]] = []
    overall_start = time.perf_counter()

    total_targets = len(devices)

    def _notify(idx: int, device: torch.device, step: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(idx, total_targets, device, step)
        except Exception:
            if DEBUG_ON:
                log.debug(
                    "clone_module_for_devices: progress callback failed (device=%s, step=%s)",
                    device,
                    step,
                )

    def _record(name: str, start_ts: Optional[float]) -> None:
        if not DEBUG_ON or start_ts is None:
            return
        clone_timings.append((name, time.perf_counter() - start_ts))

    def _emit_clone_log(method: str) -> None:
        if not DEBUG_ON:
            return
        total_duration = (time.perf_counter() - overall_start) * 1000.0
        if clone_timings:
            timing_str = ", ".join(f"{step}={duration * 1000:.2f}ms" for step, duration in clone_timings)
            log.info(f"ModuleLooper: clone {module_label} via {method} in {total_duration:.2f}ms [{timing_str}]")
        else:
            log.info(f"ModuleLooper: clone {module_label} via {method} in {total_duration:.2f}ms")

    base_device = devices[0]
    device_type = base_device.type
    homogeneous_type = all(dev.type == device_type for dev in devices)

    def backend_available(dev_type: str) -> bool:
        if dev_type == "cuda":
            return torch.cuda.is_available()
        if dev_type == "xpu" and hasattr(torch, "xpu"):
            return bool(getattr(torch.xpu, "is_available", lambda: False)())  # type: ignore[attr-defined]
        if dev_type == "mps":
            return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return False

    def _prepare_module(target_device: torch.device, step_name: str) -> None:
        start_ts = time.perf_counter()
        module.to(target_device)
        module.eval()
        rehome_module_to_device(module, target_device, move_parameters=True, move_buffers=True)
        clear_state_fn(module)
        setattr(module, "_gptqmodule_device_hint", target_device)
        _record(step_name, start_ts)

    use_replicate = (
        USE_TORCH_REPLICATE
        and homogeneous_type
        and backend_available(device_type)
        and device_type != "cpu"
    )

    stage_device = base_device if base_device.type != "cpu" else CPU

    if use_replicate:
        try:
            _prepare_module(base_device, f"stage_{base_device}")
            _notify(0, base_device, "stage")

            replicate_start = time.perf_counter()
            replicas = torch_replicate(module, devices)
            _record("replicate", replicate_start)

            for idx, (dev, replica) in enumerate(zip(devices, replicas), start=1):
                replica.eval()
                rehome_module_to_device(replica, dev, move_parameters=True, move_buffers=True)
                clear_state_fn(replica)
                setattr(replica, "_gptqmodule_device_hint", dev)
                clones[dev] = replica
                _notify(idx, dev, "replica")

            _emit_clone_log("replicate")
            return clones
        except Exception as e:
            log.info(f"Clone: fast clone failed {e}")
            clone_timings.append(("replicate_failed", 0.0))
            if stage_device != base_device:
                _prepare_module(stage_device, f"stage_{stage_device}")

    if len(devices) == 1 and devices[0].type == "cpu":
        _prepare_module(CPU, "stage_cpu")
        _notify(0, CPU, "stage")
        clones[devices[0]] = module
        _notify(1, devices[0], "reuse")
        _emit_clone_log("reuse")
        return clones

    if not use_replicate:
        _prepare_module(stage_device, f"stage_{stage_device}")
        _notify(0, stage_device, "stage")

    for idx, dev in enumerate(devices, start=1):
        start_ts = time.perf_counter()
        with _DEEPCOPY_LOCK:
            replica = copy.deepcopy(module)
        replica.eval()
        rehome_module_to_device(replica, dev, move_parameters=True, move_buffers=True)
        clear_state_fn(replica)
        setattr(replica, "_gptqmodule_device_hint", dev)
        clones[dev] = replica
        _record(str(dev), start_ts)
        _notify(idx, dev, "clone")

    _emit_clone_log("deepcopy")
    return clones


@torch.inference_mode()
def forward_batch_worker(
    module: torch.nn.Module,
    processor: "LoopProcessor",
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
    processor._set_current_batch_index(batch_index)
    module_device = getattr(module, "_gptqmodule_device_hint", None) or get_device(module)
    rehome_module_to_device(module, module_device, move_parameters=True, move_buffers=True)

    torch_sync() # try to avoid torch.AcceleratorError: CUDA error: unspecified launch failure
    inputs = [move_to(inp, device=module_device) for inp in layer_input]

    attn_tensor = None
    if attention_mask is not None:
        attn_tensor = move_to(attention_mask, device=module_device)

    additional_inputs: Dict[str, torch.Tensor] = {}
    if support_batch_quantize and attn_tensor is not None:
        additional_inputs["attention_mask"] = attn_tensor

    if position_ids is not None:
        additional_inputs["position_ids"] = move_to(position_ids, device=module_device)

    for key, value in layer_input_kwargs.items():
        additional_inputs[key] = nested_move_to(value, device=module_device)

    keep_mask = None
    if attn_tensor is not None:
        seq_len = inputs[0].shape[1] if (len(inputs) > 0 and inputs[0].dim() >= 2) else None
        keep_mask = normalize_seq_mask(attn_tensor, seq_len=seq_len)

    mask_tls = getattr(processor, "_mask_tls", None)
    if mask_tls is not None:
        mask_tls.value = keep_mask

    if reuse_kv and prev_kv is not None:
        additional_inputs["kv_last_layer"] = nested_move_to(prev_kv, device=module_device)

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
        processor._set_current_batch_index(None)

    if reuse_kv and module_output is not None and isinstance(module_output, tuple) and len(module_output) > 0:
        kv_next = module_output[-1]

    result_output = module_output if need_output else None
    return batch_index, result_output, kv_next
