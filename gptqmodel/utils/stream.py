# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .logger import setup_logger
from .threadx import DeviceThreadPool
from .torch import HAS_NPU


log = setup_logger()


@dataclass
class StreamCopyTicket:
    event: Optional[Any]
    device: Optional[torch.device]
    keys: Tuple[str, ...]
    sources: Optional[List[torch.Tensor]]
    stream: Optional[Any]
    future: Optional[Future] = None
    finalized: bool = False
    background_done: bool = False


def _discover_accelerator_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            aliases[f"cuda:{idx}"] = f"stream-sync-cuda{idx}"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for idx in range(torch.xpu.device_count()):
            aliases[f"xpu:{idx}"] = f"stream-sync-xpu{idx}"
    if HAS_NPU:
        for idx in range(torch.npu.device_count()):
            aliases[f"npu:{idx}"] = f"stream-sync-npu{idx}"
    return aliases


def _build_stream_pool() -> Tuple[DeviceThreadPool, Dict[str, str]]:
    aliases = _discover_accelerator_aliases()
    total_accelerators = len(aliases)
    workers: Dict[str, int] = {}
    for alias in aliases.values():
        workers[f"{alias}:cpu"] = 1

    # One CPU worker per accelerator (at least one worker overall).
    workers["cpu"] = max(1, total_accelerators or 1)

    pool = DeviceThreadPool(
        devices=[torch.device("cpu")],
        include_cuda=False,
        include_xpu=False,
        include_npu=False,
        include_mps=False,
        include_cpu=True,
        inference_mode=True,
        workers=workers,
    )
    return pool, aliases


STREAM_DEVICE_POOL, _ACCELERATOR_ALIAS_MAP = _build_stream_pool()

_STREAM_CACHE_LOCK = threading.RLock()
_COPY_STREAMS: Dict[Tuple[str, int], Any] = {}


def _stream_backend(device_type: str):
    if device_type == "cuda":
        return torch.cuda
    if device_type == "npu":
        return torch.npu
    return None


def _stream_backend_available(device_type: str) -> bool:
    backend = _stream_backend(device_type)
    if backend is None:
        return False
    return bool(backend.is_available())


def _resolve_device_index(device: torch.device) -> int:
    index = device.index
    if index is not None:
        return index
    backend = _stream_backend(device.type)
    if backend is not None:
        return backend.current_device()
    return 0

# reuse streams instead of creating tons of new streams
def _get_cached_copy_stream(device: torch.device) -> Any:
    idx = _resolve_device_index(device)
    with _STREAM_CACHE_LOCK:
        key = (device.type, idx)
        stream = _COPY_STREAMS.get(key)
        if stream is None:
            backend = _stream_backend(device.type)
            if backend is None:
                raise RuntimeError(f"Unsupported stream copy device: {device}")
            stream = backend.Stream(device=torch.device(device.type, idx))
            _COPY_STREAMS[key] = stream
        return stream


def _new_event(device_type: str) -> Any:
    backend = _stream_backend(device_type)
    if backend is None:
        return None
    try:
        return backend.Event(enable_timing=False, blocking=False)
    except TypeError:
        return backend.Event(enable_timing=False)


def _queue_key_for_device(device: Optional[torch.device]) -> str:
    if device is None:
        return "cpu"
    dev_type = device.type
    if dev_type == "cuda":
        index = device.index
        if index is None and torch.cuda.is_available():
            try:
                index = torch.cuda.current_device()
            except Exception:
                index = 0
        return _ACCELERATOR_ALIAS_MAP.get(f"cuda:{index}", "cpu")
    if dev_type == "xpu" and hasattr(torch, "xpu"):
        index = device.index
        if index is None and torch.xpu.is_available():
            try:
                index = torch.xpu.current_device()
            except Exception:
                index = 0
        return _ACCELERATOR_ALIAS_MAP.get(f"xpu:{index}", "cpu")
    if dev_type == "npu" and HAS_NPU:
        index = device.index
        if index is None:
            try:
                index = torch.npu.current_device()
            except Exception:
                index = 0
        return _ACCELERATOR_ALIAS_MAP.get(f"npu:{index}", "cpu")
    return "cpu"


def _drop_sources(ticket: StreamCopyTicket) -> None:
    if ticket.sources is not None:
        ticket.sources.clear()
        ticket.sources = None


def _finalize_ticket_locked(
    ticket: StreamCopyTicket,
    state: Dict[str, Any],
) -> None:
    if ticket.finalized:
        return
    ticket.finalized = True
    _drop_sources(ticket)
    event_map = state.get("streaming_event_map")
    if event_map is not None:
        for key in ticket.keys:
            event_map.pop(key, None)
    events = state.get("streaming_events")
    if isinstance(events, list):
        try:
            events.remove(ticket)
        except ValueError:
            pass
    ticket.stream = None
    ticket.future = None


def _wait_and_release_ticket(
    ticket: StreamCopyTicket,
    state: Dict[str, Any],
    state_lock: threading.RLock,
) -> None:
    with state_lock:
        if ticket.background_done:
            return
    try:
        event = ticket.event
        if event is not None and not event.query():
            event.synchronize()
    finally:
        with state_lock:
            _drop_sources(ticket)
            ticket.background_done = True
        ticket.stream = None


def _schedule_ticket(
    ticket: StreamCopyTicket,
    state: Dict[str, Any],
    state_lock: threading.RLock,
) -> None:
    queue_key = _queue_key_for_device(ticket.device)
    try:
        future = STREAM_DEVICE_POOL.submit(
            queue_key,
            _wait_and_release_ticket,
            ticket,
            state,
            state_lock,
        )
        ticket.future = future
    except Exception:
        log.exception("Failed to schedule stream sync ticket on queue %s; falling back to inline sync", queue_key)
        ticket.future = None
        _wait_and_release_ticket(ticket, state, state_lock)


def stream_tensor_dict_to_cpu(
    tensors: Dict[str, torch.Tensor],
    *,
    store_callback: Callable[[Dict[str, torch.Tensor]], None],
    state: Dict[str, Any],
    state_lock: threading.RLock,
) -> Dict[str, torch.Tensor]:
    filtered = {name: tensor for name, tensor in tensors.items() if isinstance(tensor, torch.Tensor)}
    if not filtered:
        return {}

    # sync copy
    # host_map = {name: tensor.detach().to("cpu") for name, tensor in filtered.items()}
    # with state_lock:
    #     store_callback(host_map)
    # return host_map

    first_accelerator = next(
        (
            tensor
            for tensor in filtered.values()
            if tensor.device.type in {"cuda", "npu"} and _stream_backend_available(tensor.device.type)
        ),
        None,
    )

    if first_accelerator is None:
        host_map = {name: tensor.detach().to("cpu") for name, tensor in filtered.items()}
        with state_lock:
            store_callback(host_map)
        return host_map

    host_map: Dict[str, torch.Tensor] = {}

    copy_device = first_accelerator.device
    backend = _stream_backend(copy_device.type)
    compute_stream = backend.current_stream(device=copy_device)
    copy_stream = _get_cached_copy_stream(copy_device)

    pending_sources: List[torch.Tensor] = []
    pending_keys: List[str] = []
    with backend.stream(copy_stream):
        copy_stream.wait_stream(compute_stream)
        for name, tensor in filtered.items():
            src = tensor.detach()
            if src.device.type != copy_device.type:
                host_map[name] = src.to("cpu")
                continue
            if src.device != copy_device:
                host_map[name] = src.to("cpu")
                continue
            src.record_stream(copy_stream)
            pending_sources.append(src)
            pending_keys.append(name)
            host = torch.empty(
                src.shape,
                dtype=src.dtype,
                layout=src.layout,
                device="cpu",
                pin_memory=True,
            )
            host.copy_(src, non_blocking=True)
            host_map[name] = host

    if not pending_sources:
        with state_lock:
            store_callback(host_map)
        return host_map

    done_event = _new_event(copy_device.type)
    done_event.record(copy_stream)

    ticket = StreamCopyTicket(
        event=done_event,
        device=copy_device,
        keys=tuple(pending_keys),
        sources=pending_sources,
        stream=copy_stream,
    )

    with state_lock:
        store_callback(host_map)
        event_map = state.setdefault("streaming_event_map", {})
        for key in ticket.keys:
            event_map[key] = done_event
        events = state.setdefault("streaming_events", [])
        events.append(ticket)

    _schedule_ticket(ticket, state, state_lock)
    return host_map


def stream_sync(state: Dict[str, Any], state_lock: threading.RLock) -> None:
    while True:
        with state_lock:
            pending: Iterable[StreamCopyTicket] = tuple(state.get("streaming_events", ()))
        if not pending:
            break
        for ticket in pending:
            future = ticket.future
            if future is not None:
                future.result()
            else:
                event = ticket.event
                if event is not None and not event.query():
                    event.synchronize()
            with state_lock:
                if not ticket.background_done:
                    _drop_sources(ticket)
                    ticket.background_done = True
                _finalize_ticket_locked(ticket, state)
