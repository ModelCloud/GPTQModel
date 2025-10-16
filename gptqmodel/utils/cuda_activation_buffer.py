# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from typing import Any, Callable, List, Optional

import torch


__all__ = ["ActivationPacket", "CudaEventActivationBuffer"]


@dataclasses.dataclass(slots=True)
class ActivationPacket:
    """
    Tracks a single async device->host transfer triggered from a forward hook.

    The event is recorded on the dedicated copy stream so the consumer can
    decide when to block. The `host_tensor` already points at pinned memory.
    """

    event: torch.cuda.Event
    host_tensor: torch.Tensor
    meta: Optional[Any] = None
    created_at: float = dataclasses.field(default_factory=time.perf_counter)


class CudaEventActivationBuffer:
    """
    Schedules non-blocking GPU->CPU copies using a dedicated CUDA stream + event.

    Typical usage inside a forward hook::

        buffer = CudaEventActivationBuffer(device="cuda:6")

        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            buffer.capture_async(tensor, meta=module.__class__.__name__)

        # elsewhere in consumer thread
        for packet in buffer.drain():
            packet.event.synchronize()
            process(packet.host_tensor, packet.meta)

    The hook thread returns immediately after enqueuing the async copy which
    allows the caller to release activation VRAM without waiting on D2H traffic.
    """

    def __init__(
        self,
        device: torch.device | str | int,
        stream: Optional[torch.cuda.Stream] = None,
        pin_memory: bool = True,
        host_allocator: Optional[Callable[[torch.Size, torch.dtype, torch.layout], torch.Tensor]] = None,
        host_reclaimer: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for CudaEventActivationBuffer.")

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"CudaEventActivationBuffer requires a CUDA device, got {dev}.")

        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())

        self._device = dev
        self._pin_memory = pin_memory
        self._host_allocator = host_allocator
        self._host_reclaimer = host_reclaimer

        with torch.cuda.device(self._device):
            self._copy_stream = stream or torch.cuda.Stream()

        self._pending: "queue.SimpleQueue[ActivationPacket]" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._approx_pending = 0

    def capture_async(
        self,
        activation: torch.Tensor,
        *,
        meta: Any = None,
        enqueue: bool = True,
    ) -> ActivationPacket:
        """
        Enqueue an async D2H copy of ``activation`` onto the buffer stream.

        Returns an ActivationPacket which is also available later via drain().
        """
        if activation.device != self._device:
            raise ValueError(
                f"Activation tensor is on {activation.device}, expected {self._device}."
            )

        activation = activation.detach()
        if not activation.is_contiguous():
            activation = activation.contiguous()

        host = self._allocate_host(activation)

        event = torch.cuda.Event(blocking=False, interprocess=False)

        current = torch.cuda.current_stream(self._device)
        copy_stream = self._copy_stream
        copy_stream.wait_stream(current)

        with torch.cuda.stream(copy_stream):
            host.copy_(activation, non_blocking=True)
            event.record(copy_stream)

        packet = ActivationPacket(event=event, host_tensor=host, meta=meta)
        if enqueue:
            self._pending_put(packet)
        return packet

    def drain(self, *, wait: bool = True, max_items: Optional[int] = None) -> List[ActivationPacket]:
        """
        Collect all queued packets (or up to max_items) in FIFO order.

        When ``wait`` is True we synchronize each packet's event before returning.
        """
        packets: List[ActivationPacket] = []
        pulled = 0

        while True:
            if max_items is not None and pulled >= max_items:
                break

            try:
                packet = self._pending_get()
            except queue.Empty:
                break

            pulled += 1
            if wait:
                packet.event.synchronize()
            packets.append(packet)

        return packets

    def recycle(self, packet: ActivationPacket) -> None:
        """
        Return a packet's host buffer to the allocator pool (if provided).
        """
        if self._host_reclaimer is not None:
            self._host_reclaimer(packet.host_tensor)

    def pending_count(self) -> int:
        """
        Non-blocking length check. The SimpleQueue does not expose qsize()
        reliably on all platforms, so we track with a lock-protected counter.
        """
        with self._lock:
            count = getattr(self, "_approx_pending", 0)
        return count

    def __len__(self) -> int:
        return self.pending_count()

    def __enter__(self) -> "CudaEventActivationBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.drain(wait=True)

    def _pending_put(self, packet: ActivationPacket) -> None:
        with self._lock:
            self._approx_pending = getattr(self, "_approx_pending", 0) + 1
        self._pending.put(packet)

    def _pending_get(self) -> ActivationPacket:
        packet = self._pending.get_nowait()
        with self._lock:
            self._approx_pending = max(getattr(self, "_approx_pending", 0) - 1, 0)
        return packet

    def _allocate_host(self, activation: torch.Tensor) -> torch.Tensor:
        if self._host_allocator is not None:
            host = self._host_allocator(activation.shape, activation.dtype, activation.layout)
            if not host.is_pinned():
                raise ValueError("Custom host allocator must return pinned CPU tensors.")
            return host
        return torch.empty(
            activation.shape,
            dtype=activation.dtype,
            layout=activation.layout,
            device="cpu",
            pin_memory=self._pin_memory,
        )
