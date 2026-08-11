# SPDX-License-Identifier: Apache-2.0
"""Bounded ordered replay for accuracy-preserving parallel calibration forwards."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence

import torch

if TYPE_CHECKING:
    from ..quantization.gptq import GPTQ
    from .gptq_processor import GPTQProcessor


class ParallelForwardWaveReplay(Protocol):
    """Optional processor interface consumed by :class:`ForwardExecutor`."""

    def begin_parallel_forward_wave(
        self,
        batch_indices: Sequence[int],
        canonical_device: torch.device,
    ) -> None: ...

    def seal_parallel_forward_batch(self, batch_index: int, device: torch.device) -> None: ...

    def flush_parallel_forward_wave(self) -> None: ...

    def abort_parallel_forward_wave(self) -> None: ...


@dataclass
class OrderedReplayEntry:
    """One hook-side operation deferred until canonical ordered replay."""

    batch_index: int
    sequence: int
    replay: Callable[[torch.device, dict[tuple[object, ...], torch.Tensor]], None]
    retained_bytes: int = 0


def tensor_storage_key(tensor: torch.Tensor) -> tuple[object, ...]:
    """Identify one exact tensor view so repeated logical inputs transfer once."""

    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


class OrderedHessianReplayBlock:
    """Retain one parallel wave and replay its Hessian updates in batch order.

    The block never owns a thread pool. Forward workers seal their CUDA work on
    the process-wide per-device ThreadX workers; the executor submits ``flush``
    to the canonical device's same serial worker.
    """

    def __init__(self, *, maximum_retained_bytes_per_device: int = 12 * 1024**3):
        self.maximum_retained_bytes_per_device = int(maximum_retained_bytes_per_device)
        self._lock = threading.Lock()
        self._active = False
        self._canonical_device: Optional[torch.device] = None
        self._batch_order: tuple[int, ...] = ()
        self._entries: dict[int, list[OrderedReplayEntry]] = {}
        self._events: dict[int, torch.cuda.Event] = {}
        self._retained_storage: dict[tuple[object, ...], int] = {}
        self._retained_bytes_by_device: dict[torch.device, int] = {}
        self._sequence = 0

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def begin(self, batch_indices: Sequence[int], canonical_device: torch.device) -> None:
        batch_order = tuple(int(index) for index in batch_indices)
        if not batch_order:
            raise ValueError("Ordered Hessian replay requires at least one batch index.")
        if batch_order != tuple(sorted(batch_order)):
            raise ValueError(f"Ordered Hessian replay requires increasing batch indices, got {batch_order}.")
        if len(batch_order) > 1 and any(right != left + 1 for left, right in zip(batch_order, batch_order[1:])):
            raise ValueError(f"Ordered Hessian replay requires consecutive batch indices, got {batch_order}.")

        with self._lock:
            if self._active:
                raise RuntimeError("Ordered Hessian replay wave is already active.")
            self._active = True
            self._canonical_device = torch.device(canonical_device)
            self._batch_order = batch_order
            self._entries = {index: [] for index in batch_order}
            self._events.clear()
            self._retained_storage.clear()
            self._retained_bytes_by_device.clear()
            self._sequence = 0

    def retain(self, tensor: torch.Tensor) -> torch.Tensor:
        """Retain a native-dtype tensor view and account unique backing storage."""

        value = tensor.detach()
        key = tensor_storage_key(value)
        with self._lock:
            if not self._active:
                raise RuntimeError("Cannot retain a tensor outside an active replay wave.")
            if key not in self._retained_storage:
                retained_bytes = value.untyped_storage().nbytes()
                device = torch.device(value.device)
                total = self._retained_bytes_by_device.get(device, 0) + retained_bytes
                if total > self.maximum_retained_bytes_per_device:
                    raise RuntimeError(
                        "Ordered Hessian replay exceeded its per-device retained-activation budget: "
                        f"device={device}, requested={total / 1024**3:.2f} GiB, "
                        f"limit={self.maximum_retained_bytes_per_device / 1024**3:.2f} GiB."
                    )
                self._retained_storage[key] = retained_bytes
                self._retained_bytes_by_device[device] = total
        return value

    def append(
        self,
        batch_index: int,
        replay: Callable[[torch.device, dict[tuple[object, ...], torch.Tensor]], None],
    ) -> None:
        with self._lock:
            if not self._active or batch_index not in self._entries:
                raise RuntimeError(f"Batch {batch_index} is not part of the active ordered replay wave.")
            sequence = self._sequence
            self._sequence += 1
            self._entries[batch_index].append(
                OrderedReplayEntry(batch_index=batch_index, sequence=sequence, replay=replay)
            )

    def seal_batch(self, batch_index: int, device: torch.device) -> None:
        """Record completion on the forward worker's current device stream."""

        device = torch.device(device)
        with self._lock:
            if not self._active or batch_index not in self._entries:
                return
        if device.type != "cuda":
            return
        with torch.cuda.device(device):
            event = torch.cuda.Event(enable_timing=False, blocking=False)
            event.record(torch.cuda.current_stream(device))
        with self._lock:
            self._events[batch_index] = event

    def flush(self) -> None:
        """Replay one complete wave on the canonical device and release it."""

        with self._lock:
            if not self._active or self._canonical_device is None:
                return
            canonical_device = self._canonical_device
            batch_order = self._batch_order
            entries = {index: tuple(self._entries[index]) for index in batch_order}
            events = dict(self._events)

        transfer_cache: dict[tuple[object, ...], torch.Tensor] = {}
        try:
            for batch_index in batch_order:
                event = events.get(batch_index)
                if event is not None:
                    event.synchronize()
                for entry in entries[batch_index]:
                    entry.replay(canonical_device, transfer_cache)
                transfer_cache.clear()
        finally:
            transfer_cache.clear()
            self.abort()

    def abort(self) -> None:
        """Release all retained references after failure or successful replay."""

        with self._lock:
            self._active = False
            self._canonical_device = None
            self._batch_order = ()
            self._entries.clear()
            self._events.clear()
            self._retained_storage.clear()
            self._retained_bytes_by_device.clear()
            self._sequence = 0


def move_replay_tensor(
    tensor: torch.Tensor,
    device: torch.device,
    transfer_cache: dict[tuple[object, ...], torch.Tensor],
) -> torch.Tensor:
    """Move one retained view once per wave/batch replay cache."""

    key = tensor_storage_key(tensor)
    moved = transfer_cache.get(key)
    if moved is None:
        moved = tensor if tensor.device == device else tensor.to(device=device)
        transfer_cache[key] = moved
    return moved


class GPTQOrderedHessianReplayAttachment:
    """Attach ordered wave replay to GPTQProcessor's stable capture seams."""

    def __init__(self, processor: "GPTQProcessor"):
        self.processor = processor
        self.block = OrderedHessianReplayBlock()

    @property
    def active(self) -> bool:
        return self.block.active

    def defer_batch(
        self,
        task: "GPTQ",
        inp: torch.Tensor,
        *,
        batch_index: Optional[int],
        cache_source: torch.Tensor,
        cache_extra: Optional[tuple[object, ...]],
    ) -> tuple[int, int]:
        """Retain one native-dtype observation for canonical ordered replay."""

        if batch_index is None:
            raise RuntimeError("Ordered Hessian replay requires an explicit calibration batch index.")
        sequence_count = task._sequence_count_for_input(inp)
        batch_token_size, reshaped, _ = task._reshape_input(inp)
        del reshaped
        retained_input = self.block.retain(inp)
        retained_cache_source = retained_input if cache_source is inp else self.block.retain(cache_source)

        def replay_batch(canonical_device, transfer_cache) -> None:
            moved_input = move_replay_tensor(retained_input, canonical_device, transfer_cache)
            moved_cache_source = (
                moved_input
                if retained_cache_source is retained_input
                else move_replay_tensor(retained_cache_source, canonical_device, transfer_cache)
            )
            self.processor._add_batch_with_shared_hessian_immediate(
                task,
                moved_input,
                None,
                batch_index=batch_index,
                cache_source=moved_cache_source,
                cache_extra=cache_extra,
            )

        self.block.append(batch_index, replay_batch)
        return (int(batch_token_size), int(sequence_count))

    def defer_followers(
        self,
        *,
        source_name: str,
        follower_names: list[str],
        batch_token_size: int,
        observation_count: int,
        sequence_count: int,
    ) -> None:
        """Queue follower ownership directly after its source observation."""

        batch_index = self.processor.current_batch_index()
        if batch_index is None:
            raise RuntimeError("Ordered MoE follower replay requires an explicit calibration batch index.")

        def replay_followers(_canonical_device, _transfer_cache) -> None:
            self.processor._record_moe_shared_input_followers_immediate(
                source_name=source_name,
                follower_names=follower_names,
                batch_token_size=batch_token_size,
                observation_count=observation_count,
                sequence_count=sequence_count,
                batch_index=batch_index,
            )

        self.block.append(batch_index, replay_followers)

    def begin_parallel_forward_wave(
        self,
        batch_indices: Sequence[int],
        canonical_device: torch.device,
    ) -> None:
        self.block.begin(batch_indices, canonical_device)

    def seal_parallel_forward_batch(self, batch_index: int, device: torch.device) -> None:
        self.block.seal_batch(batch_index, device)

    def flush_parallel_forward_wave(self) -> None:
        self.block.flush()

    def abort_parallel_forward_wave(self) -> None:
        self.block.abort()
