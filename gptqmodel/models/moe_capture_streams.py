# SPDX-License-Identifier: Apache-2.0
"""Bounded CUDA-stream waves for independent routed-expert input capture."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import TypeVar

import torch


T = TypeVar("T")
R = TypeVar("R")


class RoutedMoECaptureStreamAttachment:
    """Launch independent expert captures on a stable, bounded stream set.

    ThreadX still owns the only host thread for each CUDA device. This attachment
    only overlaps independent expert kernels submitted by that owner thread; it
    never creates another host worker or queues work from a second device thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[tuple[int, int], tuple[torch.cuda.Stream, ...]] = {}

    def _get_streams(
        self, device: torch.device, count: int
    ) -> tuple[torch.cuda.Stream, ...]:
        device = torch.device(device)
        key = (
            device.index if device.index is not None else torch.cuda.current_device(),
            count,
        )
        with self._lock:
            streams = self._streams.get(key)
            if streams is None:
                streams = tuple(torch.cuda.Stream(device=device) for _ in range(count))
                self._streams[key] = streams
            return streams

    def run(
        self,
        *,
        device: torch.device,
        entries: Sequence[T],
        stream_count: int,
        launch: Callable[[T], R],
    ) -> list[R]:
        """Submit entries in order and return their immediate host-side results."""

        device = torch.device(device)
        count = min(max(1, int(stream_count)), len(entries))
        if device.type != "cuda" or count == 1:
            return [launch(entry) for entry in entries]

        streams = self._get_streams(device, count)
        owner_stream = torch.cuda.current_stream(device)
        for stream in streams:
            stream.wait_stream(owner_stream)

        results = []
        for index, entry in enumerate(entries):
            with torch.cuda.stream(streams[index % count]):
                results.append(launch(entry))

        # Restore ThreadX's owner-stream dependency without synchronizing here.
        # The lifecycle's existing device-group boundary performs the one required
        # device synchronization after all Hessian updates have been submitted.
        for stream in streams:
            owner_stream.wait_stream(stream)
        return results
