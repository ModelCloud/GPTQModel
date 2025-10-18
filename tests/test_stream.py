# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest
import torch

from gptqmodel.utils.stream import stream_sync, stream_tensor_dict_to_cpu


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.01) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_stream_tensor_dict_to_cpu_cpu_path():
    src = torch.randn(4, 4, dtype=torch.float32)
    state: dict[str, object] = {}
    state_lock = threading.RLock()
    stored: dict[str, torch.Tensor] = {}

    result = stream_tensor_dict_to_cpu(
        {"payload": src},
        store_callback=lambda items: stored.update(items),
        state=state,
        state_lock=state_lock,
    )

    assert "payload" in result
    assert result["payload"].device.type == "cpu"
    torch.testing.assert_close(result["payload"], src)

    with state_lock:
        assert not state.get("streaming_events")
        assert not state.get("streaming_event_map")
    assert stored["payload"] is result["payload"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for async stream tests")
def test_stream_tensor_dict_to_cpu_cuda_background_release_preserves_events():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    payload = {"tensor": torch.randn(16, 16, device=device, dtype=torch.float16)}
    state: dict[str, object] = {}
    state_lock = threading.RLock()
    stored: dict[str, torch.Tensor] = {}

    result = stream_tensor_dict_to_cpu(
        payload,
        store_callback=lambda items: stored.update(items),
        state=state,
        state_lock=state_lock,
    )

    with state_lock:
        assert "tensor" in stored and stored["tensor"] is result["tensor"]
        assert result["tensor"].is_pinned()
        assert "streaming_event_map" in state
        assert "streaming_events" in state
        event_map = state["streaming_event_map"]
        tickets = list(state["streaming_events"])
        assert event_map["tensor"] is tickets[0].event
        ticket = tickets[0]
        assert ticket.sources, "sources should be retained until background wait completes"

    completed = _wait_until(lambda: ticket.background_done, timeout_s=10.0, interval_s=0.01)
    assert completed, "background worker did not complete within timeout"

    with state_lock:
        assert ticket.background_done is True
        assert ticket.sources is None
        assert "tensor" in state["streaming_event_map"], "event map entry must persist until explicit sync"
        assert state["streaming_event_map"]["tensor"] is ticket.event

    torch.testing.assert_close(result["tensor"].cpu(), payload["tensor"].cpu())

    stream_sync(state, state_lock)

    with state_lock:
        assert not state.get("streaming_events"), "stream_sync should clear pending tickets"
        assert "tensor" not in state.get("streaming_event_map", {}), "event map entry should be removed after sync"
