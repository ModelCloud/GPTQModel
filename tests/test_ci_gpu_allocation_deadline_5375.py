# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""#5375: the GPU allocation deadline applies to every response shape.

``allocate_gpu`` consulted ``--timeout-sec`` only in the valid "no GPU
available" branch. An allocator that answered with an invalid payload (``{}``,
a missing ``gpuIds``, a transport failure) restarted the loop without the
deadline check and polled forever.

The wait now has one monotonic deadline: the loop checks it before every
request, the request timeout and retry delays are capped to the remaining
budget, and both retry sleeps are capped the same way.
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github/scripts/ci_gpu.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ci_gpu_deadline_5375", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "base_url": "http://allocator.invalid",
        "sm": "0",
        "run_id": "1",
        "test": "deadline",
        "runner": "runner",
        "count": "1",
        "request_timeout": 0.05,
        "retries": 0,
        "retry_delay": 0.0,
        "sleep_sec": 0.01,
        "timeout_sec": 0.3,
        "require_single": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# A broken loop polls forever; cap it so the regression test fails instead of
# hanging. The cap answers with a valid id, which also exercises the success
# path instead of waiting for the deadline.
MAX_REQUESTS = 200


def _stub_allocator(monkeypatch, module, response):
    calls = []

    def fake_request(url, *, method="GET", body=None, timeout):
        calls.append(timeout)
        if len(calls) > MAX_REQUESTS:
            return {"gpuIds": "0"}
        return response

    monkeypatch.setattr(module, "request_json", fake_request)
    monkeypatch.setattr(module, "print_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.subprocess, "getoutput", lambda command: "")
    # The request payload reads the host inventory, which needs `device_smi`
    # and a parseable `nvidia-smi`; the deadline does not.
    monkeypatch.setattr(
        module,
        "build_server_info",
        lambda: {"platform": "test", "arch": "x86_64", "system": "Linux"},
    )
    monkeypatch.setattr(module, "query_gpu_inventory", lambda: [])
    return calls


def test_an_invalid_response_honours_the_allocation_deadline(monkeypatch):
    """``{}`` used to poll forever; it must now time out like ``-1`` (#5375)."""

    module = _load_module()
    calls = _stub_allocator(monkeypatch, module, {})

    started = time.monotonic()
    result = module.allocate_gpu(_args())
    elapsed = time.monotonic() - started

    assert result == 1
    assert calls, "the allocator was never asked"
    assert elapsed < 5.0, f"the invalid-response loop overran the deadline: {elapsed:.1f}s"


def test_a_missing_gpu_id_honours_the_allocation_deadline(monkeypatch):
    """A dict without ``gpuIds`` is the other invalid shape from the report."""

    module = _load_module()
    _stub_allocator(monkeypatch, module, {"message": "try later"})

    assert module.allocate_gpu(_args()) == 1


def test_the_no_gpu_response_still_times_out(monkeypatch):
    """The original branch keeps its deadline behavior."""

    module = _load_module()
    _stub_allocator(monkeypatch, module, {"gpuIds": "-1"})

    assert module.allocate_gpu(_args()) == 1


def test_an_expired_deadline_skips_the_request(monkeypatch):
    """A deadline already in the past must not start another attempt."""

    module = _load_module()
    calls = _stub_allocator(monkeypatch, module, {})

    result = module.request_json_with_retry(
        "http://allocator.invalid/get",
        method="POST",
        body={},
        timeout=1.0,
        retries=3,
        retry_delay=1.0,
        deadline=time.monotonic() - 1.0,
    )

    assert result is None
    assert calls == []


def test_a_transport_failure_cannot_restart_the_clock(monkeypatch):
    """Retries and their delays are capped to the remaining budget (#5375)."""

    module = _load_module()
    calls = []

    def failing_request(url, *, method="GET", body=None, timeout):
        calls.append(timeout)
        raise OSError("allocator unreachable")

    monkeypatch.setattr(module, "request_json", failing_request)

    started = time.monotonic()
    result = module.request_json_with_retry(
        "http://allocator.invalid/get",
        method="POST",
        body={},
        timeout=1.0,
        retries=50,
        retry_delay=0.05,
        deadline=time.monotonic() + 0.2,
    )
    elapsed = time.monotonic() - started

    assert result is None
    assert calls
    assert elapsed < 2.0, f"retries ignored the deadline: {elapsed:.1f}s"
    assert all(timeout <= 0.2 + 1e-6 for timeout in calls), calls
