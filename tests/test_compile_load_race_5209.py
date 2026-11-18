# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A compiled model must keep serving while another model loads (#5209).

#5159 serialized only the *first* call of a compiled callable/module with model
construction. Every later call ran outside ``_MODEL_LOAD_LOCK``, and
``from_quantized`` called ``torch._dynamo.reset()`` on every load. A later call
re-enters dynamo tracing whenever a guard changes -- the load's bf16
``set_default_dtype`` window is exactly such a change -- so:

* the load's ``torch._dynamo.reset()`` wiped the other model's compiled frame
  while its forward thread was running it, which segfaults in dynamo's frame
  evaluation on a GIL build (the #5209 crash), and
* without the reset, the forward re-traced inside the bf16 window and dynamo's
  ``preserve_global_state`` restored bf16 after the load restored fp32, failing
  the load with ``model load left a corrupted state`` (#5127 again).

These tests pin the three contracts of the fix: a load does not reset dynamo,
a compiled forward waits for an in-flight load instead of tracing through its
dtype window, and repeated concurrent loads leave the default dtype and the
served outputs intact.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models import loader as loader_module
from gptqmodel.utils.torch import dynamo_traced_frames, torch_compile
from tests.test_model_load_lock_4204 import (
    _SignallingLock,
    _write_tiny_rtn_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_dynamo():
    return pytest.importorskip("torch._dynamo")


def test_a_load_does_not_reset_another_models_compiled_frames(tmp_path, monkeypatch):
    """Loading a model must not discard the frames other models already compiled.

    ``torch._dynamo.reset()`` is process-wide: it clears every compiled frame,
    including a function the caller compiled itself and a model that is already
    ``optimize()``d. Re-tracing is the single-thread cost, and running the reset
    concurrently with a compiled frame is the #5209 segfault.
    """

    dynamo = _require_dynamo()
    checkpoint = _write_tiny_rtn_checkpoint(tmp_path)

    compiled = torch_compile(lambda value: value * 2, backend="eager")
    value = torch.ones(2)
    assert compiled(value).tolist() == [2.0, 2.0], "the frame was not compiled"

    resets = []
    real_reset = dynamo.reset

    def reset_spy(*args, **kwargs):
        resets.append(1)
        return real_reset(*args, **kwargs)

    monkeypatch.setattr(dynamo, "reset", reset_spy)

    GPTQModel.load(
        str(checkpoint),
        device="cpu",
        dtype=torch.bfloat16,
        backend=BACKEND.GPTQ_TORCH,
    )

    assert resets == [], "a model load reset dynamo's process-wide cache (#5209)"
    after_load = dynamo_traced_frames()
    assert compiled(value).tolist() == [2.0, 2.0]
    assert dynamo_traced_frames() == after_load, (
        "the load discarded another caller's compiled frame; the next call re-traced"
    )


def _assert_compiled_call_waits_for_the_lock(lock, call, *args):
    """Run ``call`` while another thread holds the model-global lock.

    Pre-#5209 the proxy short-circuited on ``state["compiled"]`` and never took
    the lock, so the call finished while the holder still owned it. With the fix
    every call waits, which is what serializes a re-trace with a load.
    """

    holding = threading.Event()
    release = threading.Event()
    results = []
    errors = []

    def holder():
        with lock:
            holding.set()
            release.wait(timeout=60)

    def worker():
        try:
            results.append(call(*args))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    holder_thread = threading.Thread(target=holder, daemon=True)
    holder_thread.start()
    assert holding.wait(timeout=30), "the test holder never took the lock"

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    waited = lock.waited.wait(timeout=1.0)
    release.set()
    holder_thread.join(timeout=60)
    worker_thread.join(timeout=60)

    assert waited, "a compiled call ran while a load held the model-global lock"
    assert not worker_thread.is_alive(), "the compiled call never finished"
    assert not errors, f"the compiled call failed: {errors!r}"
    assert len(results) == 1


def test_every_compiled_call_takes_the_model_global_lock(monkeypatch):
    """The callable proxy must take the lock on later calls, not just the first."""

    _require_dynamo()
    lock = _SignallingLock()
    monkeypatch.setattr(loader_module, "_MODEL_LOAD_LOCK", lock)

    compiled = torch_compile(lambda value: value * 2, backend="eager")
    value = torch.ones(2)
    assert compiled(value).tolist() == [2.0, 2.0], "the frame was not compiled"

    _assert_compiled_call_waits_for_the_lock(lock, compiled, value)


def test_every_compiled_module_forward_takes_the_model_global_lock(monkeypatch):
    """The module proxy must take the lock on later forwards, not just the first."""

    _require_dynamo()
    lock = _SignallingLock()
    monkeypatch.setattr(loader_module, "_MODEL_LOAD_LOCK", lock)

    module = torch.nn.Linear(2, 2)
    compiled = torch_compile(module, backend="eager")
    assert isinstance(compiled, torch.nn.Module)
    value = torch.ones(1, 2)
    assert compiled(value).shape == (1, 2), "the module forward was not compiled"

    _assert_compiled_call_waits_for_the_lock(lock, compiled, value)


def test_optimized_forward_does_not_trace_through_a_loads_bf16_window(tmp_path, monkeypatch):
    """A warm compiled forward must not capture and restore a load's bf16 dtype.

    The load thread is held inside transformers' bf16 ``set_default_dtype``
    window while it owns ``_MODEL_LOAD_LOCK``. The forward thread calls the
    ``optimize()``d model: pre-fix the later call is lock-free, so its
    global-state guard fails on bf16 and dynamo re-traces with bf16 captured.
    The backend parks the trace until the load has restored fp32; when the trace
    finishes it restores bf16, so the process default dtype is left wrong.
    Post-fix the forward waits for the lock, traces (if at all) with fp32, and
    the load's window is never captured.
    """

    _require_dynamo()
    from torch._dynamo.backends.registry import lookup_backend

    eager_backend = lookup_backend("eager")
    checkpoint = _write_tiny_rtn_checkpoint(tmp_path)

    state = {"armed": False, "thread": None}
    trace_started = threading.Event()
    release_trace = threading.Event()

    def backend(graph_module, example_inputs):
        if state["armed"] and threading.get_ident() == state["thread"]:
            # The frame is traced with whatever default dtype is current; park
            # it until the load has restored fp32 so the restore ordering is
            # deterministic instead of a scheduling race.
            trace_started.set()
            release_trace.wait(timeout=60)
        return eager_backend(graph_module, example_inputs)

    model = GPTQModel.load(
        str(checkpoint),
        device="cpu",
        dtype=torch.float16,
        backend=BACKEND.GPTQ_TORCH,
    )
    model.optimize(backend=backend)
    assert model.compiled, "optimize() did not install the compiled forward"

    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        model.model(ids)  # first compiled call, warm

    original_dtype = torch.float32
    torch.set_default_dtype(original_dtype)
    state["armed"] = True

    lock = _SignallingLock()
    monkeypatch.setattr(loader_module, "_MODEL_LOAD_LOCK", lock)

    load_bf16 = threading.Event()
    release_load = threading.Event()
    load_ident = {}
    load_errors = []
    forward_errors = []

    real_set_default_dtype = torch.set_default_dtype

    def guarded_set_default_dtype(dtype):
        real_set_default_dtype(dtype)
        if (
            threading.get_ident() == load_ident.get("id")
            and dtype == torch.bfloat16
            and not load_bf16.is_set()
        ):
            # Hold the load inside its bf16 window while it owns the lock.
            load_bf16.set()
            release_load.wait(timeout=60)

    monkeypatch.setattr(torch, "set_default_dtype", guarded_set_default_dtype)

    def _load():
        load_ident["id"] = threading.get_ident()
        try:
            GPTQModel.load(
                str(checkpoint),
                device="cpu",
                dtype=torch.bfloat16,
                backend=BACKEND.GPTQ_TORCH,
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            load_errors.append(exc)

    def _forward():
        state["thread"] = threading.get_ident()
        try:
            with torch.inference_mode():
                model.model(ids)
        except BaseException as exc:  # noqa: BLE001 - reported below
            forward_errors.append(exc)

    load_thread = threading.Thread(target=_load, daemon=True)
    load_thread.start()
    assert load_bf16.wait(timeout=60), "the load never entered its bf16 window"

    forward_thread = threading.Thread(target=_forward, daemon=True)
    forward_thread.start()

    # Pre-fix the lock-free forward starts tracing through the bf16 window and
    # reaches the test backend; post-fix it waits for the load lock first.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if trace_started.wait(timeout=0.05):
            break
        if lock.waited.wait(timeout=0.05):
            break

    release_load.set()
    load_thread.join(timeout=60)
    assert not load_thread.is_alive(), "the load thread did not finish"
    release_trace.set()
    forward_thread.join(timeout=60)
    assert not forward_thread.is_alive(), "the forward thread did not finish"

    assert torch.get_default_dtype() == original_dtype, (
        "a concurrent compiled forward captured and restored the load's bf16 "
        f"default dtype: {torch.get_default_dtype()}"
    )
    assert not load_errors, f"concurrent load failed: {load_errors!r}"
    assert not forward_errors, f"concurrent compiled forward failed: {forward_errors!r}"


_CHILD = r"""
import sys
import threading
import time

import torch

from gptqmodel import BACKEND, GPTQModel

checkpoint, rounds, loads, forwards = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
)
print(f"gil_enabled={sys._is_gil_enabled()}", flush=True)

model = GPTQModel.load(
    checkpoint, backend=BACKEND.GPTQ_TORCH, dtype=torch.float16, device="cpu"
)
model.optimize(backend="eager")
ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
with torch.inference_mode():
    reference = model.model(ids).logits.clone()

errors = []
for rnd in range(rounds):
    round_errors = []

    def loader():
        try:
            for _ in range(loads):
                other = GPTQModel.load(
                    checkpoint,
                    backend=BACKEND.GPTQ_TORCH,
                    dtype=torch.bfloat16,
                    device="cpu",
                )
                del other
                # Let the serving thread take the lock between loads instead of
                # modelling an unbounded reload loop.
                time.sleep(0.02)
        except BaseException as exc:  # noqa: BLE001 - reported below
            round_errors.append(("load", repr(exc)[:300]))

    def forward():
        try:
            with torch.inference_mode():
                for _ in range(forwards):
                    logits = model.model(ids).logits
                    if not torch.allclose(logits, reference, atol=1e-3, rtol=1e-3):
                        round_errors.append(("output", "compiled logits changed"))
                        return
        except BaseException as exc:  # noqa: BLE001 - reported below
            round_errors.append(("forward", repr(exc)[:300]))

    load_thread = threading.Thread(target=loader)
    forward_thread = threading.Thread(target=forward)
    load_thread.start()
    forward_thread.start()
    load_thread.join(timeout=600)
    forward_thread.join(timeout=600)
    if load_thread.is_alive() or forward_thread.is_alive():
        round_errors.append(("timeout", "a thread did not finish"))

    default_dtype = torch.get_default_dtype()
    if default_dtype != torch.float32:
        round_errors.append(("dtype", f"default dtype is {default_dtype}"))
    torch.set_default_dtype(torch.float32)
    print(f"round={rnd} errors={round_errors}", flush=True)
    errors.extend(round_errors)

print(f"done errors={errors} default_dtype={torch.get_default_dtype()}", flush=True)
sys.exit(0 if not errors else 1)
"""


def test_optimized_forward_survives_repeated_concurrent_loads(tmp_path):
    """The issue's scenario: serve an ``optimize()``d model while another loads.

    Runs in a child process so a pre-fix segfault is reported as a failed exit
    status (SIGSEGV, returncode -11) instead of taking the test session down.
    Each round starts a load thread and a forward thread together, then asserts
    the load and the forward both succeeded, the process default dtype is still
    fp32, and every compiled forward produced the warm reference logits.
    """

    _require_dynamo()
    checkpoint = _write_tiny_rtn_checkpoint(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    # CPU-only: the race is dynamo's global state, not the accelerator.
    env.pop("CUDA_VISIBLE_DEVICES", None)

    rounds = 3
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(checkpoint),
            str(rounds),
            "2",
            "40",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"a compiled forward racing a model load failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-6000:]}\nstderr:\n{result.stderr[-4000:]}"
    )
    assert "done errors=[] default_dtype=torch.float32" in result.stdout
