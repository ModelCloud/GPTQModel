<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Device thread pool shutdown

`DeviceThreadPool` has one shutdown contract: once `shutdown()` starts, every
task the pool already accepted is drained and resolved, and every later
`submit` / `submit_serial` is rejected. The contract is enforced by a single
lock, `_dispatch_lock`, that both sides take.

Before this change the two halves of the contract disagreed. A worker's run
loop exited as soon as its `_stop` flag was set, so a task that had been
accepted while the worker was busy stayed in the queue with a pending future
and a non-zero in-flight count. Submission had the mirror problem: nothing
checked the stop event, so a late caller could enqueue behind the worker's
sentinel (nothing would ever run it) or lazily start a fresh worker after the
pool had shut down.

## New flow

### Submission: accept or reject atomically

```mermaid
flowchart TD
    S["submit(device, fn) / submit_serial(device, fn)"] --> L["acquire _dispatch_lock (RLock)"]
    L --> Q{"_stop_event set?"}
    Q -- "Yes (shutdown started)" --> R["raise RuntimeError<br/>no new work accepted"]
    Q -- "No" --> P["_pick_worker_locked / _refresh_serial_worker_locked<br/>may lazily spawn a worker"]
    P --> M["_mark_scheduled(key)<br/>inflight += 1"]
    M --> E["worker.submit(...)<br/>enqueue task ahead of sentinels"]
    E --> U["release _dispatch_lock"]
    U --> D["worker _run loop: execute queued tasks in FIFO order"]
    D --> T{"sentinel reached?"}
    T -- "No" --> D
    T -- "Yes (queue drained)" --> X["break; _on_worker_exit removes worker"]
    R --> Z["caller sees RuntimeError"]
    style R fill:#fdd,stroke:#c00
    style Z fill:#fdd,stroke:#c00
    style X fill:#dfd,stroke:#060
```

### Shutdown: stop accepting, then drain

```mermaid
flowchart TD
    H["shutdown(wait=True)"] --> A["acquire _dispatch_lock"]
    A --> B["_stop_event.set()"]
    B --> C["snapshot _worker_groups"]
    C --> D["for each worker: w.stop()<br/>set _stop, enqueue sentinel"]
    D --> E["release _dispatch_lock"]
    E --> F["_gc_event.set(); join janitor"]
    F --> G{"wait?"}
    G -- "Yes" --> I["join every worker thread"]
    G -- "No" --> J["return"]
    I --> K["workers drained every accepted task<br/>then exited on the sentinel"]
    style B fill:#ffd,stroke:#a60
    style K fill:#dfd,stroke:#060
```

`stop()` queues its sentinel with `_q.put`, after every task the pool already
accepted, so reaching the sentinel proves the queue is drained. That is why the
run loop exits on the sentinel (`while True:` plus `break` on the non-task item)
instead of on `_stop.is_set()`.

## Before and after

Pointers are to `origin/main` at `d1d3da04`.

| Concern | Before | After |
| --- | --- | --- |
| Worker exit condition | `while not self._stop.is_set():` at `gptqmodel/utils/threadx.py:518`; exits before reading queued tasks once stop is set | `while True:` at the same loop; exits only on the sentinel, which `stop()` queues after accepted work |
| `submit` acceptance | worker picked and task enqueued with no shutdown check (`threadx.py:1146`) | `_require_accepting_locked()` runs under `_dispatch_lock` before selection and enqueue |
| `submit_serial` acceptance | enqueued after the `_dispatch_lock` block was released (`threadx.py:1175`) | check, selection and enqueue all inside the same lock block |
| Lazy worker spawn | `_pick_worker()` (`threadx.py:1465`) took the lock itself and could spawn a worker after shutdown | `_pick_worker_locked()` runs under the caller's lock, after the acceptance check |
| Shutdown ordering | `shutdown()` set `_stop_event` then snapshotted under the lock, but issued `w.stop()` outside it (`threadx.py:1233`, `:1257`) | sets `_stop_event` and issues every `w.stop()` inside `_dispatch_lock` |
| Lock type | `threading.Lock()` (`threadx.py:725`) | `threading.RLock()`: `_SyncWorker.stop()` re-enters `_on_worker_exit`, which takes the dispatch lock |

## Failure and ambiguity contract

```mermaid
stateDiagram-v2
    state "Accepting" as A
    state "Draining" as D
    state "Stopped" as S
    [*] --> A
    A --> A: submit accepted, task queued
    A --> D: shutdown() sets _stop_event
    A --> A: submit after shutdown observed -> RuntimeError
    D --> D: workers drain queued tasks
    D --> S: every worker joined
    S --> S: submit -> RuntimeError
    style S fill:#dfd,stroke:#060
```

| Situation | Before | After |
| --- | --- | --- |
| Task queued when `shutdown()` starts | abandoned; future pending forever, `_inflight` never drops, `wait()` can hang | drained in FIFO order, future resolved, `_inflight` reaches 0 |
| `submit` racing `shutdown()` | could enqueue behind the sentinel and never run | serialized: enqueued ahead of every sentinel (and drained) or rejected |
| `submit` after `shutdown()` | accepted; future never resolves | `RuntimeError("DeviceThreadPool is shutting down; ...")` |
| `submit_serial` after `shutdown()` | spawned a worker and enqueued | same `RuntimeError` |
| Empty worker group after shutdown | lazy selection repopulated it | rejected before selection; no thread is started |
| `_SyncWorker` stop during shutdown | stops were issued outside the lock, so no re-entry happened | `RLock` lets the shutdown thread re-enter `_on_worker_exit` without self-deadlocking |

There is no silent fallback: a rejected submission raises, and a task that was
accepted before shutdown is guaranteed to run. The only ambiguity is ordering
between a `submit` and a `shutdown` that race — the lock decides it, and either
outcome is well defined.

## Scope

The change adds no PyTorch API and no environment-variable switch; it only
reorders existing `threading` primitives, so `pytorch_gil_0_compat.md` is
unchanged. Regression coverage lives in `tests/test_threadx_shutdown.py`;
the existing threadx suites (`tests/test_threadx.py`,
`tests/test_threadx_cpu.py`, `tests/test_threadx_cpu_locks.py`,
`tests/test_threadx_warmup_accounting.py`) stay green.
