---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-pytorch-gil0
description: Free-threaded (PYTHON_GIL=0) PyTorch safety. Use before writing, reviewing, or debugging any code that calls PyTorch from more than one thread, runs on a 3.13t/3.14t build, or touches CUDA/HIP peer-to-peer copies, streams, or device selection. Requires reading pytorch_gil_0_compat.md first.
---

# PyTorch on a free-threaded interpreter

`PYTHON_GIL=0` removes the global interpreter lock. It does **not** make a C
extension thread-safe. PyTorch wheels built for `3.14t` still contain entry
points that take a global lock, that are unsafe to call concurrently, or that
silently corrupt data. We have measured this: `torch.cuda.comm.broadcast_coalesced`
both hangs and returns torn copies when called from several threads on a
GIL-disabled build, while the same call is fine on a GIL build.

## Required reading

Before writing or reviewing any PyTorch code that can run on more than one
thread, read `pytorch_gil_0_compat.md` at the repository root. It is the
authoritative ledger of every PyTorch API we use, its measured GIL=0 result,
and the workaround for each unsafe entry.

If the API you need is not in the ledger, treat it as UNVERIFIED:

1. Add a probe to `scripts/gil0_api_probe.py`.
2. Run it with `PYTHON_GIL=0 python scripts/gil0_api_probe.py --only <name>`
   on a free-threaded build with at least two GPUs.
3. Record the result in `pytorch_gil_0_compat.md` (table row plus, for an
   unsafe API, a "Manual findings" entry).

Do not merge a multi-threaded code path that calls an API the ledger does not
cover, and never upgrade a row to SAFE based on a single-threaded run.

## Rules

- Never call `torch.cuda.comm.*` from worker threads. Use
  `gptqmodel.utils.p2p.peer_copy_batch`, which drives
  `cudaMemcpyPeerAsync`/`hipMemcpyPeerAsync` through `ctypes` (the GIL is
  released for the call).
- `torch.nn.parallel.replicate` internally calls `broadcast_coalesced`. It is
  serialized by `_THREAD_SAFE_PARALLEL` in `looper_helpers.py`, but that only
  prevents concurrent entry; it does not make the entry point safe.
  `clone_module_for_devices` skips it when the GIL is disabled and uses the
  deepcopy + `move_module_to_device` path. `use_torch_replicate` is off by
  default; enable it only after qualifying replicate for the target build.
- `torch.cuda.set_device` is thread-local. Set it at the top of every worker;
  never inherit the device of the thread that spawned it.
- Join work with per-thread streams and events. Avoid
  `torch.cuda.synchronize()`, which waits for the whole device and serializes
  unrelated workers.
- After a failed runtime call, drain the sticky error with
  `cudaGetLastError`/`hipGetLastError`
  (`gptqmodel.utils.p2p._clear_last_error_for_backend`). HIP leaves error 704
  (`peer access is already enabled`) behind otherwise, and the *next*
  unrelated PyTorch call raises it.
- Prefer copies that are already device-to-device. Host staging through
  `cpu()` is correct but defeats P2P and doubles traffic.
- Guard every fast path with a fallback that is correct without the fast path,
  and test the fallback.

## Verifying

A concurrency claim needs evidence from a free-threaded interpreter, not from
a GIL build:

```bash
PYTHON_GIL=0 python -c "import sys; assert sys._is_gil_enabled() is False"
PYTHON_GIL=0 python -m pytest tests/test_p2p_copy_utils.py -q
PYTHON_GIL=0 python scripts/gil0_api_probe.py --all --json /tmp/gil0.json
```

Hangs do not raise, so run concurrency tests under a timeout and keep the
probe-per-subprocess pattern. A test that passes once proves nothing; run it
several times before calling an API safe.
