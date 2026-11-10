<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PyTorch GIL=0 (free-threaded) compatibility ledger

GPT-QModel runs multi-GPU quantization and inference with `PYTHON_GIL=0` on
free-threaded CPython (3.13t/3.14t) so worker threads really run in parallel.
A `3.14t` wheel only proves the extension *can* run without the GIL. It does
not prove that an entry point is safe to call from several threads. This file
records every PyTorch API we have encountered, what we measured, and how to
work around the ones that are unsafe.

Treat anything not listed here as **UNVERIFIED**. Add a probe before relying on
it from a thread.

## Regenerating the tables

```bash
PYTHON_GIL=0 python scripts/gil0_api_probe.py --all --json /tmp/gil0.json
python scripts/gil0_api_probe.py --only torch.cuda.mem_get_info   # one API
```

Each probe runs 8 threads x 4 iterations in its own subprocess with a 90 s
timeout, so a hang is recorded instead of wedging the run. Do not hand-edit the
result column: add a probe to `scripts/gil0_api_probe.py`, then update the row.
Manual findings that a probe cannot express go in
[Manual findings](#manual-findings).

## Rules for GPT-QModel code

1. **Never call `torch.cuda.comm.*` from worker threads.** `broadcast_coalesced`
   has hung and returned torn copies; `reduce_add` has hung inside NCCL and
   segfaulted. Use
   `gptqmodel.utils.p2p.peer_copy_batch`, which calls
   `cudaMemcpyPeerAsync`/`hipMemcpyPeerAsync` through `ctypes` and releases the
   GIL for the enqueue.
2. **`torch.cuda.set_device` is thread-local.** Set it at the top of every
   worker. Never inherit the device of the thread that spawned it.
3. **Join with events and per-thread streams.** `torch.cuda.synchronize()`
   waits for the whole device and serializes unrelated workers; wait on the
   destination stream instead.
4. **Drain sticky runtime errors.** After a failed runtime call, call
   `cudaGetLastError`/`hipGetLastError`
   (`gptqmodel.utils.p2p._clear_last_error_for_backend`). HIP otherwise leaves
   error 704 behind and the *next* unrelated PyTorch call raises it.
5. **Every fast path needs a correct fallback.** Test the fallback, not just
   the fast path.
6. **A single passing run proves nothing.** Hangs and torn copies are
   intermittent. Repeat a concurrency test several times, under a timeout.

## Environment the table was produced on

| Item | Value |
|---|---|
| Python | 3.14.10 free-threading build, `PYTHON_GIL=0`, `sys._is_gil_enabled() is False` |
| Torch | 2.16.0.dev20261025+rocm10.1 (ROCm 7.16) for the ROCm rows; 2.16.0.dev+cu130 for the NVIDIA GIL=0 broadcast check |
| GPUs | 4x AMD Radeon AI PRO R9700, 12/12 ordered pairs P2P enabled; 2x RTX 4090 |
| Kernel | Linux 7.2.13, glibc 2.43 |
| Probed on | 2026-11-02 |

The CUDA host used for the rest of the P2P work (2x RTX 4090, torch
2.15.0+cu132) runs a GIL build, so its probes cannot detect GIL=0 races. The
`broadcast_coalesced` corruption above was reproduced separately on the
free-threaded NVIDIA wheel (torch `2.16.0.dev+cu130`, Python 3.14.10t). Re-run
`scripts/gil0_api_probe.py` on any new free-threaded wheel before trusting the
results below for that build.

## Status legend

| Result | Meaning |
|---|---|
| SAFE (concurrent) | 8 threads x 4 iterations completed with correct results |
| SAFE (single-thread only) | correct as a one-off / main-thread call; do not parallelize |
| UNSAFE (hangs) | a concurrent call never returned and had to be killed |
| UNSAFE (crash) | the process died (SIGSEGV) |
| UNSAFE (torn data) | a concurrent call returned corrupted data |
| CAUTION | passed once, but shares a known-unsafe path or was not repeated |
| UNVERIFIED | not probed on a free-threaded build |

## Device, stream, and memory runtime

| API | Result | Where we use it | Notes / workaround |
|---|---|---|---|
| `torch.cuda.is_available` | SAFE (concurrent) | everywhere | Cheap; safe to call from workers. |
| `torch.cuda.device_count` | SAFE (concurrent) | device pool setup | Cache it; the value cannot change at runtime. |
| `torch.cuda.get_device_name` | SAFE (concurrent) | logs, telemetry | Pure metadata query. |
| `torch.cuda.get_device_capability` | SAFE (concurrent) | kernel dispatch | Pure metadata query. |
| `torch.cuda.get_device_properties` | SAFE (concurrent) | VRAM planning | Pure metadata query. |
| `torch.cuda.current_device` | SAFE (concurrent) | device bookkeeping | Thread-local, no lock contention. |
| `torch.cuda.set_device` | SAFE (concurrent) | worker entry points | Thread-local. Set it explicitly in every worker; the probe asserts isolation. |
| `torch.cuda.current_stream` | SAFE (concurrent), thread-local | copies, events | Returns the calling thread's stream. Never cache another thread's stream. |
| `torch.cuda.Stream` | SAFE (concurrent), per-thread | forward workers | Create one stream per worker; do not share a stream across threads. |
| `torch.cuda.stream` (context manager) | SAFE (concurrent), per-thread | scoped work | Only affects the calling thread. |
| `torch.cuda.device` (context manager) | SAFE (concurrent), thread-local | P2P peer copies | Sets the calling thread's current device and restores it on exit. **Required** around a HIP peer copy: see manual finding 5. |
| `torch.cuda.Event` (record / wait / synchronize) | SAFE (concurrent), per-thread | P2P copy ordering, Hessian readiness | Record on the producing stream, `wait_event` on the consuming one. Do not synchronize an event from a thread that did not record it. |
| `torch.cuda.synchronize` | SAFE (concurrent) | tests, diagnostics | Safe but waits for the whole device. Use a stream sync in hot paths. |
| `torch.cuda.empty_cache` | SAFE (concurrent) | VRAM pressure handling | Takes the allocator lock; safe, but it serializes workers. Call it from one place, not per worker. |
| `torch.cuda.mem_get_info` | SAFE (concurrent) | Hessian staging budget, VRAM checks | Was suspected in an early hang; the traceback proved the hang was in `broadcast_coalesced`. Keep it out of tight loops. |
| `torch.cuda.memory_allocated` | SAFE (concurrent) | telemetry | Cheap allocator query. |
| `torch.cuda.max_memory_allocated` | SAFE (concurrent) | telemetry | Cheap allocator query. |
| `torch.cuda.reset_peak_memory_stats` | SAFE (concurrent) | per-layer telemetry | Resets process-wide peaks; do not use it to measure a single worker. |
| `torch.cuda.memory_stats` | SAFE (concurrent) | telemetry | Builds a dict; slower than the scalar queries. |
| `torch.cuda.set_per_process_memory_fraction` | SAFE (concurrent), process-wide | VRAM budgeting | Global setting; call once during setup, never from workers. |
| `torch.cuda.get_rng_state_all` | SAFE (concurrent), process-wide | determinism / tests | Serializes RNG state; slow and global. |
| `torch.cuda.set_rng_state_all` | SAFE (concurrent), process-wide | determinism / tests | Restores global RNG; do not use to give workers independent streams. Use `torch.Generator` per worker instead. |
| `torch.cuda.can_device_access_peer` | SAFE (concurrent), needs 2 GPUs | `gptqmodel.utils.p2p` | Only reports capability; it does not enable access. |
| `torch.cuda.is_initialized` | SAFE (concurrent) | startup checks | Avoid calling before you want CUDA initialized. |
| `torch.cuda.is_current_stream_capturing` | SAFE (concurrent) | graph guards | Thread-local. |
| `torch.cuda.get_arch_list` | SAFE (concurrent) | diagnostics | Pure metadata query. |
| `torch.cuda.CUDAGraph` + `torch.cuda.graph` | SAFE (single-thread only) | not used yet | Graph capture is process-global. Capture on the main thread, then replay from one stream per worker. |

## Cross-device collective copies

| API | Result | Where we use it | Notes / workaround |
|---|---|---|---|
| `torch.cuda.comm.broadcast_coalesced` | UNSAFE (hangs / torn data) on **both** ROCm and CUDA; payload-dependent | was `gptqmodel/utils/p2p.py`; removed | Confirmed broken on both backends under `PYTHON_GIL=0`. ROCm 4x R9700 (torch `2.16.0.dev20261025+rocm10.1`, Python 3.14.10t): 6/6 realistic trials failed, 4 hangs inside `torch._C._broadcast_coalesced` and 2 torn copies; a 4-trial rerun gave 2 torn + 1 SIGSEGV. NVIDIA 2x RTX 4090 (torch `2.16.0.dev+cu130`, Python 3.14.10t): 6/6 trials torn, and a 10-trial rerun gave 8 torn + 2 hangs. Nuance: small payloads (two 8192-element tensors) passed 8/8; 1024x1024 (4 MiB) payloads tear ~80% of the time. Treat any size as unsafe. Replaced by `gptqmodel.utils.p2p.peer_copy_batch`. |
| `torch.cuda.comm.broadcast` | CAUTION | not used | Same `torch._C._broadcast` family. One probe run passed; do not call it from workers. |
| `torch.cuda.comm.reduce_add` | UNSAFE (hang / crash) | not used | Segfaults (SIGSEGV, exit -11) under concurrent use, and on NVIDIA hung 2/3 trials inside `torch/nn/parallel/comm.py:111` -> `torch/cuda/nccl.py:121` (`nccl.reduce`); a 2-thread x 2-task control hung the same way. Reduce Hessian partials with `add_` on explicit streams instead. |
| `torch.cuda.comm.gather` | CAUTION | not used | One probe run passed; not repeated. Prefer explicit `peer_copy_batch` into preallocated buffers. |
| `torch.cuda.comm.scatter` | CAUTION | not used | One probe run passed; not repeated. Same caution as `gather`. |
| `torch.nn.parallel.replicate` | CAUTION (serialized), skipped when GIL=0 | `gptqmodel/utils/looper_helpers.py` replica fast path | Internally calls `broadcast_coalesced`, so it inherits the failure above. It is called through `_THREAD_SAFE_PARALLEL` (a module-wide lock), and `clone_module_for_devices` now skips it entirely when `has_gil_disabled()`; the deepcopy path uses `move_module_to_device`/`peer_copy_batch`. `use_torch_replicate` is off by default; enable it only after qualifying replicate for the target build. |
| `gptqmodel.utils.p2p.peer_copy_batch` (ours) | SAFE (concurrent) | module moves, calibration input staging, Hessian copies | `ctypes` call into `cudaMemcpyPeerAsync`/`hipMemcpyPeerAsync`; the GIL is released for the call. The enqueue runs with the **destination device current**, which HIP requires (manual finding 5). Returns `False` when the group is not a simple two-device batch so callers can fall back to `Tensor.to`. |

## Vendored NF12 operators

| API | Result | Where we use it | Evidence and limit |
|---|---|---|---|
| `gptqmodel_nf12` encode/decode/linear/linear_fp32/linear_bf16_fp32/embedding | SAFE (concurrent), NVIDIA CUDA | online BF16 weight preparation and inference | `scripts/gil0_api_probe.py --only gptqmodel_nf12.ops` passed four separate processes, each 8 threads × 4 iterations, on 2026-11-10 with Python 3.14.7 GIL disabled, PyTorch 2.15.0+cu132 and two NVIDIA PG506-230 SM80 GPUs. The probe prewarms JIT compilation, assigns worker devices explicitly, checks exact decoded/embedded values, BF16/FP32 linear results, and FP32 Tensor Core scores, then synchronizes each worker stream. ROCm is explicitly unsupported by this CUDA package; GPT-QModel's separate HIP runtime remains its path. |

## Tensor creation and elementwise ops

| API | Result | Where we use it | Notes / workaround |
|---|---|---|---|
| `torch.empty` | SAFE (concurrent) | staging buffers | Allocator is internally locked; contention only, no correctness risk. |
| `torch.zeros` | SAFE (concurrent) | buffers, masks | Fills on the current stream. |
| `torch.ones` | SAFE (concurrent) | tests, probes | - |
| `torch.full` | SAFE (concurrent) | constant buffers | - |
| `torch.randn` | SAFE (concurrent) | calibration data, tests | Uses the per-device RNG; pass a `torch.Generator` for reproducibility per worker. |
| `torch.rand` | SAFE (concurrent) | tests | Same RNG caveat as `randn`. |
| `torch.arange` | SAFE (concurrent) | position ids, index buffers | - |
| `torch.eye` | SAFE (concurrent) | tests | - |
| `torch.tensor` | SAFE (concurrent) | constants | Copies host data; prefer `as_tensor` when the input is already a tensor. |
| `torch.as_tensor` | SAFE (concurrent) | constants | No copy when dtype/device already match. |
| `torch.from_numpy` | SAFE (concurrent) | host-side data | CPU only; shares memory with the array. Do not resize the source. |
| `torch.zeros_like` | SAFE (concurrent) | accumulators | - |
| `torch.ones_like` | SAFE (concurrent) | masks | - |
| `torch.empty_like` | SAFE (concurrent) | Hessian staging | Used for the single P2P staging buffer in `quantization/gptq.py`. |
| `torch.full_like` | SAFE (concurrent) | tests | - |
| `torch.cat` | SAFE (concurrent) | batching, packing | - |
| `torch.stack` | SAFE (concurrent) | batching | - |
| `torch.split` | SAFE (concurrent) | sharding | - |
| `torch.unsqueeze` | SAFE (concurrent) | shape plumbing | - |
| `torch.diagonal` | SAFE (concurrent) | math kernels | - |
| `torch.matmul` | SAFE (concurrent) | calibration, Hessian | cuBLAS/hipBLAS handle per stream; safe. |
| `torch.einsum` | SAFE (concurrent) | math kernels | - |
| `torch.where` | SAFE (concurrent) | masking | - |
| `torch.maximum` | SAFE (concurrent) | clamping | - |
| `torch.clamp` | SAFE (concurrent) | quantization bounds | - |
| `torch.round` | SAFE (concurrent) | rounding | - |
| `torch.rsqrt` | SAFE (concurrent) | normalization | - |
| `torch.topk` | SAFE (concurrent) | act-order selection | - |
| `torch.argsort` | SAFE (concurrent) | act-order selection | - |
| `torch.sum` | SAFE (concurrent) | reductions | Reduction kernels use the current stream. |
| `torch.mean` | SAFE (concurrent) | reductions | - |
| `torch.all` | SAFE (concurrent) | validation | - |
| `torch.equal` | SAFE (concurrent) | tests | - |
| `torch.isfinite` | SAFE (concurrent) | NaN guards | - |
| `torch.isnan` | SAFE (concurrent) | NaN guards | - |
| `torch.bitwise_and` / `_or` / `_xor` | SAFE (concurrent) | quantization packing | - |
| `torch.floor_divide` | SAFE (concurrent) | packing | - |
| `torch.abs` / `exp` / `log` / `pow` | SAFE (concurrent) | math kernels | - |
| `torch.save` + `torch.load` | SAFE (concurrent), process-wide | checkpoints | File I/O; safe but serializes on the filesystem. Write distinct paths per worker. |
| `torch.Generator(device="cuda")` | SAFE (concurrent), per-thread | deterministic sampling | One generator per worker. Never share a generator across threads. |
| `Tensor.to` (cuda:0 -> cuda:1) | SAFE (concurrent) (needs 2 GPUs) | fallback for every P2P fast path | Goes through the PyTorch copy machinery. Correct, but per-tensor and slower than `peer_copy_batch`; use it only as the fallback. |
| `Tensor.copy_` (non_blocking, cross-device) | SAFE (concurrent) (needs 2 GPUs) | P2P verify, Hessian adopt | Must synchronize the destination stream before reading the result. |
| `torch.inference_mode` | SAFE (concurrent), thread-local | inference paths | Thread-local grad mode; no shared state. |
| `torch.no_grad` | SAFE (concurrent), thread-local | calibration | Same as above. |
| `torch.is_autocast_enabled("cpu")` + `torch.autocast("cpu", enabled=False)` | SAFE (concurrent CPU) | GGUF Torch CPU FP32 reduction | On Python 3.14.10t / Torch 2.16.0.dev20261102+cu130 with `PYTHON_GIL=0`, the probe ran 8 threads x 4 iterations in three subprocesses. The active flag remained thread-local; the nested disabled context returned the exact FP32 dot `100000`, while surrounding BF16 dots stayed `99840`. |
| `torch.is_autocast_enabled("cuda")` + `torch.autocast("cuda", enabled=False)` | SAFE (concurrent CUDA) | NF12-AD wide fallback GEMM, Torch FP8 wide fallback | Probed 2026-11-10 on Python 3.14.7t / Torch 2.15.0+cu132 with `PYTHON_GIL=0`, 2x RTX 4090: 8 threads x 4 iterations, 3 subprocesses, all PASS. The CUDA autocast flag stayed thread-local and the nested disabled context produced the FP32 dot `100000` while the surrounding BF16 dots stayed `99840`. An 8-thread x 3-process stress of `NF12ADLinear`'s FP16 wide branch returned no hangs and no torn data; the bitwise differences observed were FP32-vs-FP64 rounding ties that also appear single-threaded. |
| `torch.get_autocast_dtype("cuda")` | SAFE (concurrent CUDA) | NF12-AD fallback predicate choosing FP16 vs wide | Probed 2026-11-10 on Python 3.14.7t / Torch 2.15.0+cu132 with `PYTHON_GIL=0`, 2x RTX 4090: 8 threads x 4 iterations in 3 subprocesses, three repeated runs, all PASS (probe `torch.get_autocast_dtype + torch.autocast (CUDA)`). The value stayed thread-local: `float16` outside a context, `bfloat16` inside one, unchanged across a nested `enabled=False` context. |

## Manual findings

### 1. `torch._C._broadcast_coalesced` hangs and tears on BOTH backends (2026-11-02)

Confirmed broken on both vendors under `PYTHON_GIL=0`; do not treat this as a
ROCm-only bug.

**ROCm: 4x Radeon AI PRO R9700**, torch `2.16.0.dev20261025+rocm10.1`, Python
3.14.10t.

- Worker threads blocked inside `torch/nn/parallel/comm.py:67`
  (`broadcast_coalesced`) and never returned. The process spun at roughly 200%
  CPU and had to be killed after 224 s; `faulthandler` captured the stack in the
  C call.
- A separate run returned a replica with 1048567 of 1048576 elements wrong, so
  the failure is silent corruption, not only a hang.
- Frequency: an isolated `broadcast_coalesced` probe passed 8/8, but the
  realistic pattern (module leaves plus two 1024x1024 inputs plus
  `mem_get_info` per thread) failed in **6 of 6 trials**: 4 hangs (killed after
  60 s) and 2 torn copies. A later 4-trial rerun gave 1 pass, 2 torn copies and
  1 SIGSEGV, so the failure mode is broader than hang/tear.

**NVIDIA: 2x RTX 4090**, torch `2.16.0.dev+cu130`, Python 3.14.10t.

- The same realistic pattern returned **torn copies in 6 of 6 trials**; a
  10-trial rerun gave **8 torn copies + 2 hangs**. No clean run in either set.
- Payload size matters: a probe with two 8192-element tensors passed 8/8, while
  a 1024x1024 (4 MiB) variant tore 7/8 and hung once. Small inputs can pass and
  still be unsafe at production sizes.

Fix: `gptqmodel.utils.p2p.peer_copy_batch` issues
`cudaMemcpyPeerAsync`/`hipMemcpyPeerAsync` through `ctypes` (GIL released) on
the destination stream, after joining the source stream with an event. It is
covered by `tests/test_p2p_copy_utils.py` (8 concurrent threads, on 2x RTX 4090
and on 4x R9700 with the GIL disabled) and by an end-to-end two-GPU quantize ->
save -> reload -> generate smoke run on both hosts.

### 2. `torch.cuda.comm.reduce_add` hangs and segfaults

The probe died with SIGSEGV (`exit -11`) under 8 concurrent threads on ROCm. On
NVIDIA it hung in 2 of 3 trials, with the stack inside
`torch/nn/parallel/comm.py:111` -> `torch/cuda/nccl.py:121` (`nccl.reduce`); a
2-thread x 2-task control hung the same way, so this is the collective path, not
the payload shape. Do not reduce cross-device tensors with `comm.reduce_add`;
add partials explicitly into a destination buffer on the destination stream.

### 3. HIP sticky error 704 after enabling peer access

`hipDeviceEnablePeerAccess` returns 704 (`peer access is already enabled`) when
a pair was enabled earlier, and leaves that code as the calling thread's last
error. The next unrelated PyTorch call then raises
`RuntimeError: peer access is already enabled`. Always drain the code with
`hipGetLastError`/`cudaGetLastError` after a probe
(`gptqmodel.utils.p2p._clear_last_error_for_backend`), including on the
failure path of the verification copy.

### 4. Triton `nanobind: leaked N instances` at interpreter exit

Printed during shutdown on GIL=0 builds. It comes from Triton's bindings, not
from GPT-QModel, and does not affect results. Do not treat it as a test
failure, and do not suppress it globally; it is a useful signal if it starts
appearing mid-run.

### 5. HIP peer copies bind to the *current* device's context

`hipMemcpyPeerAsync` associates the stream argument with the calling thread's
current device context, not with the device that owns the stream. Issuing the
copy while the source device was current put the work on the source device's
queue, so the destination stream -- and `Tensor.cpu()` -- never waited for it.

Measured on 4x Radeon AI PRO R9700, torch `2.16.0.dev20261025+rocm10.1`,
`PYTHON_GIL=0`: a `p2p_move_nested` of
`{ids: 32 KiB int64, hidden: 4 x 256 KiB f32}` read back an all-zero `ids`
**24 of 24 single-threaded trials**, while the four `hidden` tensors were
correct. Only `torch.cuda.synchronize(source)` or a full `synchronize()` made
the values appear; `synchronize(destination)` did not, which pinned the copy to
the source queue. The same shape returned torn `ids` in a concurrent
8-thread x 24-task run, failing 3 of 3 processes.

Fix: `peer_copy_batch` wraps the stream lookup, the event join and the copy
loop in `with torch.cuda.device(target_device)`. Re-measured after the fix:
24/24 concurrent tasks byte-exact in 3 of 3 processes, and no synchronize is
needed before a read on the destination device.

CUDA (2x RTX 4090, torch `2.16.0.dev+cu130`, `PYTHON_GIL=0`) ordered the copy
correctly even without the device scope, so the defect is HIP-specific. Keep
the scope on both backends: it is what makes the ordering guarantee explicit
rather than incidental.

Regression test:
`tests/test_p2p_copy_utils.py::test_p2p_move_nested_is_ordered_on_destination_stream`
(fails on the pre-fix code, passes after).

## Adding a probe

1. Add the callable to `build_probes()` in `scripts/gil0_api_probe.py`.
2. Run it on a free-threaded build with two or more GPUs:
   `PYTHON_GIL=0 python scripts/gil0_api_probe.py --only <name>`.
3. Repeat it several times. A single PASS is not evidence.
4. Add the row above, with the workaround if the result is not SAFE.

## Where P2P is used

`gptqmodel.utils.p2p.enable_peer_access` / `enable_for_devices` probe and
enable every ordered pair of the GPUs a run actually uses, cache the result,
and log one actionable warning when a pair cannot be enabled (with the
motherboard/BIOS/hardware compatibility wording). `peer_copy_batch` then backs
module replication and per-layer moves, calibration input staging, Hessian
partial reduction, and shared-Hessian adoption. Control it with
`BaseQuantizeConfig.p2p` (or `GPTQModel.load(..., p2p=...)` for inference):
`P2PConfig(mode="off")` skips probing, and `P2PConfig(probe=True)` additionally
measures each directed link. `P2PConfig` is a no-op for CPU-only runs and for
CPU<->CPU transfers: CPU devices are skipped, no warning is emitted, and the
move helpers keep their regular copy path.
