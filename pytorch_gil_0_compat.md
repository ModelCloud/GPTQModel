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
   GIL for the enqueue when the allocator permits raw copies. Native
   allocators on CUDA and ROCm instead return `False` so movers use PyTorch's
   direct `Tensor.copy_`/`Tensor.to` path and its allocator-managed mappings.
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
| SAFE (serialized) | correct because concurrent callers wait on, or share, a process-wide lock or reference-counted guard |
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
| `Tensor.record_stream` | UNVERIFIED under GIL=0 | Torch streaming dequant `gptqmodel/nn_modules/qlinear/torch.py:706`; P2P raw copy `gptqmodel/utils/p2p.py:1841-1842`; activation staging `gptqmodel/utils/cuda_activation_buffer.py:119`; host streaming `gptqmodel/utils/stream.py:277`; marlin scratch `gptqmodel/utils/marlin_scratch.py:317`, `:443`, `:458`; GPTQ workspace `gptqmodel/quantization/gptq.py:274`, `:280` | Not probed on a free-threaded build, and the ledger has no evidence it is safe when two threads record the same tensor. It only extends allocator lifetime; it is not an ordering primitive. The Torch streaming path is opt-in (`GPTQ_TORCH_STREAMING`, `qlinear/torch.py:245`) and caches one dequant stream per device in a plain dict with an unlocked check-then-set (`qlinear/torch.py:856-862`), so concurrent first use of one module can race the cache (last stream wins). Use per-thread streams and do not share one tensor across concurrent streaming forwards. |
| `Stream.wait_stream` | UNVERIFIED under GIL=0 | Torch streaming dequant `gptqmodel/nn_modules/qlinear/torch.py:704`, `:728`; host streaming `gptqmodel/utils/stream.py:268`; activation staging `gptqmodel/utils/cuda_activation_buffer.py:113`; ParoQuant stage capture `gptqmodel/quantization/paroquant/optimization.py:1942`, `:1957`; EXL3 quantize `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py:521`, `:606`, `:643`, `:707`, `:1012`, `:1056`, `:1389`, `:1403` | Not probed on a free-threaded build. It inserts an event dependency between two streams and changes stream ordering, not tensor data; the risk is sharing one stream object across threads. `gptqmodel/utils/stream.py` caches its copy stream under `_STREAM_CACHE_LOCK` (`stream.py:74`, `:103-114`), while the TorchLinear dequant-stream cache (`qlinear/torch.py:856-862`) is unlocked. Use one stream per worker. |
| `torch.cuda.CUDAGraph` + `torch.cuda.graph` | SAFE (single-thread only); UNVERIFIED under GIL=0 | ParoQuant stage optimizer: `gptqmodel/quantization/paroquant/optimization.py:1960-1962` (`_run_stage_gptqmodel_cudagraph`) | Graph capture is process-global. Capture is reached only on a CUDA tensor stage with `fused_rotation` set, `inputs_train.shape[0] >= batch_size`, and `stage_cudagraph` not False (`optimization.py:1877-1889`); ROCm is excluded (`optimization.py:1881`). The looper disables it for the default `opt_scope="module"` (`gptqmodel/looper/paroquant_processor.py:673-683`, `gptqmodel/quantization/config.py:5764`), but `opt_scope` of `compute_block`/`layer` and the public `optimize_paroquant_linear(..., stage_cudagraph=None)` default (`optimization.py:1877-1878`) reach the capture. Capture on the thread that owns the model and replay from that thread's stream; do not capture or replay concurrently. The probe is single-shot (`scripts/gil0_api_probe.py:45`, `:123-124`) and was not run in this change. |

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
| `gptqmodel_nf12` encode/decode/decode_fp32/sub_fp32/linear/linear_fp32/linear_bf16_fp32/embedding/rowmajor_pack/rowmajor_embedding | SAFE (concurrent), NVIDIA CUDA | online BF16 weight preparation, fused `w - wq` diff, and inference | `scripts/gil0_api_probe.py --only gptqmodel_nf12.ops` covers canonical and row-major embedding, including the one-ID persistent-CTA row-major launch, the FP32 widening store, and the fused widening-subtract store. The canonical row passed four separate processes, each 8 threads × 4 iterations, on 2026-11-10 with Python 3.14.7 GIL disabled, PyTorch 2.15.0+cu132 and two NVIDIA PG506-230 SM80 GPUs; the `decode_fp32` row was re-run on 2026-11-14 with Python 3.14.10, PyTorch 2.16.0.dev20261102+cu130 and two RTX 4090s, and the `sub_fp32` check was added to the same probe. The probe prewarms JIT compilation, assigns worker devices explicitly, checks exact decoded/embedded values, `decode_fp32` versus `decode().float()` bit equality, `sub_fp32` versus `decode_fp32` with a zero `wq`, BF16/FP32 linear results, FP32 Tensor Core scores, and exact row-major versus canonical gather bits for one and multiple IDs, then synchronizes each worker stream. ROCm is explicitly unsupported by this CUDA package; GPT-QModel's separate HIP runtime remains its path. |
| `gptqmodel.utils.nf12_hip.embedding` / `linear` / `decode_fp32` / `sub_fp32` / `rowmajor_pack` / `rowmajor_embedding` | SAFE (concurrent), ROCm RDNA4 | canonical and production row-major BF16 embedding lookup, FP32 widening decode, fused `w - wq` diff, plus M16 linear inference | `scripts/gil0_api_probe.py --only gptqmodel_nf12_hip.embedding` passes canonical lookup, row-major lookup, `decode_fp32` bit equality against `decode().float()`, `sub_fp32` bit equality against `decode_fp32` with a zero `wq`, and cached-dispatcher M16 linear on two Radeon AI PRO R9700 GPUs with Python 3.14.10, PyTorch 2.16.0.dev20261025+rocm10.1 and `PYTHON_GIL=0`. JIT and both devices are prewarmed under a lock; workers select their device explicitly, use separate streams, and compare exact BF16 embedding bits including signed zero and invalid-ID rows against a dense Torch gather. The linear output is checked bit-exactly against the first decoded weight column with a one-hot activation. The probe covers read-only packed inputs, not simultaneous mutation, and does not qualify gfx1250 hardware. |
| `gptqmodel.utils.macaq_triton.mxfp4_w4a4_linear` | SAFE (concurrent) with cold JIT serialized, ROCm gfx1201 | MACaQ packed W4A4 inference | `scripts/gil0_api_probe.py --only 'MACaQ Triton W4A4'` passed three separate child processes on two Radeon AI PRO R9700 GPUs with Python 3.14.10t, PyTorch 2.16.0.dev20261025+rocm10.1, Triton 3.9, and `PYTHON_GIL=0`. Each process used eight threads × four calls, thread-local device selection and streams, and compared complete BF16 outputs exactly with serial prewarmed outputs. The probe covers read-only packed weights, the fused activation transform, activation packing and GEMM. Cold concurrent compilation is covered by the process-wide `_TRITON_COMPILE_LOCK` in `gptqmodel.utils.nogil_patcher` (manual finding 10); the probe still prewarms, and shared-weight mutation is out of scope. |
| `awq_gemm_kernel` (`gptqmodel.quantization.awq.modules.triton.gemm`) | SAFE (concurrent) with cold JIT serialized, ROCm gfx1201 | AWQ 4-bit GEMM inference | Issue #4094: `PYTHON_GIL=0`, empty `TRITON_CACHE_DIR`, 24 threads calling `generate()` on `AWQ-Llama-3.2-1B-g128-gemm` aborted 4/4 runs with `Critical nanobind error: inst_dealloc("triton._C.libtriton.ir.type")` and exit code 134. With `_TRITON_COMPILE_LOCK`, the same 24-thread cold run exits 0 in 4/4 runs (`tests/test_awq_triton_cold_jit.py` repeats it twice). Warm-cache runs were already safe. |
| `gptqmodel_nf12.prepare_fp32_gate` and packed FP32 B1 router | SAFE after JIT prewarm; CAUTION on cold Triton compilation | lossless compressed router preparation and staged verifier projection | The original probe passed 3/3 child processes on 2026-11-11, each 8 threads × 4 iterations, Python 3.14.7 GIL disabled, PyTorch 2.15.0+cu132 and two NVIDIA PG506 SM80 GPUs. On 2026-11-12, source-replacement checks passed 13/14 child processes with Python 3.14.10 GIL disabled, PyTorch 2.16.0.dev20261102+cu130 and two RTX 4090s. Three later unprewarmed children crashed with a Triton nanobind deallocation error during the replacement fallback; after the probe prewarmed that kernel on each GPU under its initialization lock, 3/3 children passed. The specific cold-compilation root cause is not proven. Preparation serializes cache publication and Torch indexing under `_gate_lock`; source revisions are checked at lookup. Mutating through `.data` or an inference tensor requires an explicit reprepare. ROCm is unsupported by this CUDA package. |
| Tensor `_version`, `data_ptr`, `device`, `dtype`, `shape`, `is_inference` | SAFE (concurrent), NVIDIA CUDA in measured build | NF12 prepared-view revision key | `scripts/gil0_api_probe.py --only 'Tensor cache revision metadata'` passed three child processes on 2026-11-12, each 8 threads × 4 iterations, on the same free-threaded Python 3.14.10, PyTorch 2.16.0.dev20261102+cu130, two RTX 4090s. The probe checks that ordinary `copy_` increments `_version` while identity/metadata stay stable, and that inference tensors report no version counter. It does not test concurrent mutation of one shared source tensor. |
| `gptqmodel_nf12.linear_fp32` Ampere prefill gate | SAFE (concurrent), NVIDIA CUDA | 384x5120 FP32 router prefill from NF12 weights | `scripts/gil0_api_probe.py --only gptqmodel_nf12.prefill_gate` passed three separate processes on 2026-11-11, each 8 threads × 4 iterations on two NVIDIA PG506 SM80 GPUs with Python 3.14.7 GIL disabled and PyTorch 2.15.0+cu132. The probe prewarms the Triton kernel, gives each worker a device and input, checks the FP32 output and bias, and synchronizes its stream. It covers `torch.cuda.get_device_capability`, call-local `nf12.decode`, `torch.empty`, and the Triton launch. ROCm takes the existing fallback. |

## MACaQ compiled quantization

| API | Result | Where we use it | Evidence and limit |
|---|---|---|---|
| `torch.compile` / `_compiled_flip_sweep` | SAFE when serialized, ROCm `gfx1201` | MACaQ default flip refinement | `scripts/gil0_api_probe.py --only 'MACaQ compiled flips ROCm'` passed three separate processes on two Radeon AI PRO R9700 GPUs with Python 3.14.10t, PyTorch 2.16.0.dev20261025+rocm10.1, and `PYTHON_GIL=0`. Each process used eight threads × four calls, including first-use compilation, thread-local devices and streams, and exact packed-code comparison to eager sweeps. The probe asserts that compilation actually occurred. `_COMPILED_FLIP_LOCK` serializes compiler creation and all compiled calls; this does not qualify unsynchronized compiler calls, other ROCm architectures, or free-threaded NVIDIA Python. |

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
| `torch.compile` / compiled MACaQ flip sweep | UNVERIFIED under GIL=0 | optional MACaQ quantization speed path | The port invokes compiled sweeps only when `sys._is_gil_enabled()` is true. Free-threaded workers use the eager sweep, which is also the numerical fallback. Qualify compiler creation and concurrent first-use on a free-threaded Torch build before enabling this path there. |
| `torch.einsum` | SAFE (concurrent) | math kernels | - |
| `torch.where` | SAFE (concurrent) | masking | - |
| `torch.maximum` | SAFE (concurrent) | clamping | - |
| `torch.clamp` | SAFE (concurrent) | quantization bounds | - |
| `torch.round` | SAFE (concurrent) | rounding | - |
| `torch.rsqrt` | SAFE (concurrent) | normalization | - |
| `torch.topk` | SAFE (concurrent) | act-order selection | - |
| `torch.argsort` | SAFE (concurrent) | act-order selection | - |
| `_deterministic_segment_sum` (VPTQ k-means) | SAFE (concurrent), CUDA and ROCm | VPTQ weighted k-means codebook reduction | `scripts/gil0_api_probe.py --only 'VPTQ deterministic segment sum'` passed three independent processes per backend with the GIL disabled, each 8 threads x 4 iterations on assigned devices with per-thread streams. The probe covers `torch.bincount`, `torch.cumsum` (float64 prefix), `Tensor.movedim` / `contiguous` / `double`, `Tensor.clamp_min`, and boolean-mask assignment, and compares every result with an independent float64 `index_add_` oracle (relative error < 1e-6). Empty buckets must be exactly zero and repeat calls must be bitwise identical. |
| `torch.sum` | SAFE (concurrent) | reductions | Reduction kernels use the current stream. |
| `torch.mean` | SAFE (concurrent) | reductions | - |
| `torch.all` | SAFE (concurrent) | validation | - |
| `torch.equal` | SAFE (concurrent) | tests | - |
| `torch.isfinite` | SAFE (concurrent) | NaN guards | - |
| `torch.isnan` | SAFE (concurrent) | NaN guards | - |
| `torch.bitwise_and` / `_or` / `_xor` | SAFE (concurrent) | quantization packing | - |
| `Tensor.index_select` row gather | SAFE (concurrent), NVIDIA CUDA | quantized embedding row decode (packed word row, shift, and group gathers) | Probed 2026-11-12 on Python 3.14.7t / Torch 2.15.0+cu132 with `PYTHON_GIL=0`, 8 threads x 4 iterations, three separate processes, all PASS. Each worker decoded 256 token rows on its assigned device from a golden-packed table built once and compared bit-exactly against a precomputed result; a torn or cross-thread read would differ. The AMD/HIP runtime is covered by the ROCm embedding row-decode test, not this CUDA probe. |
| `dequantize_gemm` tiled AWQ path | SAFE (concurrent), CUDA and ROCm | Torch AWQ and ParoQuant dense fallback | On 2026-11-13, `AWQ tiled dequantize_gemm` passed three independent subprocesses per backend with the GIL disabled, each running 8 threads x 4 iterations on assigned devices and per-thread streams. The independent FP64 oracle checks every output of 1100x2048 layers, group sizes 128 and -1, FP16/BF16/FP32 scales, signed packed words, and FP32 scales `1e-40` and `65536`. The composed probe covers bounded row tiles, complete/partial-group broadcasts, `torch.is_grad_enabled`, in-place integer subtraction, `torch.mul(..., out=...)`, and release of tile references. Each worker synchronizes its own stream before exact comparison. |
| `_scales_for_storage` FP16 GPTQ guard | SAFE (concurrent), NVIDIA CUDA | GPTQ pack workers | `GPTQ FP16 scale storage guard` passed three separate processes on 2026-11-12 with Python 3.14.7 GIL disabled, Torch 2.15.0.dev20260928+cu130, and two NVIDIA PG506 GPUs. Each process ran 8 threads x 4 iterations across both devices, checked exact finite FP16 cast results, and verified that mixed nonfinite, overflow, and underflow inputs were classified correctly. This probes the composed validation path, including its reductions and error formatting; ROCm concurrency was not measured in this run. |
| `_floor_scale_for_storage` and bounded GPTQ MSE search | SAFE (concurrent), CUDA and ROCm | GPTQ weight scale selection, including normalized smoothing domains | `scripts/gil0_api_probe.py --only 'GPTQ weight scale floor'` passed three separate processes per backend on 2026-11-13, each 8 threads × 4 iterations with the GIL disabled and explicit per-worker device selection. CUDA: Python 3.14.7 / Torch 2.15.0+cu132; ROCm: Python 3.14.10 / Torch 2.16.0.dev20261029+rocm10.1. Checks zero/arithmetic-underflow floors, symmetric/asymmetric MSE selection, tensor-valued normalized-domain bounds and exact reconstructed values. Independent quantization/packed-byte boundary tests cover FP16/BF16/FP32 inputs. |
| `_calc_affine_qparams` and `pseudo_quantize_dequant` affine fit | SAFE (concurrent), CUDA and ROCm | ParoQuant initialization and no-calibration fallback | `scripts/gil0_api_probe.py --only 'ParoQuant affine fit'` passed three independent processes per backend on 2026-11-16 with the GIL disabled. Each process ran 8 threads × 4 calls with explicit worker device selection and separate streams, covering FP16/BF16/FP32 extrema, `amin`/`amax`, FP32 promotion, `minimum`/`maximum`, normalized zero-point selection, scale fitting and output rounding. Every integer zero point and code matched an independent FP64 reference; outputs matched the independently rounded storage-precision oracle exactly. |
| `torch.floor_divide` | SAFE (concurrent) | packing | - |
| `torch.abs` / `exp` / `log` / `pow` | SAFE (concurrent) | math kernels | - |
| `torch.save` + `torch.load` | SAFE (concurrent), process-wide | checkpoints | File I/O; safe but serializes on the filesystem. Write distinct paths per worker. |
| `torch.set_default_dtype` / `torch.get_default_dtype` | UNSAFE (torn process state) when overlapped; serialized by `_MODEL_LOAD_LOCK` | transformers model construction in `ModelLoader.from_pretrained` / `from_quantized` | Process-global, not thread-local. transformers' `from_pretrained` saves/sets/restores it and enters meta-device and no-init contexts; two overlapping loads produced mixed dtypes, a meta `lm_head`, untied embeddings and a stuck default dtype (issue #4129, manual finding 11). All model construction holds `gptqmodel.models.loader._MODEL_LOAD_LOCK` (an `RLock`). |
| `torch.Generator(device="cuda")` | SAFE (concurrent), per-thread | deterministic sampling | One generator per worker. Never share a generator across threads. |
| `Tensor.to` (cuda:0 -> cuda:1) | SAFE (concurrent) (needs 2 GPUs) | fallback for every P2P fast path | Goes through the PyTorch copy machinery. Correct, but per-tensor and slower than `peer_copy_batch`; use it only as the fallback. |
| `Tensor.copy_` (non_blocking, cross-device) | SAFE (concurrent) (needs 2 GPUs) | P2P verify, Hessian adopt | Must synchronize the destination stream before reading the result. |
| `torch.inference_mode` | SAFE (concurrent), thread-local | inference paths | Thread-local grad mode; no shared state. |
| `torch.no_grad` | SAFE (concurrent), thread-local | calibration | Same as above. |
| `torch.is_autocast_enabled("cpu")` + `torch.autocast("cpu", enabled=False)` | SAFE (concurrent CPU) | GGUF Torch CPU FP32 reduction | On Python 3.14.10t / Torch 2.16.0.dev20261102+cu130 with `PYTHON_GIL=0`, the probe ran 8 threads x 4 iterations in three subprocesses. The active flag remained thread-local; the nested disabled context returned the exact FP32 dot `100000`, while surrounding BF16 dots stayed `99840`. |
| `torch.is_autocast_enabled("cuda")` + `torch.autocast("cuda", enabled=False)` | SAFE (concurrent CUDA) | NF12 wide fallback GEMM, Torch FP8 wide fallback | Probed 2026-11-10 on Python 3.14.7t / Torch 2.15.0+cu132 with `PYTHON_GIL=0`, 2x RTX 4090: 8 threads x 4 iterations, 3 subprocesses, all PASS. The CUDA autocast flag stayed thread-local and the nested disabled context produced the FP32 dot `100000` while the surrounding BF16 dots stayed `99840`. An 8-thread x 3-process stress of `NF12Linear`'s FP16 wide branch returned no hangs and no torn data; the bitwise differences observed were FP32-vs-FP64 rounding ties that also appear single-threaded. |
| `torch.get_autocast_dtype("cuda")` | SAFE (concurrent CUDA) | NF12 fallback predicate choosing FP16 vs wide | Probed 2026-11-10 on Python 3.14.7t / Torch 2.15.0+cu132 with `PYTHON_GIL=0`, 2x RTX 4090: 8 threads x 4 iterations in 3 subprocesses, three repeated runs, all PASS (probe `torch.get_autocast_dtype + torch.autocast (CUDA)`). The value stayed thread-local: `float16` outside a context, `bfloat16` inside one, unchanged across a nested `enabled=False` context. |

## Linear algebra and BLAS backend selection

| API | Result | Where we use it | Notes / workaround |
|---|---|---|---|
| `torch.linalg.cholesky` | SAFE (concurrent), ROCm R9700; UNVERIFIED on CUDA | GPTQ Hessian inverse `gptqmodel/quantization/gptq.py:1866`, `:1867`; QQQ `gptqmodel/quantization/qqq.py:822`, `:824`; VPTQ `gptqmodel/quantization/vptq.py:652`, `:659`; EXL3 `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py:438`; TT block-float `gptqmodel/utils/tt_blockfloat_quant.py:511`; GSQ `gptqmodel/quantization/gsq_training.py:1318`; background linalg warmup `gptqmodel/utils/linalg_warmup.py:88` | Issue #4207 reports a probe on 4x Radeon AI PRO R9700, Python 3.14t `PYTHON_GIL=0`: 8 threads x 6 iterations with per-thread streams, 1024x1024 fp32, bitwise equal to a single-thread run, 4/4 processes (2 cold, 2 warm). CUDA was not probed, so CUDA is UNVERIFIED. The GPTQ/QQQ/VPTQ calls run on looper device workers (`gptqmodel/looper/module_looper.py:183-187`, `gptqmodel/looper/weight_only_processor.py:145`); the background warmup is a separate thread and serializes under `_GLOBAL_WARMUP_LOCK` (`gptqmodel/utils/linalg_warmup.py:25`, `:119`). `scripts/gil0_api_probe.py` has no cholesky probe yet; add one and run it on the target build before relying on CUDA. |
| `torch.linalg.cholesky_ex` | SAFE (concurrent), ROCm R9700; UNVERIFIED on CUDA | GPTQ damp probe `gptqmodel/quantization/gptq.py:1844`; GSQ scalar `gptqmodel/quantization/gsq_scalar.py:79`; EXL3 diagnostics `gptqmodel/utils/exl3_remote.py:135`; ROCm quantize prewarm `gptqmodel/utils/linalg_warmup.py:204` | Same #4207 probe as `torch.linalg.cholesky`. `check_errors=False` makes a non-positive-definite input report through `info` instead of raising. CUDA UNVERIFIED. |
| `torch.cholesky_inverse` | SAFE (concurrent), ROCm R9700; UNVERIFIED on CUDA | GPTQ `gptqmodel/quantization/gptq.py:1865` (`out=` form), `:1868`; QQQ `gptqmodel/quantization/qqq.py:823`; VPTQ `gptqmodel/quantization/vptq.py:658`; TT block-float `gptqmodel/utils/tt_blockfloat_quant.py:618` | Same #4207 probe. The `out=` form reuses the factor buffer, so do not read the factor afterwards. CUDA UNVERIFIED. |
| `torch.backends.cuda.preferred_blas_library` (process-global getter/setter) | SAFE (serialized) through `quant_blas_backend`; UNVERIFIED if the setter is called outside it | Guard `gptqmodel/utils/rocm.py:267-277` (getter) and `:309-431` (`quant_blas_backend`); callers `gptqmodel/quantization/gptq.py:971`, `:983`, `:1821`, `gptqmodel/models/base.py:1966`, `:2260` | The setter is process-global, so `quant_blas_backend` holds `_QUANT_BLAS_GUARD_COND` (an `RLock`, `rocm.py:290-291`) and reference-counts `_QUANT_BLAS_GUARD_DEPTH`: same-target scopes share the pin and only the last exit restores it (`rocm.py:374-431`); scopes that need different targets wait instead of flipping the backend under a running GEMM (`rocm.py:354-372`). `tests/test_concurrency_guards.py:57` runs two threads and asserts the first exit does not restore. This is serialization, not a thread-safe setter: never call `torch.backends.cuda.preferred_blas_library(...)` directly from a worker; enter `quant_blas_backend`. The guard is a no-op on non-ROCm hosts or when the API is absent (`rocm.py:340-348`). No GIL=0 probe was run for this row; the status is by construction plus the CPU unit test. |

## Host thread-pool control (threadpoolctl)

| API | Result | Where we use it | Notes / workaround |
|---|---|---|---|
| `threadpoolctl.threadpool_limits` | UNVERIFIED under GIL=0; CAUTION (not reference-counted) | `gptqmodel/nn_modules/qlinear/__init__.py:2391` (`pack_original`); `gptqmodel/utils/bitblas.py:115` (`convert_to_bitblas`); `gptqmodel/quantization/rco.py:295`, `:300` (`_budget_dp`, raw `threadpoolctl` import) | `_ThreadpoolLimiter.__init__` snapshots the current limit and applies the new one; `__exit__` restores that snapshot, so overlapping scopes on different threads can restore each other's limit (issue #4207). `gptqmodel.utils.safe.THREADPOOLCTL` wraps the callable in a shared `threading.Lock` (`gptqmodel/utils/safe.py:49-52`), which serializes construction but not the `with` body or `__exit__`; `rco.py:295` imports the raw module. `tests/test_threadpoolctl.py:50` checks the effective count inside eight device-pool workers but does not overlap two live scopes. Worker-thread uses were removed and left commented at `gptqmodel/utils/threadx.py:613` and `gptqmodel/utils/model.py:1029`, `:1122`, `:1265`. This audit found no `threadpool_limits` call in a finaliser (`__del__`/`atexit`/`weakref.finalize`); the issue's finaliser-thread claim is UNVERIFIED. Do not overlap scopes from workers; if a scope is needed, hold an external process-wide lock around the whole `with` block. |

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

### 6. Pre-enabling peer access bypasses native allocator metadata

`hipDeviceEnablePeerAccess` against a native allocator running
`expandable_segments` grants access only to the segments that exist at that
moment. Segments reserved afterwards stay unreachable: the copy returns 0 and
moves nothing. Measured with two ROCm devices (gfx1201, ROCm 10.1, 4 MiB
payload, peer access pre-enabled with ctypes, source buffer allocated first):
`Tensor.copy_` copied nothing in 40 of 80 rounds (3 of 3 processes).

PyTorch grants access per segment from its own copy path
(`ExpandableSegment::addPeer` → `hipMemSetAccess`). Letting PyTorch own
enablement copied **80 of 80 rounds byte-exactly in 3 of 3 processes** with the
same `expandable_segments:True` setting, and the direct link stayed at 28.7 GB/s
against 22.7 GB/s for ordered host staging. So the fix is to stop pre-enabling,
not to abandon P2P — this is the default ROCm configuration.

The raw `hipMemcpyPeerAsync` call is still withheld there, because it is the
call measured to return success without moving bytes on that allocator. Probe
correctness and timing both go through `_copy_payloads_like_movers`, which
selects `Tensor.copy_` whenever the raw enqueue is withheld, so the probe
measures the path the movers use.

The same native-allocator rule is needed on CUDA. With manual runtime peer
enablement, a bidirectional mover test using distinct source/destination streams
and eight free-threaded workers intermittently raised `CUDA_ERROR_INVALID_VALUE`
for an inaccessible VMM operand. It failed once in the combined regression run
and again in an independent repeat. A control delegating enablement and
withholding raw copies passed three independent processes. `_enable_one` now
uses `_pytorch_owns_peer_enablement` before either vendor's runtime enablement
call, and `_raw_peer_copy_unsafe` withholds raw copies on native
allocators for both hardware families.

Allocator metadata must come from the active backend, not just the last parsed
setting. A real rejected `_accelerator_setAllocatorSettings` backend change left
`backend:cudaMallocAsync` as the parsed string while the live backend was still
`native`. `_allocator_is_native` trusts only the live backend. Native allocators
always delegate to PyTorch, independently of the parsed settings string.

A follow-up regression retained VMM tensors while changing allocator settings.
A partial or empty update omitted `expandable_segments` from the reported string
while the live setting remained true. Explicitly disabling expandable segments
left existing VMM allocations alive. A rejected update likewise could not prove
those allocations safe for raw copies. The four ownership assertions failed at
`f9707a0eb`; the corrected regression passes all four cases on CUDA and ROCm. `_pytorch_owns_peer_enablement` now delegates every native allocator
and unknown backend, so settings changes never reauthorize raw copies. This also
protects segments created after a later enablement of expandable allocation.

A non-native ROCm backend (`cudaMallocAsync`) is different: `Tensor.copy_` returned
mismatched bytes, and so did a plain same-device copy, so that configuration
fails closed and stages through pinned host memory
(`PeerStatus.reason_code == REASON_ROCM_VMM`).

Regression tests: `tests/test_p2p.py` covers the allocator classification, the
delegation, the withheld raw path, the probe path selection, both warning
wordings, and the mixed-batch ordering.

### 7. The P2P allocator guard must be thread-safe metadata

Cross-device raw-copy attempts and `enable_peer_access` consult the allocator
guards, so they run on free-threaded workers on both CUDA and ROCm. The guards
only read the active backend (`torch.cuda.get_allocator_backend`) and the P2P
config. They never issue the raw peer copies they suppress. The settings-string
reader (`torch._C._accelerator_getAllocatorSettings`) has been removed from the
threaded path because it cannot describe the lifecycle of existing allocations.

| API | Result | Where we use it | Notes / workaround |
|---|---|---|---|
| `gptqmodel.p2p.allocator_guard` (`_allocator_is_native`, `_raw_peer_copy_unsafe`, `_rocm_peer_link_unusable`, `_pytorch_owns_peer_enablement`, `_rocm_nested_peer_disabled`) | SAFE (concurrent), metadata only | raw-copy attempts, `enable_peer_access` and `_enable_one` | The probe now reads metadata on both CUDA and ROCm, including the explicit nested-copy policy. It compares all four guard verdicts across 8 threads x 4 iterations. It never issues a peer copy or pre-enables runtime peer access. On 2026-11-13, after removing settings-string authorization, the updated probe passed 3/3 independent processes on each of CUDA and ROCm with `PYTHON_GIL=0`. The mover fallback test separately passed 3/3 independent processes per backend, each with 8 workers and 32 bidirectional tasks on per-thread streams. Native raw copies are now withheld even with expandable segments disabled; the nested-copy opt-in cannot override this guard. |
| `gptqmodel.utils.p2p` policy snapshot + status cache (`_policy_snapshot`, `_publish_status`, `_STATUS_CACHE`) | SAFE (concurrent) | `configure_p2p`, `enable_peer_access`, `_cached_status_for`, `clear_p2p_cache` | `_CONFIG_LOCK` is an `RLock` guarding the active policy, the monotonic `_POLICY_GENERATION`, and `_STATUS_CACHE`. A verdict is bound to the generation it measured under: the cache key starts with that generation, and a probe that finishes after a concurrent `configure_p2p` is returned to its own caller but not published. `configure_p2p` publishes the new policy and clears the cache in one critical section, so a late writer cannot reinsert an entry the clear removed. Regression: `tests/test_p2p.py::test_policy_change_during_a_probe_does_not_cache_a_stale_verdict` parks a probe inside the bandwidth measurement, installs a stricter `min_gbps` from another thread, and asserts the next call re-probes to `degraded` instead of reusing the stale `enabled` object. Passed 5/5 under `PYTHON_GIL=0` (`sys._is_gil_enabled() is False`) on 2026-11-14. |

### 8. DeviceThreadPool task failures propagate through futures

`_DeviceWorker._run` used to call `os._exit(1)` after setting a task
exception on its future. That bypassed every `finally`, `atexit` handler,
checkpoint signal restore, and caller `try/except`; whether the main thread
observed the future first was a race. Warmup failures took the same path, and
`_SyncWorker.submit` already only set the exception. Both paths now record the
exception on the pending future and keep the worker alive. The shared
`_WorkerWarmupState` caches a warmup failure, so later tasks on that worker
report the same exception instead of hanging.

`LoopProcessor.log_save_async` is fire-and-forget, so it now attaches a
done-callback that logs a failed write instead of silently dropping it.

Regression: `tests/test_threadx_error_propagation.py` runs a free-threaded
subprocess that fails a normal task and a warmup, asserts both futures raise,
asserts a later task still succeeds, and asserts an `atexit` marker was written.
It fails on the pre-fix code because the subprocess exits 1 before the marker.

### 9. Triton autotuner `nargs` is thread-local

`Autotuner.run` and `CustomizedTritonAutoTuner.run` used to write the current
call's argument mapping to `self.nargs` on the shared kernel object. A second
thread running the same kernel overwrote it while the first thread was still
inside `_bench`, `prune_configs`, `pre_hook`, `reset_to_zero`, or
`restore_value`, so a cold-key benchmark could run against another thread's
tensors. `best_config` was shared the same way.

The process-wide Triton patch now installs a data-descriptor `nargs` and
`best_config` backed by a per-instance `threading.local()`; the shared `cache`
stays guarded by `_cache_lock`. The GPTQ GGUF autotuner passes `nargs`
explicitly to `_bench`/`prune_configs`, keeps a thread-local `nargs` property,
and guards its cache check-then-set with `_cache_lock`.

Regression: `tests/test_triton_autotuner_concurrency.py` parks one thread in a
fake `_bench`, starts a second thread on the same kernel, and asserts the first
thread still observes its own `nargs`. Both tests fail on the pre-fix code and
pass with `PYTHON_GIL=0` (`sys._is_gil_enabled() is False`).

### 10. Triton JIT compilation is serialized process-wide

`TritonPatch` used to patch only the autotuner. With an empty
`TRITON_CACHE_DIR` and `PYTHON_GIL=0`, several threads missing the same kernel
cache entry entered libtriton's compiler together. libtriton is not
free-thread safe: the process died with
`Critical nanobind error: inst_dealloc(...): attempted to delete an unknown
instance` and exit code 134 (SIGABRT). On ROCm gfx1201, 24 threads launching
`awq_gemm_kernel` aborted 4/4 cold runs; the same shape aborted 4/4 at 32
threads. GIL=1 and warm-cache GIL=0 runs were clean.

`patch_triton_jit_compile()` now wraps `JITFunction._do_compile` with
`_single_flight_triton_compile`: it takes `_TRITON_COMPILE_LOCK`, rechecks the
per-device kernel cache for the specialization key, and compiles only on a
real miss. Same-key misses collapse into one compile; different keys are
serialized because the unsafe state is inside libtriton, not one JITFunction.
The lock is only taken on a cache miss, so steady-state launches are unchanged.
The MACaQ Triton kernels use the same `JITFunction.run` path and are covered by
the same patch.

Regression: `tests/test_awq_triton_cold_jit.py` runs a `PYTHON_GIL=0` child
with an empty `TRITON_CACHE_DIR` and 24 concurrent `generate()` calls, twice,
and asserts exit 0. `tests/test_triton_patch_api.py` pins single-flight for one
key and serialization across distinct keys.

### 11. transformers model construction mutates the process-global default dtype

`transformers`' `from_pretrained` saves, sets and restores
`torch.set_default_dtype` and builds parameters through the meta-device and
no-init contexts. Those are process-global, not thread-local. Two dense
`GPTQModel.load()` calls in different threads interleaved them: on an R9700
with Python 3.14.10t, Qwen2.5-0.5B and Qwen3-0.6B at `dtype=float16` came back
with mixed fp16/fp32 parameters (252 fp16 / 38 fp32 in the first round), a meta
`lm_head.weight`, untied embeddings and `torch.get_default_dtype()` stuck at
fp16. Measured 6/6 corrupted rounds at `PYTHON_GIL=0` and 5/6 with the GIL on;
the same loads serially were clean.

`ModelLoader.from_pretrained` and `from_quantized` now hold `_MODEL_LOAD_LOCK`
(an `RLock`) for the whole construction, and `assert_loaded_model_state`
rejects a mixed-dtype, meta or stuck-default-dtype result. Do not call
`torch.set_default_dtype` from a worker that can overlap a model load.

Regression: `tests/test_concurrent_load.py` (two threads, six rounds, both GIL
modes) and `tests/test_load_state_assertions.py`.

## Adding a probe

1. Add the callable to `build_probes()` in `scripts/gil0_api_probe.py`.
2. Run it on a free-threaded build with two or more GPUs:
   `PYTHON_GIL=0 python scripts/gil0_api_probe.py --only <name>`.
3. Repeat it several times. A single PASS is not evidence.
4. Add the row above, with the workaround if the result is not SAFE.

## CI enforcement

A `3.14t` job that does not set `PYTHON_GIL=0` can re-enable the GIL while
importing `triton._C.libtriton` and then run fully serialized: every test that
needs a free-threaded interpreter skips, and the build stays green. The default
matrix in `.github/workflows/unit_tests.yml` keeps the GIL because some
test-only libraries still need it; the dedicated `free-threaded` job passes
`free_threaded: true` to `.github/workflows/unit_tests_reusable.yml`, which
exports `PYTHON_GIL=0` for every later step and runs
`.github/scripts/ci_assert_free_threaded.py` after the package is installed.
That gate fails the job unless `sys._is_gil_enabled()` is still `False` after
importing `torch`, `gptqmodel`, `triton`, and `gptqmodel.utils.p2p`, so a
silently serialized lane is a red build instead of a skipped one.

The lane runs bounded concurrency tests: `tests/test_threadx_cpu_locks.py`,
`tests/test_threadx_shutdown.py`, `tests/test_looper_concurrency.py`, and
`tests/test_checkpoint_store.py` on CPU; `tests/test_stream.py` and
`tests/test_thread_pool_exit.py` on one GPU; `tests/test_p2p.py` and
`tests/test_p2p_copy_utils.py` on two GPUs for the P2P and native-kernel
ownership paths. `tests/test_ci_free_threaded_lane.py` pins the workflow wiring
and checks that the gate really fails when the GIL is enabled.

## Where P2P is used

`gptqmodel.utils.p2p.enable_peer_access` / `enable_for_devices` probe and
enable every ordered pair of the GPUs a run actually uses, cache the result
under the policy generation that produced it (a `configure_p2p` from another
thread invalidates every verdict it did not measure), and log one actionable
warning when a pair cannot be enabled (with the
motherboard/BIOS/hardware compatibility wording). `peer_copy_batch` then backs
module replication and per-layer moves, calibration input staging, Hessian
partial reduction, and shared-Hessian adoption. Control it with
`BaseQuantizeConfig.p2p` (or `GPTQModel.load(..., p2p=...)` for inference):
`P2PConfig(mode="off")` skips probing, and `P2PConfig(probe=True)` additionally
measures each directed link. `P2PConfig` is a no-op for CPU-only runs and for
CPU<->CPU transfers: CPU devices are skipped, no warning is emitted, and the
move helpers keep their regular copy path.
