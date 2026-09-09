# Source map and transfer limits

Reviewed source: [cohere-ai/cohere-megakernel at 67d0b9c](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c),
2026-09-09. The short link supplied by the user resolves to this repository.
These notes are a distillation, not vendored implementation. Upstream is
Apache-2.0; preserve applicable license notices if later copying code.

All links below are pinned to the reviewed commit. Symbol names make it possible
to find the relevant mechanism after line numbers change.

| Source | Observed mechanism | Lesson for a port |
| --- | --- | --- |
| [README](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/README.md) | One persistent block per SM executes a decode task list; the model has independent attention/FFN branches. Release scope is single H100, SM90a, BF16, small batches. | Reconstruct the target model DAG; hardware gates and reported gains are source-specific. |
| [schedule.py](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/schedule.py), `_build_nmc_decode_named_waves`, `_nmc_barrier_dims` | Tasks carry explicit dependency thresholds; QKV readiness is tracked at KV-head granularity. Named waves reject built-but-unplaced work. | Count actual producers and validate schedule completeness rather than inserting blanket operator barriers. |
| [schedule.py](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/schedule.py), `ATTN_DYNAMIC_SENTINEL`, `attn_drain` | Full attention uses shared drain queues with live per-row split counts; sliding-window work retains static scheduling. Context variants are prepared ahead of time. | Match scheduling complexity to runtime imbalance and keep wait targets consistent with the tasks actually emitted. |
| [megakernel.cuh](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh), `controller_loop`, `worker_loop`, `NmcRole` | A controller prefetches fixed-width instructions into a ring; producer, storer and compute roles have separate compile-time call paths. | Define a small task ABI and reader-completion protocol before composing kernels; verify role-specific resource allocation. |
| [megakernel.cuh](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh), `PREFETCH_STAGES`, `empty_bar` | Weight TMA can precede activation waits. QKV stage release is per warp because peer warps can still be reading. | Prefetch immutable operands only; shared stages remain live until every reader finishes. |
| [schedule.py](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/schedule.py), `moe_prefetch_stages` | The default disables MoE prefetch at batch one after a small-batch regression. | Prefetch depth is an operation/workload tuning parameter, not an unconditional improvement. |
| [megakernel.cuh](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh), `worker_sync`, `wait_cross_sm`, `arrive_cross_sm` | Worker-only barriers exclude the controller; NOP paths preserve matching barrier calls. Counter publication includes async-proxy and device fences; waits poll with nanosleep. | Barrier phase alignment and store visibility are separate invariants. Re-prove ordering for the target rather than copying this low-level protocol blindly. |
| [launch.cuh](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/launch.cuh) | The launch uses `num_sms` blocks and the compiled shared-memory footprint. | Verify simultaneous progress on the actual device; a small grid alone is not a deadlock proof. |
| [abi.h](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/abi.h), `NmcLaunchDesc`, `DecodeService` | The launch descriptor crosses a separately compiled boundary. Bulk batch mutation requires the decode loop to be parked; raw pointer owners must stay alive. | Keep ABI evolution, ownership and completed pause handshakes explicit in a native integration. |
| [runtime.cu](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/runtime.cu), `nmc_zero_regions_kernel`, `synchronize_decode_step_with_watchdog` | Scratch/control state is reset between steps. Watchdog diagnostics are preallocated; a timeout cannot cancel a wedged GPU kernel. | Include reset costs and route-transition tests; never free live buffers merely because the host timed out. |
| [SM profiler notes](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/sm-profiler/README.md), plus `profiled_wait_cross_sm` | Per-SM traces use low-overhead range handles; task ranges can pause during dependency waits. The profiler documents inaccurate events immediately after block barriers. | Interpret gaps and ranges using their instrumentation semantics and confirm gains without instrumentation. |
| [test_decode_layers.py](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/tests/test_decode_layers.py), [test_kv_bindings.py](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/tests/test_kv_bindings.py), [test_token_callback.py](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/tests/test_token_callback.py) | Tests cover layer comparisons, KV lifecycle/masking/rebinding and callback exception/lifetime/concurrency behavior. | Kernel output checks alone do not validate serving integration. Select tests around the boundary being changed. |

## Claims that need qualification

The README reports a 1.58x batch-one decode speedup and 1.25x–1.41x end-to-end
serving speedups against its stated vLLM baseline. These are upstream reports,
not measurements performed while creating this skill. Decode-only measurements
use synthetic KV values, while serving measurements include real prompts and
prefill. Upstream notes differing generated token counts and variable expert
routing. Do not compare these numbers as if they were the same experiment.

“One kernel” describes the main decode forward region. The source explicitly
places embedding/initial normalization outside it, and the native runtime also
performs state reset and sampling. Prefill remains separate and pauses decode.
Measure the whole target operation before claiming a one-launch decoding system.

The skill's QvQ compressed-weight guidance and ZML graph-capture requirements
are transfer guidance based on the target repositories, not features proven by
Cohere. In particular, this review establishes no upstream or target CUDA Graph
compatibility and no performance result on another architecture.
