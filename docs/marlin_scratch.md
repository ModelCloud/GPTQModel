# Marlin caller-owned temporary buffers

The GEMM wrapper accepts `c_tmp=None, a_tmp=None` at the end of its original
argument list. Both FP16 and BF16 torch.ops schemas have the same extension.
Absent buffers retain the original C++ allocation behavior. Provided buffers
are passed to the CUDA GEMM itself; outputs are still freshly allocated.
Kernel tiles, packing, arithmetic and reduction precision are unchanged.

## Ownership and capacity

`marlin_scratch_sizes(a, M, K, use_fp32_reduce, has_act_order)` returns **element
counts**, not bytes. The query and GEMM use one shared C++ implementation:

* Reduction: `SMs * min(ceil(M / 16) * 16, 64) * max_thread_n` FP32 elements.
* Permutation: `M * K` elements of the activation dtype.
* An inactive feature, or M=0, needs zero elements for its corresponding buffer.

Here K is the **executed K**, including padding. Python caches query results,
but does not duplicate the CUDA capacity formula. Wide integer arithmetic and
overflow checks protect element/byte calculations and the existing integer
kernel interface.

C++ checks every supplied nonempty scratch for device, dtype, contiguity,
capacity, 16-byte address alignment, and shared storage with every GEMM tensor
argument and the other scratch. Shared storage is conservatively rejected even
when views do not overlap. Invalid supplied buffers produce explicit errors;
they do not silently fall back. Empty inputs retain the existing short circuit.

The original per-layer `workspace` holds reduction locks and is separate from
both scratch buffers. It cannot substitute for either. Its existing ownership
also means ordinary module forwards must not be assumed safe across streams.

## Opt-in eager reuse

```python
from gptqmodel.utils.marlin_scratch import MarlinScratchContext

scratch = MarlinScratchContext("cuda:0", max_cached_bytes=64 << 20)
with scratch:
    output = model(inputs)
    output2 = model(other_inputs)
scratch.clear()
```

A context binds one CUDA device, one stream and one host thread. Sequential
layers share one growing reduction buffer and one activation buffer. It also
owns a separate zero-initialized lock workspace. Dtype changes isolate the
activation buffer. The default **64 MiB total retained-byte limit includes
locks**. Requests that cannot fit use temporary buffers; retained capacity
never grows without the limit. The limit does not bound outputs, allocator
reserved memory or temporarily outstanding allocations during growth/clear.

Capacity hits do not allocate new scratch; growth occurs only when needed.
The default path remains unchanged until the caller explicitly enables reuse,
because a stable latency benefit has not yet been established on a GPU.
No scratch is stored in a module or checkpoint. Reinitialization and new weights
therefore cannot leave a cache tied to old parameters. Device moves require a
context for the target device; scratch is not silently migrated. `clear()`
releases retained ownership while CUDA allocator stream tracking protects
queued asynchronous uses. No per-call device synchronization is introduced.

Reentrant use, a different thread/stream/device, and nested active contexts are
rejected. This release makes **no concurrent-forward support claim**.

## CUDA Graph boundary

The eager context explicitly rejects capture. Its dynamic cache is not a graph
ownership mechanism. Low-level caller-provided scratch may be captured only
when the caller precompiles/warms the extension, allocates sufficient capacity
before capture, and retains exclusive ownership of scratch **and lock
workspace** for the entire graph lifetime. Do not reuse a graph's storage for
another graph or eager execution that can overlap it. The caller must retain
all tensor owners and observe CUDA stream lifetime rules. The graph test uses
ten replays with changing inputs, rather than a single successful capture.

No GPU is available in this development environment, so numerical correctness,
low-level graph replay and performance remain **unverified on GPU**.

## Why not allocate scratch per layer?

Illustrative costs, not measurements: with 132 SMs and `max_thread_n=256`,
maximum reduction scratch is **8.25 MiB**. At M=2048 and K=28672,
FP16/BF16 permutation scratch alone is **112 MiB**.

For an illustrative 80-layer model with seven projections per layer, six with
K=8192 and one with K=28672, permanent M=2048 scratch for every projection would
consume about **28.26 GiB** (4.51 GiB reduction plus 23.75 GiB permutation).
This excludes weights, outputs, locks and allocator overhead. Shared bounded
execution-context reuse avoids that per-projection commitment; large requests
may still temporarily allocate beyond 64 MiB.

The context's device must be the current CUDA device at entry. For a different
device, enter `torch.cuda.device(target_device)` before entering its scratch
context. Affinity remains fixed across repeated entries into the same context.

## Reproduction and measurement

```bash
python -m pytest -q tests/test_marlin_jit.py tests/test_marlin_permute.py \
    tests/test_marlin_scratch_capacity.py tests/test_marlin_scratch_interface.py \
    tests/test_marlin_scratch_context.py tests/test_marlin_scratch_regressions.py \
    tests/test_marlin_scratch_cuda.py
python scripts/benchmark_marlin_scratch.py --list-cases
python scripts/benchmark_marlin_scratch.py --graph --output /tmp/marlin-scratch-ab
```

The default matrix has 198 cases: FP16/BF16; GPTQ with/without act-order and AWQ
4-bit without act-order; M=1,8,16,17,32,64,128,512,2048; square, narrow output,
long K and padded shapes. Padded act-order is excluded because the existing
Marlin layer does not support it. FP32 reduction is on throughout the performance
matrix. Correctness tests additionally cover reduction off, GPTQ 8-bit and
aligned AWQ 8-bit. Padded AWQ 8-bit remains unsupported by the existing layer.

Each case uses the same layer, weights, inputs, kernel dispatch, dtype and
precision. An independent dense reference checks logical codes/group IDs before
packing. Randomized paired rounds record raw and median eager results:

* CPU enqueue cost measured without waiting inside the timed submission loop.
* Full forward wall time including completion of queued GPU work.
* GPU event span, separately from profiler-summed kernel execution time.
* Allocator allocation requests/bytes from allocator history and actual driver
  allocation events from a Chrome profiler trace. `aten::empty` counts are
  included as an auxiliary observation, never interpreted as cudaMalloc counts.
* Peak/process resident bytes and retained context bytes released by clear.
* First allocation and growth costs, excluding extension compilation.
* Optional sequential graph replay using exclusively owned explicit scratch.

Graph capture already provides stable allocation addresses for the legacy
temporary path. Replay speedup must be measured; eager allocation savings do not
predict graph replay speedup. Context entry is outside the repeated timed loop,
matching a context around sequential model execution.

The benchmark writes `results.json`, source revision/hash/diff provenance and
per-case profiler traces. Missing CUDA produces one explicit `not_run` record
per case. Failures remain failures. All performance results in the accompanying
CPU-only run are `not_run`; there is no measured speedup to report.

## Validation record (2026-09-17)

The combined command above produced **88 passed, 87 skipped** in 10.45 seconds
with Torch 2.13.0+cu130 and no available CUDA device. Skips are GPU tests. The
capacity test compiles and executes the actual shared C++ helper, including
counts greater than 32 bits and overflow failures. CPU tests verify dispatch,
schemas, actual padded dimensions, allocator hits/growth, retained-byte bounds,
partial fallback, clear, dtype changes, loader/backend exceptions and affinity.

Both dispatch `.cpp` files passed g++ syntax checks. The actual GEMM host body
and capacity query also passed a syntax check with kernel symbols declared as
stubs; this is explicitly separate from CUDA tile compilation. Launcher CUDA
translation units compiled with nvcc for SM80 using the current Torch C++11
ABI setting (`_GLIBCXX_USE_CXX11_ABI=1`). Full generated tile compilation, linked
extension loading, GPU numerical tests and graph replay were not performed.

A pre-change run of `tests/test_marlin_import_min.py` failed because that test
unconditionally loads CUDA despite CUDA being unavailable. This existing
environment-dependent failure is excluded from the combined CPU acceptance
command; the test has not been modified.

[marlin_scratch_validation.json](marlin_scratch_validation.json) contains the
test/compile record and all **198 individual A/B case statuses**, including
source revision and hashes. Every performance case is `not_run` because CUDA
is unavailable. Profiler/allocator reductions and latency improvements remain
unmeasured, so reuse remains opt-in.
