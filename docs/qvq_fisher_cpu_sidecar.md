# Dual-Xeon CPU sidecars for Sketch-B / Fisher collection

Status: **active investigation targeting at least 1.5x CPU/GPU hybrid speedup**.
The hardware/AMX availability probe and 16/32/64/96-thread baseline sweep are
complete. oneDNN setup, real-tensor capture, and crossover probes are underway. The tracked queue is
[`experiments/qvq_fisher_cpu_sidecar_queue_20260905.json`](experiments/qvq_fisher_cpu_sidecar_queue_20260905.json).

The objective is additional end-to-end acceleration by overlapping suitable CPU
work with GPU forward/backward and Fisher accumulation. Compare every candidate
against code commit `1acd13b95bcb2c4bfc3556d7e710da1385f911f5`, which already
contains overlapped factor offload and Hopper FP32 projection tiles. Comparing
against the older pre-optimization collector would double-count existing gains.
The preceding measurements are in [the collection report](qvq_yaqa_collection_overlap.md).

## Expanded allocation verified on 2026-09-05

The updated effective cpuset and process affinity expose **96 logical CPUs**,
representing **71 physical cores** (34 on socket 0, 37 on socket 1). Twenty-five
physical cores have both SMT siblings in this allocation. This is enough to begin
substantial dual-socket experiments; it is not all 96 physical host cores.

| NUMA node | Allowed logical CPUs |
|---|---:|
| 0, GPU-local sysfs hint | 23 |
| 1 | 24 |
| 2 | 28 |
| 3 | 21 |

The CPU cgroup has no time quota (`cpu.max = max 100000`); memory remains capped
at 128,000,000,000 bytes. AMX BF16 and INT8 execution checks passed again after
the allocation change. The queue preserves the original allocation in
`hardware_history` and records the current per-core sibling groups.

Use actual core groups for worker placement and reserve GPU-submission capacity.
The new benchmark options `--cpu-threads` and `--cpu-interop-threads` make thread
counts explicit; outputs now include affinity, physical-core groups, NUMA lists,
cgroup limits, and thread-pool environment settings.

## Initial allocation, before the host update

| Item | Observation |
|---|---|
| CPU | Two Intel Xeon Platinum 8575C sockets |
| Advertised topology | 48 cores/socket, 96 physical cores, 192 logical CPUs |
| Initial allocation | **16 logical CPUs**, constrained by effective cpuset and process affinity |
| NUMA | Four advertised nodes; all four memory nodes are available to this cpuset |
| ISA flags | AVX-512, AVX-512 BF16, AVX-512 VNNI, AMX tile/BF16/INT8 |
| AMX execution | Native BF16 and INT8 tile operations executed and returned the expected results |
| GPU locality hint | H200 at PCI `0000:1c:00.0`; sysfs associates it with node 0 |
| Initially local allocated CPUs | Only CPU 17 intersects the GPU-local CPU list in this allocation |

`lscpu` describes the larger host topology; it does not grant access to those
cores. The initial affinity/cpuset was:

```text
17,30,32,56,64,66,72,84,90,121,123,129,138,147,155,178
node 0: 17
node 1: 30,32,121,123,129,138
node 2: 56,64,66,147,155
node 3: 72,84,90,178
```

`nvidia-smi topo`/hwloc reported inconsistent virtualized topology. Treat sysfs
locality as a starting hypothesis and verify it with actual pinned-buffer transfer
measurements. Re-probe allocation and topology before every experiment. Full
96-core/192-thread scaling requires an allocation exposing the intended cores;
no host-wide affinity, cpuset, CPU-online, or memory-policy changes were made.

The Linux x86-64 [AMX probe](../scripts/probe_qvq_cpu_amx.c) requests tile-state
permission in its own process, executes `tdpbf16ps` and `tdpbssd`, and checks all
256 output elements for each operation. GCC 15.2.0 compiled it successfully;
objdump also confirmed both instructions. The returned permission masks were
`0x202e7` before and `0x602e7` after the request. This proves availability, not
application performance or model quality.

```bash
gcc -O2 -Wall -Wextra -mamx-tile -mamx-bf16 -mamx-int8 \
  scripts/probe_qvq_cpu_amx.c -o /tmp/qvq-amx-probe
/tmp/qvq-amx-probe
```

## Ordered experiment queue

| ID | Experiment | Concrete question and measurement |
|---|---|---|
| cpu00 | Hardware and ISA availability — complete | What CPU allocation, NUMA hints, and executable instruction sets exist? |
| cpu01 | Critical-path attribution | How much wall time remains in host finalization, launch gaps, copies, GPU math, and synchronization? Bound the attainable gain before implementing workers. |
| cpu02 | NUMA-aware transfer buffers | Do GPU-local first-touch allocation, reusable pinned buffers, and bounded handoff improve net collection time? Compare local, other-node, and interleaved placement within the allocation. |
| cpu03 | Exact batch metadata pipeline | Can preparation of boolean masks, integer indices, counts, and next-batch staging overlap current GPU work? Preserve valid-token order, loss semantics, seed scheduling, and accumulation order. |
| cpu04 | Parallel host factor finalization | Can CPU workers normalize already-host diagonals, perform remaining consistency checks, and construct factor descriptors while later modules backpropagate? Preserve existing GPU source-diagonal reductions initially. |
| cpu05 | Small FP32 operation crossover | Which real small-module contractions or reductions beat the GPU after packing, D2H/H2D, queue, and synchronization costs? Start with AVX-512 FP32 and the exact collector reference. |
| cpu06 | AMX BF16 / oneDNN | Verify actual framework dispatch, then measure naturally BF16 work and explicitly separate precision experiments. Include layout conversion and packing amortization. |
| cpu07 | AMX INT8 / VNNI | Test naturally integer work where available. Any conversion of floating Fisher data to INT8 is a separate estimator/precision experiment with its own reference artifacts. |
| cpu08 | Combined sidecar confirmation | Combine only independently successful candidates; verify factor/quantization parity, resource bounds, and net end-to-end improvement. |
| cpu09 | Full dual-socket scaling | After a suitable CPU allocation is available, compare physical-core scaling, socket placement, and SMT; do not extrapolate from the current 16-CPU cpuset. |

The current collector uses FP32 statistics and Gaussian projections. BF16/INT8
AMX throughput is not an automatic replacement for those operations. Keep the
strict FP32 lane and precision experiments distinct. A CPU reduction or a new
BLAS backend can also change rounding even when its dtype remains FP32.

Do not move Gaussian generation to the CPU merely because cores are idle:
matching the seed alone does not establish identical random values, and moving
large projections may increase PCIe traffic. Likewise, do not reintroduce the
full host factor scans eliminated in the previous optimization.

## Proposed overlap and worker design

The first prototype should consume work that is already destined for the CPU:

```text
GPU backward/sketch -> copy-stream completion event -> bounded CPU work queue
       |                                                   |
       +-> later modules continue                          +-> normalize/check/build
                                                                   |
                                                       ordered result collection
```

- Reserve allocated CPU capacity for GPU submission. Compare a small persistent
  worker pool with native batched parallelism; avoid per-layer thread creation.
- Enqueue CPU reads only after the relevant copy event completes. Track buffer
  ownership, errors, cancellation, and result readiness explicitly. Return no
  factor until the collector's existing validity checks and final drain succeed.
- Bound queue depth and pinned memory. Record backpressure and drain time rather
  than hiding them outside the timed region. Avoid serializing entire tensors
  through a Python process queue.
- Benchmark module-level parallelism versus within-operation threading. Prevent
  nested OpenMP/BLAS oversubscription; do not repeatedly mutate global Torch
  thread settings while other workers are active.
- Choose worker masks from the actual cpuset and core siblings. Sweep small
  worker counts first; using every logical CPU is not the acceptance metric.
- If process isolation is needed, initialize workers with a CUDA-safe creation
  method and verify their AMX permissions and shared-buffer ownership explicitly.

Keep large GPU contractions on the GPU unless full-path crossover measurements
show otherwise. CPU preprocessing, copies, and useful compute must overlap the
GPU critical path to produce a collection speedup. Downstream Hessian
materialization or quantization may be a separate opportunity, but report its
latency separately from collection.

## Fixed measurements and promotion gates

1. Re-establish the committed GPU-only control with the same CPU allocation,
   affinity, Torch/BLAS threads, model, backend, data, and seed as each sidecar arm.
   Preserve both the earlier B4/16-row and B16/64-row, 400-target, T64/R256 cases;
   expand to longer sequences and more batches before broader claims.
2. Use real captured activations/gradients from small modules for crossover
   decisions, then replay all 400 Qwen3.5-27B proxy targets. Synthetic inputs are
   suitable for ISA, algebra, and bandwidth checks, not model-quality decisions.
3. Record warmed wall-time distributions, CPU compute/queue/drain time, copy
   bytes and time, GPU gaps and throughput, memory placement, pinned memory,
   resident memory, and worker utilization. Include all packing, staging, and
   handoff costs. Use paired confidence intervals for net improvement.
4. Require exact factor parity first: sources, exact diagonals, cached source
   diagonals, normalizers, and seeds. Quantization implementation changes must
   preserve deterministic packed codes, bank IDs, and serialized metadata.
   Keep failures and precision candidates separate from production defaults.
5. Confirm actual CPU dispatch with oneDNN logs and instruction evidence; inspect
   CPU SIMD/AMX code and profile changed GPU instructions when applicable. Re-run
   correctness and unprofiled timing after each accepted implementation phase.
6. Exercise weighted/masked batches, small normalizers, repeated captures,
   non-finite rejection, worker/copy failures, cleanup, and non-default streams.
   Retain CPU/non-target GPU fallbacks. A busy CPU or fast standalone AMX GEMM
   does not constitute a successful sidecar result.

The queue is a persistent research backlog. The initial expanded-allocation
baseline sweep is recorded below; further sidecar prototypes require separate
measured trials. CPU09 can proceed within the expanded allocation, while claims
about all 96 physical host cores still require that full physical-core allocation.

## Expanded-allocation baseline sweep

All runs use the unchanged committed GPU collector (source hashes recorded in
JSON), all 96 allowed logical CPUs, one Torch inter-op thread, and the specified
intra-op thread count. Qwen3.5-27B geometry proxy, BF16 model, FP32 statistics,
all 400 targets, B16 / 64 sequences at rows 96–159, seed 20260909, T64/R256,
CUDA accumulators, no activation checkpointing. Each independent process has
two warmups and three measured repeats. Model loading is excluded; complete
collection, host factor delivery, and drain are included. GPU preflight and
pre-timing exclusivity passed with the documented 64 MiB driver allowance.

```text
+---------+-----------+----------------------+------------+------------+
| Threads | Median s  | Min..max s           | Forward s  | Back/sketch|
+---------+-----------+----------------------+------------+------------+
|      16 |  3.996794 |  3.995769..4.004401 |   0.343179 |   3.279112 |
|      32 |  4.007734 |  4.005361..4.015998 |   0.343175 |   3.285817 |
|      64 |  3.995700 |  3.993620..3.998341 |   0.343473 |   3.277697 |
|      96 |  4.007389 |  4.006633..4.027146 |   0.346590 |   3.287897 |
+---------+-----------+----------------------+------------+------------+
```

The medians differ by about 0.3%; this sweep establishes no meaningful gain from
increasing the existing collector's thread count. Keep the 16-thread control
for sidecar comparisons and allocate extra CPU capacity to explicit independent
work rather than assuming a 96-thread setting creates overlap.

Raw artifacts: `/tmp/qvq-sidecar-baseline-cpu{16,32,64,96}.json` and matching logs.
The compact scorecard is
[`experiments/qvq_fisher_cpu_scaling_20260905.json`](experiments/qvq_fisher_cpu_scaling_20260905.json).
These are baseline timings, not an accepted sidecar optimization or a full-host
SMT scaling result. No collection math or production thread default changed.

```bash
# Repeat for 16, 32, 64, and 96 in separate processes; allow an idle cooldown.
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea \
python scripts/benchmark_qvq_yaqa_qwen38.py \
  --rows 64 --batch-size 16 --row-start 96 --seed 20260909 \
  --all-targets --arms streaming_256 --accumulator-device cuda \
  --no-activation-checkpointing --idle-max-driver-memory-mib 64 \
  --warmup 2 --repeats 3 --cpu-threads 16 --cpu-interop-threads 1 \
  --output /tmp/qvq-sidecar-baseline-cpu16.json
```

## Initial critical-path trace

A separate warmed trace used the same B16/64-sequence workload and 16 host
threads under the 96-CPU affinity. This is a **profiled diagnostic**, not a new
speedup measurement. The active capture took 4.070 seconds; GPU kernel interval
union was 3.776 seconds (92.8% of profiled wall). Including memcpy and memset,
GPU activity occupied 3.850 seconds. There were 89,152 kernel launches.

Dominant kernel families (top eight by total time, with all remaining work retained):

```text
+-----------------------------+-------+----------+---------+
| Family                      | Calls | GPU s    | Share % |
+-----------------------------+-------+----------+---------+
| FP32 cuBLAS TN 32x32        |  3008 | 0.770689 |   20.41 |
| Hopper FP32 projection      |  2368 | 0.402260 |   10.65 |
| FP32 CUTLASS NT 128x128     |  3008 | 0.388579 |   10.29 |
| Gaussian projection RNG     |  1600 | 0.252788 |    6.69 |
| FP32 sums, shared family    |  6592 | 0.191689 |    5.08 |
| BF16 model GEMM NNT         |  1476 | 0.182278 |    4.83 |
| FP32 cuBLAS NN 32x32        |   640 | 0.143280 |    3.79 |
| FP32 cuBLAS NT 32x32        |   384 | 0.134898 |    3.57 |
| All remaining kernels       | 70076 | 1.309686 |   34.68 |
+-----------------------------+-------+----------+---------+
```

`_project_kernel` maps directly to `qvq_yaqa_cuda.py`; Gaussian generation maps
to `projection.normal_` in `_streaming_projected_updates`. The prior executed
collector audit identifies the small FP32 TN GEMM family in token-Gram
contractions. FP32 GEMM and reduction family totals can span several call sites;
the trace alone does not assign every shared-family call to one expression.
Full kernel names, counts, and residual totals are retained in the scorecard.

| Overlap observation | Diagnostic time | Implication / dependency |
|---|---:|---|
| Gaps inside GPU activity span | 0.171 s | Only an upper bound on launch/synchronization opportunities; dependencies must be checked. |
| Factor D2H copies | 0.336 s | 2,400 copies, 7,016,939,520 bytes, copy stream 21. |
| Factor copies overlapping kernels | 0.276 s / 82.2% | Existing offload already overlaps substantially with backward. |
| Factor copy duration without compute overlap | 0.060 s | Test NUMA placement and handoff before assuming this is entirely removable. |

The CPU `cudaLaunchKernel` API total was 1.393 seconds across 68,148 calls. It
largely overlaps device execution and must not be added to GPU time or treated
as fully reclaimable latency.

| Candidate composition | Existing source boundary | Required gate |
|---|---|---|
| Copy completion -> grouped CPU diagonal normalization and descriptor construction | `_YaqaFactorTransfer` and final `YaqaGramSketch` construction in `qvq_yaqa.py` | Read only completed host buffers; preserve checks, divisor behavior, and exact returned factors. |
| Batch mask/count preparation -> repeated collector consumers | `active_mask` preparation and `accumulate_gradient` masking in `qvq_yaqa.py` | Preserve token order, masking, weights, and reduction semantics; measure saved GPU work. |
| Small GPU contraction -> CPU work overlapping another GPU operation | `_streaming_projected_updates` | Include all transfers/packing/drain; exact output and quantization gates precede promotion. |

These are proposed experiment boundaries, not implemented fusion or offload wins.
The trace prioritizes actual computation/copy crossover measurements over expecting
large gains from host bookkeeping alone. AMX BF16/INT8 application experiments
remain separate from the strict FP32 control.

Artifacts:

- `/tmp/qvq-sidecar-cpu16-trace.json` (295,385,174 bytes)
- `/tmp/qvq-sidecar-cpu16-trace-metadata.json`
- [Compact trace scorecard](experiments/qvq_fisher_cpu_trace_20260905.json)

Capture command: use the 16-thread baseline command above with `--repeats 1`,
`--trace-output /tmp/qvq-sidecar-cpu16-trace.json`, and a distinct output JSON.
CPU01 is complete for this bounded baseline/trace scope. CPU02 (NUMA handoff)
and CPU05 (full-path small-operation crossover) are the next prioritized trials;
physical-core placement and SMT comparisons remain queued within CPU09.

## oneDNN setup and dispatch probes

The 1.5x hybrid investigation uses the unchanged current GPU collector as its
control. Installing a CPU library does not itself change the collector.

PyTorch reports built-in oneDNN 3.12.0, commit
`80afa71049cd69a3df32adcccb623b12cd7baa22`, with OpenMP and AVX-512 support.
A CPU BF16 matrix multiply with `ONEDNN_VERBOSE=1` selected
`brg_matmul:avx10_1_512_amx` and returned the exact all-ones reference.

The requested Intel packages were installed in `/root/venv-py3.14t-gil0`:
`onednn==2026.0.2` and `onednn-devel==2026.0.2`, including their declared Intel
runtime dependencies. This distribution contains oneDNN 3.11.4 with a DPC++
runtime. Its standalone CPU engine failed because SYCL reported zero CPU devices
in this VM. Package installation is therefore not sufficient evidence of usable
CPU dispatch. No global library-path override was added.

The existing system oneDNN reports 3.9.1 at runtime. The standalone
[`probe_qvq_onednn.cpp`](../scripts/probe_qvq_onednn.cpp) verified all 65,536
outputs for each of BF16, INT8, and FP32 matmul. BF16 and INT8 selected
`brg_matmul:avx10_1_512_amx`; FP32 selected `brg_matmul:avx512_core`.
These all-ones checks establish execution and elementary algebra only, not
performance, collector rounding parity, or model quality.

A native OpenMP build from upstream tag `v3.12` (the same commit as PyTorch's
embedded version) was built and installed separately at `/opt/qvq/onednn-3.12.0`.
The build enables all primitive and CPU ISA families, GPU runtime `NONE`, and
uses eight compile jobs. The upstream option for disabling graph support is
`ONEDNN_BUILD_GRAPH`; `DNNL_BUILD_GRAPH` is ignored, so this build also includes
the default graph component. Build log: `/tmp/qvq-onednn-build.log`.

Native build reproduction (graph enabled explicitly):

```bash
git clone --depth 1 --branch v3.12 https://github.com/uxlfoundation/oneDNN.git /tmp/qvq-onednn-src
cmake -S /tmp/qvq-onednn-src -B /tmp/qvq-onednn-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/qvq/onednn-3.12.0 \
  -DDNNL_CPU_RUNTIME=OMP -DDNNL_GPU_RUNTIME=NONE \
  -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_GRAPH=ON
cmake --build /tmp/qvq-onednn-build -j8
cmake --install /tmp/qvq-onednn-build
g++ -O2 -Wall -Wextra -I/opt/qvq/onednn-3.12.0/include \
  scripts/probe_qvq_onednn.cpp -L/opt/qvq/onednn-3.12.0/lib \
  -Wl,-rpath,/opt/qvq/onednn-3.12.0/lib -ldnnl -o /tmp/qvq-onednn-native-probe
ONEDNN_VERBOSE=1 OMP_NUM_THREADS=16 /tmp/qvq-onednn-native-probe
```

The installed native build passed the same standalone probe: AMX BF16 and INT8,
and AVX-512 FP32, with all outputs verified. Runtime reports OpenMP, 16 threads,
and no GPU runtime. Full verbose evidence: `/tmp/qvq-onednn-native-probe.log`.
The installed library path is selected by an executable-local rpath; PyTorch's
embedded oneDNN and the system library remain independently usable.

### Initial real-input oneDNN crossover

The [operator report](experiments/qvq_fisher_onednn_crossover_20260905.json)
uses captured real Qwen3.5-27B activations and gradients, B4/B16, T64,
K5120/17408, 16 CPU threads, and FP32 output. All captured inputs round-trip
through BF16 exactly. GPU reference operations remain strict FP32.

Direct synchronous offload did not win. For B16/T64/K17408 gradient Gram,
GPU time was 0.542 ms, AMX CPU compute alone 0.175 ms, but full CPU roundtrip
was 2.105 ms (0.257x GPU speed). FP32 AVX-512 roundtrip was 5.328 ms.
Transfer and synchronization dominate; the promising compute-only ratio must
not be presented as a collector gain. All eight tested CPU results differed
bitwise from the GPU reference, including FP32 oneDNN. None is accepted as a
quantization-preserving replacement. Next investigate advance staging/reuse and
bounded overlap rather than synchronous per-hook offload.

The capture run used zero warmups and serialized tensors while collecting; its
latency is diagnostic only. Capture metadata and data are at
`/tmp/qvq-sidecar-capture-metadata.json` and `/tmp/qvq-sidecar-real-tensors/`.

## NUMA bandwidth and strict CPU pipeline experiments

Explicit memory binding confirmed a substantial transfer difference. Each run
used a fresh process, the H200 UUID, five warmups, twenty repetitions, and a
128 MiB pinned allocation. `/proc/self/numa_maps` confirmed that all 32,768 pages
resided on the requested node.

| Memory and CPU node | D2H GB/s | H2D GB/s |
|---|---:|---:|
| 0 | 39.20 | 45.80 |
| 1 | 37.50 | 43.97 |
| 2 | 21.24 | 40.24 |
| 3 | 21.54 | 41.04 |

These are transfer rates, not collector speedups. The reusable
[`benchmark_qvq_cpu_gpu_transfer.py`](../scripts/benchmark_qvq_cpu_gpu_transfer.py)
was also run under node-0 binding and reproduced 39.24 GB/s D2H and 46.21 GB/s
H2D. It records actual page placement, CPU inventory, preflight identity, and
all timing samples. Example, with `CUDA_VISIBLE_DEVICES` set to the physical UUID:

```bash
numactl --cpunodebind=0 --membind=0 python scripts/benchmark_qvq_cpu_gpu_transfer.py \
  --label node0 --output /tmp/qvq-transfer-node0.json
```

[Full NUMA measurements](experiments/qvq_fisher_numa_transfer_20260905.json).

A separate AVX-512 CPU prototype preserves increasing-K FP32 FMA accumulation,
computes triangular token-Gram tiles, and mirrors the upper triangle. For the
captured real BF16-valued inputs, it matched the GPU Grams bitwise. Cache blocking
and oneDNN's exact BF16-to-FP32 layout conversion reduced the B16/T64/K17408
operator from 2.13 ms to 1.28 ms on 16 local CPU threads. This is still slower
than GPU compute alone and requires overlap to be useful.

The bounded full-collector prototype retains GPU projections and diagonal
contractions, copies BF16-valued activations/gradients to CPU workers, and returns
strict FP32 CPU Grams. The main thread drains ready jobs and preserves per-batch
accumulation order. It passed the existing full-model comparison for **all 800
factors**, including source, diagonal, source diagonal, normalizer, and seed,
on B16/16 rows, row start 96, seed 20260909. This is one tested case, not general
acceptance or a performance win. The production collector remains unchanged.

Matched warmed one-batch measurements (two warmups, three repeats, CPU/memory
binding to socket-0 CPUs/node-0 memory, passive OpenMP waits):

| Arm | Median collection seconds | GPU-control speedup |
|---|---:|---:|
| Current GPU control | 1.057914 | 1.000x |
| CPU pipeline, two workers × eight threads | 4.315800 | 0.245x |
| CPU pipeline, one worker × sixteen threads | 4.699468 | 0.225x |
| Two workers with reusable scratch/primitives | 3.295074 | 0.321x |

All transfer and final-drain costs are included. Scratch reuse improved the
prototype, but it remains rejected for performance. Instrumentation attributes
most remaining worker time to packing and strict computation; standalone
concurrent probes also show large placement-dependent disparities. Isolate
native thread-team placement and memory locality before another full-model run.
The successful factor comparison applies to the original prototype; the cached
variant must repeat that gate before any promotion.

[Strict pipeline evidence](experiments/qvq_fisher_strict_cpu_pipeline_20260905.json)
records the source hashes, operator probes, fixed reference, and raw report paths.
Experimental collector/native sources remain under `/tmp/qvq-sidecar-*` and
`/tmp/qvq-cpu-strict-gram-*`; none is enabled in the production path.
