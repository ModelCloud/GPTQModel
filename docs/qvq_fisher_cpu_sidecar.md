# Dual-Xeon CPU sidecars for Sketch-B / Fisher collection

Status: **queued investigation**. The hardware/AMX availability probe is complete;
performance experiments have not launched. The tracked queue is
[`experiments/qvq_fisher_cpu_sidecar_queue_20260905.json`](experiments/qvq_fisher_cpu_sidecar_queue_20260905.json).

The objective is additional end-to-end acceleration by overlapping suitable CPU
work with GPU forward/backward and Fisher accumulation. Compare every candidate
against code commit `1acd13b95bcb2c4bfc3556d7e710da1385f911f5`, which already
contains overlapped factor offload and Hopper FP32 projection tiles. Comparing
against the older pre-optimization collector would double-count existing gains.
The preceding measurements are in [the collection report](qvq_yaqa_collection_overlap.md).

## Hardware and allocation verified on 2026-09-05

| Item | Observation |
|---|---|
| CPU | Two Intel Xeon Platinum 8575C sockets |
| Advertised topology | 48 cores/socket, 96 physical cores, 192 logical CPUs |
| Current allocation | **16 logical CPUs**, constrained by effective cpuset and process affinity |
| NUMA | Four advertised nodes; all four memory nodes are available to this cpuset |
| ISA flags | AVX-512, AVX-512 BF16, AVX-512 VNNI, AMX tile/BF16/INT8 |
| AMX execution | Native BF16 and INT8 tile operations executed and returned the expected results |
| GPU locality hint | H200 at PCI `0000:1c:00.0`; sysfs associates it with node 0 |
| Local allocated CPUs | Only CPU 17 intersects the GPU-local CPU list in this allocation |

`lscpu` describes the larger host topology; it does not grant access to those
cores. The current affinity/cpuset is:

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

The queue is a persistent research backlog, not a running background job. The
next executable step is cpu01 under the current allocation; cpu09 has a distinct
full-core resource prerequisite.
