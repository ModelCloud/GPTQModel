# Amplin Ampere GPTQ inference design and prototype journal

Last updated: 2026-07-24 UTC

## Purpose and journal rules

Amplin is an Ampere-first GPTQ inference kernel family. It starts from the serialized GPTQ contract, but it does not
assume that either the historical `int32` packing or Marlin's runtime repack and schedule are optimal for `sm_80`.
The initial goal is narrower: prove or reject a direct, group-aware W4A16 decode schedule on the requested Ampere
device before changing backend selection or checkpoint formats.

This file is the design source of truth and append-only experiment journal.

- Every implemented prototype, correctness run, benchmark, and profiler pass gets a dated entry.
- Successful and failed experiments are committed separately and pushed. Failed code is reverted; its hypothesis,
  exact command, result, and conclusion remain here.
- Measurements state the source revision, hardware UUID, shape, dtype, quantization contract, warmup, sample count,
  and timing method. Hypotheses are labeled as hypotheses.
- Correctness tests live under `tests/`; timed experiments live under `scripts/`.
- Architecture-specific launch paths use runtime properties, never a fixed CUDA index, product name, PCI address,
  or observed SM count.
- Backend integration and automatic selection remain out of scope until raw-operator correctness and a reproducible
  latency win are established.

## Current status

Prototype order and the V0 capability contract are frozen. The first raw JIT operator is registered and passes
canonical-layout, FP16/BF16 numerical, repeated-call, current-stream, and negative-boundary checks on the requested
`sm_80` device. It is not connected to a QuantLinear backend or automatic selection. The matched benchmark harness
passes its short smoke run and two gate-quality runs. The raw schedule reproducibly clears the 5% gate in FP16 and
BF16 and is retained for profiling. Backend integration remains gated on profiler evidence and broader shapes.

Starting revision: `00f3d5ef` on branch `future-1`.

## Current execution target

Effective 2026-07-24, all new Amplin GPU tests, builds, benchmarks, and profiler runs use only physical GPUs 4-6
in the live PCI-bus-ordered inventory. Each run resolves the selected allowed inventory entry to its UUID before
launch and exposes only the required allowed device or devices. Earlier GPU 1 and GPU 2 records below remain
historical experiment provenance, and GPU 3 is excluded from new work.

## Requested target and reproducibility snapshot

The user requested physical GPU 1 for the prototype. Commands resolve and record its UUID before running work; code
will still dispatch from the tensors' runtime device.

| Property | Value |
| --- | --- |
| Inventory index at capture | 1 |
| UUID | `GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855` |
| PCI bus at capture | `00000000:2B:00.0` |
| Device | NVIDIA PG506-232 |
| Compute capability | 8.0 |
| SMs | 124 |
| Memory | 98,304 MiB (`102191202304` bytes reported by PyTorch) |
| Warp size | 32 |
| Max threads per block | 1,024 |
| Shared memory per block, default | 49,152 bytes |
| Shared memory per SM | 167,936 bytes |
| Registers per SM | 65,536 |
| FP16 | Supported |
| BF16 | Supported |
| Driver | 610.43.02 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| NVCC | 13.0.88 |
| Python | 3.14.5 |
| Host C++ compiler | GCC 13.4.0 |

Isolated prototype builds use `TORCH_CUDA_ARCH_LIST=8.0`. Distributable build settings will retain the repository's
broader architecture policy.

## Existing format and baseline

### Canonical GPTQ W4 group-128 layout

For natural K order (`desc_act=False`) and `torch.int32` pack words:

```text
qweight: [K / 8, N]        # eight unsigned 4-bit codes per int32 along K
qzeros:  [K / 128, N / 8]  # eight zero-points per int32 along N
scales:  [K / 128, N]
```

The first symmetric prototype uses logical zero 8 and therefore does not read `qzeros`. The packed `int32` is only a
storage word; it is not the arithmetic dtype.

### Marlin runtime layout

Marlin repacks W4 weights to `[K / 16, 2N]`, preserving the byte count while applying its 16x64 Tensor Core
interleave. This is a runtime-derived layout, not the serialized GPTQ contract.

### Measured Marlin decode baseline

The source audit preceding Amplin measured Marlin at:

```text
M=1, K=4096, N=4096, GPTQ W4, group_size=128, sym=True,
desc_act=False, activation/scales=FP16
kernel duration:             16.704 us
compute throughput SOL:      15.27%
memory throughput SOL:       21.30%
DRAM bytes:                  8.69 MB
minimum weight+scale bytes:  about 8.659 MB
dynamic shared memory:       166.912 KiB
registers per thread:        94
achieved occupancy:          6.14%
no-eligible-warp issue:      76.14%
launch:                      grid 124, block 128
```

The near-minimum DRAM traffic means that a new layout cannot win merely by reading fewer packed weight bytes.
Amplin must reduce scheduling, unpack/dequantization, reduction, or resource costs, or expose more useful memory-level
parallelism.

## V0 capability contract

| Property | V0 contract |
| --- | --- |
| Method/format | GPTQ canonical packed weights |
| Bits | Exactly 4 |
| Group size | Exactly 128 |
| Symmetry | Exactly `sym=True`, logical zero 8 |
| Activation order | Exactly `desc_act=False` |
| Pack word | Contiguous `torch.int32` |
| Activations | Contiguous 2-D `[1, K]`, FP16 or BF16 |
| Scales | Contiguous `[K / 128, N]`, same dtype and device as activation |
| Accumulation | FP32, with scale applied once to each completed group partial |
| Output | `[1, N]`, same dtype and device as activation |
| Initial shape | Exactly `M=1`, `K=4096`, and `N` divisible by 16 |
| Device | Runtime-gated to CUDA compute capability 8.0 |
| Bias/adapters | Unsupported in V0 |
| Training/autograd | Unsupported in V0 |
| Fallback | No production routing change; existing backends remain authoritative |

Unsupported inputs will be rejected by the raw prototype with actionable messages. A future backend wrapper must
fall back without changing quantization semantics.

## First kernel hypothesis: group-local half-warp decode

The first kernel consumes canonical GPTQ `qweight` directly, with no runtime repack, workspace clear, global
reduction, atomics, or locks.

For `K=4096`, there are 32 group-128 K partitions. One 512-thread CTA owns 16 adjacent output columns:

1. Each of 16 warps contains two half-warps.
2. Each half-warp owns one group-128 partition and one lane owns one output column.
3. Each lane loads 16 canonical `int32` words, decodes 128 W4 codes, and accumulates `(code - 8) * x` in FP32.
4. The scale is loaded once after the unscaled group dot product.
5. The two half-warp partials combine with a warp shuffle.
6. A roughly 1.125 KiB shared array holds the 16 paired warp partials for a CTA-local reduction.
7. Warp 0 writes the final 16 outputs directly.

This is deliberately a SIMT decode design. W4A16 does not map directly to Ampere integer Tensor Cores without also
quantizing activations, and padding `M=1` into a Tensor Core M tile can spend work that a bandwidth-oriented GEMV
does not need. Tensor Core paths remain open for larger M.

### Candidate layouts to compare after the canonical kernel

| Layout | Role | Decision question |
| --- | --- | --- |
| Canonical GPTQ `[K/8, N]` | First implementation and no-repack reference | Is direct checkpoint consumption already faster? |
| Marlin `[K/16, 2N]` interleave | Production comparison | Does its Tensor Core layout help or hinder a decode schedule? |
| Amplin group tile `[N/16, K/128, 16, 16 words]` | Proposed derived layout | Does explicit group/column tiling improve vector loads or reduce address work enough to repay repacking? |

Layout changes will be benchmarked as derived runtime layouts first. Amplin will not introduce a new serialized
format until pack/unpack bit equality, save/reload behavior, conversion cost, and external compatibility are proven.

## Prototype order and gates

1. **Freeze contract and baselines.** Record hardware, software, GPTQ/Marlin layouts, first schedule, correctness
   reference, and performance gates.
2. **Register a raw JIT operator.** Add only the exact `sm_80`, W4, group-128, M=1/K=4096 path. Keep wrappers thin
   and use the tensor device guard and current CUDA stream.
3. **Prove canonical-layout correctness.** Generate bounded logical codes, pack through the real GPTQ-compatible
   layout, independently dequantize, and compare FP32-reference and dense low-precision outputs for FP16 and BF16.
   Test shape/dtype/device errors, repeated calls, and a non-default stream.
4. **Measure the minimal kernel.** Use preallocated tensors, at least 50 warmups, per-iteration CUDA events, and
   median/mean/std/p95/min/max. Compare the same codes, scales, activation, dtype, and shape with Marlin.
5. **Tune only from evidence.** Sweep CTA width, output columns per CTA, accumulation chains, vectorized activation
   loads, reduction shape, and launch size. Retain only reproducible wins.
6. **Run Nsight after timing identifies the best raw variant.** Compare DRAM bytes, achieved bandwidth, registers,
   occupancy, eligible warps, instruction mix, and stalls with the measured Marlin baseline.
7. **Compare layouts.** Measure canonical, Marlin, and Amplin group-tile layouts including one-time repack cost and
   steady-state inference latency.
8. **Generalize shape routes.** Add representative M=1 projection shapes, then design a separate small-M
   group-factored HMMA path. Do not force one schedule across decode and prefill.
9. **Integrate as an explicit-only backend.** Only after raw correctness and performance gates pass, add truthful
   capability declarations, pack/post-init lifecycle, save/reload tests, and fallback tests.
10. **Consider automatic selection and fusion.** Raise priority or fuse QKV/gate-up only with end-to-end model
    evidence and preserved unsupported-case fallbacks.

### Initial correctness gate

- Exact output shape, dtype, device, and finite values.
- Independent eager dequantization agrees with the packed codes exactly.
- Raw output compared in FP32 with a stated observed max absolute and relative error.
- FP16 and BF16 tested separately on the requested physical GPU.
- Malformed or unsupported inputs fail before launch.
- No errors under repeated calls and a non-default stream.

### Initial performance gate

For matched `M=1, K=N=4096, W4, group_size=128, sym=True, desc_act=False`:

- Amplin raw-kernel p50 must beat Marlin raw-kernel p50 by at least 5% in both FP16 and BF16 across repeated runs.
- No hidden repack, allocation, memset, workspace reduction, or synchronization may be excluded from Amplin's
  production-call comparison.
- A result below the gate remains a useful prototype result but does not justify backend integration.

## Experiment journal

### 2026-07-24 — Success: V0 contract and prototype order frozen

Revision before change: `00f3d5ef`

Actions:

- Re-probed the complete GPU inventory and resolved requested physical GPU 1 to UUID
  `GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`.
- Recorded PyTorch/CUDA/compiler versions, compute capability, SM count, memory, FP16/BF16 availability, and resource
  limits.
- Inspected canonical GPTQ packing, Marlin's runtime repack, GrassHopper's direct canonical-layout kernel, TriLin's
  CTA-local M=1 reduction, the shared JIT extension registry, and existing correctness/benchmark patterns.
- Froze the exact raw-operator contract, first SIMT schedule, correctness gate, performance gate, and experiment
  order above.

Commands:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,compute_cap,driver_version --format=csv,noheader
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_properties(1))'
nvcc --version
git status -sb
gh pr status
```

Result: success. The design is narrow enough to implement without changing serialization, backend selection, CPU
fallbacks, or non-target GPU behavior.

Next action: register the minimal raw Amplin JIT operator and run its boundary/correctness tests on the UUID-pinned
requested device.

### 2026-07-24 — Success: canonical-layout W4A16 raw operator

Revision before change: `0a427a23`

Implementation:

- Added a shared-JIT `gptqmodel_amplin::gemv` operator with an exact V0 boundary.
- Added the canonical `[K/8, N]` group-local half-warp kernel described above.
- Launch geometry is `N/16` CTAs with 512 threads per CTA and 1,152 bytes of static shared partial storage.
- The operator derives its CUDA device and current stream from the input tensor.
- C++ validation rejects non-CUDA, mixed-device, malformed, non-contiguous, non-int32-packed, dtype-mismatched, and
  non-`sm_80` inputs before launch.
- Added registry tests plus an actual CUDA test generated through `TorchLinear.pack_original`.
- Left checkpoint serialization, QuantLinear classes, backend selection, and all fallbacks unchanged.

Build command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
GPTQMODEL_EXT_VERBOSE=1 \
pytest -q -s tests/kernels/test_amplin.py
```

Observed build:

```text
JIT fingerprint directory: a88bd7d802a383ce
JIT build time:            31 s
CUDA architecture flag:   -gencode=arch=compute_80,code=sm_80
CUDA optimization flags:  -O3 --optimize=3 -Xptxas -O3,-dlcm=ca
BF16 flag:                 -DENABLE_BF16
line info:                 -lineinfo
```

Correctness result:

```text
tests/kernels/test_amplin.py: 6 passed
registry + pack-only checks:  15 passed

shape:                       M=1, K=4096, N=64
format:                      GPTQ canonical int32 W4
group_size/sym/desc_act:     128 / True / False
reference:                   independent unpack + FP32 dequantized matmul

dtype   max_abs_error   mean_abs_error   max_relative_error   max_abs_output
FP16    0.000118017     0.000049733      0.000418550          0.954589844
BF16    0.000916600     0.000254104      0.002763849          0.953125000
```

The test tolerance is `2e-4` absolute for FP16 and `2e-3` for BF16, based on the observed deterministic case rather
than a widened generic tolerance. The same outputs pass on a non-default stream and are bitwise stable across repeated
calls. Unsupported boundary cases fail with the expected error before a kernel launch.

Result: success. The checkpoint-native dataflow is correct for the exact V0 contract.

Next action: add a CUDA-event benchmark that builds matched canonical codes/scales/activations for Amplin and Marlin,
then measure raw operator and production-call costs on the same requested GPU.

### 2026-07-24 — Failure: benchmark smoke command omitted repository import path

Revision: `a701e97b`

Attempted command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_vs_marlin.py \
  --dtype both --warmup 5 --iters 10 --rounds 1 \
  --json-out /tmp/amplin-smoke.json
```

Result:

```text
ModuleNotFoundError: No module named 'gptqmodel'
```

Classification: harness invocation failure. The active interpreter does not have this checkout installed, and direct
execution places `scripts/` rather than the repository root on `sys.path`. No extension loaded, no CUDA kernel
launched, and no timing was collected.

Resolution: rerun with `PYTHONPATH=/root/GPT-QModel-Ultra-2`, matching the repository's existing kernel benchmark
environment. This failure does not change the kernel hypothesis.

### 2026-07-24 — Success: matched benchmark harness smoke validation

Revision before change: `1f0d5bbf`

Implementation:

- Added `scripts/benchmark_amplin_vs_marlin.py`.
- Constructs one legal canonical W4 tensor, scale tensor, and activation shared by both kernels.
- Builds Marlin from the same canonical buffers, then allows its existing `post_init()` to repack and permute them.
- Compares both outputs with an independent unpacked FP32 dequantized matmul before timing.
- Measures pre-resolved raw operators and their normal Python/module call paths separately.
- Records per-call CUDA-event distributions, one batched CUDA-event mean per round, and batched synchronized wall
  time. The batched event avoids placing two timestamp events around every tiny kernel and is the primary smoke
  comparison.
- Emits the complete hardware inventory, visible-device UUID, software revision, layout bytes, errors, timings, and
  speedups to both an ASCII table and optional JSON.

Validated command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_vs_marlin.py \
  --dtype both --warmup 5 --iters 20 --rounds 2 \
  --json-out /tmp/amplin-smoke-v2.json
```

Smoke correctness:

```text
dtype  Amplin max/mean abs vs FP32    Marlin max/mean abs vs FP32    max cross error
FP16   0.000471950 / 0.000050950      0.000471950 / 0.000076424      0.000488281
BF16   0.003831029 / 0.000390688      0.003831029 / 0.000599994      0.003906250
```

Smoke timing:

```text
path             dtype  samples  call p50 us  batch event mean us  wall mean us
Amplin raw       FP16        40       35.840                18.022        18.037
Marlin raw       FP16        40       54.272                24.294        26.608
Amplin wrapper   FP16        40       69.632                49.613        47.462
Marlin module    FP16        40       98.304                73.139        68.731
Amplin raw       BF16        40       29.696                17.997        18.282
Marlin raw       BF16        40       43.008                19.456        21.146
Amplin wrapper   BF16        40       66.560                44.314        36.215
Marlin module    BF16        40       79.872                54.938        56.830
```

Canonical and Marlin runtime layouts both contain 8,388,608 weight bytes and 262,144 scale bytes. The matched logical
read payload including activation and output is 8,667,136 bytes.

Result: benchmark harness success, but not a performance-gate result. Forty samples, short warmup, and clock/outlier
effects are insufficient for a retention decision. The smoke suggests the first raw schedule is worth a full run.
It also shows that repeatedly resolving the extension op in `amplin.gemv` is material host overhead; production
integration will need a cached handle or a QuantLinear-owned operator after the raw kernel gate.

The initial corrected smoke populated the Marlin JIT cache in 143 seconds for FP16 and 85 seconds for BF16. Build time
was excluded from all timing.

Next action: run 100 warmups and five rounds of 500 samples per path for both dtypes, then decide the V0 schedule
against the 5% raw-path gate.

### 2026-07-24 — Success: first gate-quality raw benchmark

Revision: `965e547f`

Pre-run state:

```text
UUID:          GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855
utilization:   0%
memory used:   0 MiB
temperature:   34 C
SM clock:      210 MHz before warmup
memory clock:  1593 MHz
power state:   P0
```

Command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_vs_marlin.py \
  --dtype both --warmup 100 --iters 500 --rounds 5 \
  --json-out /tmp/amplin-v0-baseline-gpu1.json
```

Matched correctness was identical to the smoke run:

```text
dtype  Amplin max/mean abs vs FP32    Marlin max/mean abs vs FP32    max cross error
FP16   0.000471950 / 0.000050950      0.000471950 / 0.000076424      0.000488281
BF16   0.003831029 / 0.000390688      0.003831029 / 0.000599994      0.003906250
```

Raw timing:

```text
path         dtype  samples  call p50 us  call p95 us  batch event mean us  wall mean us  logical GB/s
Amplin raw   FP16      2500       28.672       33.792                14.908        14.920        581.37
Marlin raw   FP16      2500       40.960       48.128                17.841        19.528        485.81
Amplin raw   BF16      2500       28.672       32.768                15.808        15.249        548.27
Marlin raw   BF16      2500       41.984       50.176                18.774        18.278        461.67
```

Normal call-path timing:

```text
path             dtype  samples  call p50 us  batch event mean us  wall mean us
Amplin wrapper   FP16      2500       54.272                32.852        33.669
Marlin module    FP16      2500       77.824                55.492        54.996
Amplin wrapper   BF16      2500       55.296                33.545        37.488
Marlin module    BF16      2500       77.824                55.341        56.906
```

Primary raw batched-event result:

```text
FP16 speedup: 17.841 / 14.908 = 1.197x
BF16 speedup: 18.774 / 15.808 = 1.188x
```

Result: success. The first canonical-layout schedule clears the required 5% gate by 19.7% FP16 and 18.8% BF16 in
the primary batched CUDA-event measurement. Synchronized batched wall timing independently agrees. Per-call event
means contain rare millisecond-scale host/system outliers, so the journal reports p50/p95 and batched event time
rather than using the untrimmed mean as the decision statistic.

Interpretation: because canonical and Marlin weight/scale byte counts are identical, the improvement supports the
schedule/resource hypothesis rather than a payload-size explanation. The wrapper measurements are diagnostic only:
`amplin.gemv` currently resolves and locks the extension on every call and must not be the eventual production
interface.

Decision: provisionally retain the V0 kernel. Do not add a backend yet. Repeat the full matched run, then profile the
pre-resolved raw kernels to separate kernel duration from event/call instrumentation and inspect resource use.

### 2026-07-24 — Success: full benchmark repeat retains V0 schedule

Revision: `dcf0b2ca`

The target was again idle immediately before the run: 0% utilization, 0 MiB used, 34 C, 210 MHz idle SM clock,
1593 MHz memory clock, and P0. The command and workload were identical to the first gate-quality run except for the
output path:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_vs_marlin.py \
  --dtype both --warmup 100 --iters 500 --rounds 5 \
  --json-out /tmp/amplin-v0-repeat-gpu1.json
```

Repeat raw timing:

```text
path         dtype  samples  call p50 us  call p95 us  batch event mean us  wall mean us  logical GB/s
Amplin raw   FP16      2500       28.672       32.768                15.904        15.899        544.97
Marlin raw   FP16      2500       39.936       49.152                17.326        17.128        500.25
Amplin raw   BF16      2500       29.696       33.792                15.864        18.529        546.35
Marlin raw   BF16      2500       41.984       48.128                18.305        18.512        473.48
```

Repeat batched-event speedup:

```text
FP16: 17.326 / 15.904 = 1.089x
BF16: 18.305 / 15.864 = 1.154x
```

Cross-run summary:

```text
dtype  Amplin batch mean us  Marlin batch mean us  combined speedup  Amplin p50 range  Marlin p50 range
FP16                 15.406                 17.584             1.141x       28.672-28.672    39.936-40.960
BF16                 15.836                 18.540             1.171x       28.672-29.696    41.984-41.984
```

The independent correctness values are bit-for-bit identical to the first run because the deterministic input,
canonical words, and scales are unchanged. The repeat confirms the raw event gate in both dtypes. BF16 synchronized
wall time was effectively tied in this one run (`18.529` versus `18.512` us), illustrating why raw CUDA-event and
repeated p50 evidence, rather than a single host-timed sample, controls the kernel decision.

Result: success. Retain the exact V0 raw kernel for profiler analysis. It clears the 5% raw p50 gate by at least 28%
in both complete runs and the lower-overhead batched-event gate by 8.9-19.7% FP16 and 15.4-18.8% BF16.

Next action: capture pre-resolved Amplin and Marlin kernels with Nsight Systems/Compute, record exact kernel
durations, registers, occupancy, DRAM traffic, eligible-warps/stalls, and instruction mix, then choose the first
evidence-based tuning variable.

### 2026-07-24 — Success: bounded raw-kernel profiler harness

Revision before change: `761797c3`

Implementation:

- Added `scripts/profile_amplin_vs_marlin.py`.
- Keeps JIT loading, Marlin repacking, independent FP32 dequantization, correctness checks, allocator warmup, and
  device warmup outside the measured capture.
- Uses pre-resolved raw Amplin and Marlin operators.
- Emits one named NVTX range per selected kernel path and optionally brackets all selected ranges with
  `cudaProfilerStart()` and `cudaProfilerStop()`.
- Accepts one dtype and an explicit `amplin`, `marlin`, or `both` path so Nsight Compute can replay a bounded target
  without profiling setup work.
- Repeats the runtime `sm_80` and BF16 capability checks and records the selected device, UUID visibility, complete
  GPU inventory, software versions, Git revision, quantization contract, shape, launch count, and dense-reference
  error.

Validation:

```bash
ruff check scripts/profile_amplin_vs_marlin.py

PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/profile_amplin_vs_marlin.py \
  --path both --dtype fp16 --warmup 2 --launches 3

PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/profile_amplin_vs_marlin.py \
  --path amplin --dtype fp16 --warmup 1 --launches 1 \
  --cuda-profiler-api
```

Result:

```text
ruff:                    passed
device:                  NVIDIA PG506-232, sm_80, 124 SM
both-path smoke:         passed, 3 raw launches per NVTX range
CUDA profiler API smoke: passed, 1 bounded Amplin launch
Amplin max/mean FP16:    0.000471950 / 0.000050950
Marlin max/mean FP16:    0.000471950 / 0.000076424
```

The profiler harness is successful and ready for Nsight Systems capture. Before interpreting a square-only profile,
the acceptance matrix will be expanded per the requested deployment targets: exact Qwen3-8B and Laguna S 2.1 linear
shapes discovered from the local model configs, plus decode and batched-token regimes. The existing fixed
`M=1, K=N=4096` kernel remains a V0 data point rather than the final Amplin shape contract.

### 2026-07-24 — Success: real-model shape and batching contract

Revision: `2dbcbfc0`

The target shapes were read from local configs and safetensors headers without materializing checkpoint tensors:

```text
Qwen3 dense:       /monster/data/model/Qwen3-8B
Qwen3 GPTQ W4:     /monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512
Laguna dense:      /monster/data/model/Laguna-S-2.1
Laguna GPTQ W4:    /monster/data/model/Laguna-S-2.1-GPTQModel-compat
```

Relevant source fingerprints:

```text
Qwen3 dense config:
  f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
Qwen3 GPTQ config:
  12c761af18450450ebbe7c87bcd4a0bd88cb3cffdd32bf9f433f24101b3d158f
Qwen3 GPTQ safetensors index:
  a537619f997cdde743534a36001ebbaa56af839def1718177106c63871fb9184
Laguna dense config:
  8309d2ab0da8ac0981b8803b1a4637d843c10fdf7851ddd202ca918fb682392c
Laguna GPTQ config:
  11fbb529c67628358e2ee737620fa75740f1b99e467af60aae48ca9fd4967180
Laguna GPTQ safetensors index:
  25652decee07fd11f7f679396a92afcca4e2bf96528e61125ad378fa4455c7ed
```

Both quantized checkpoints use GPTQ W4, group size 128, symmetric logical zero 8, `desc_act=False`, and int32 packing.
Their safetensors headers confirm `I32` qweights and `F16` scales.

Qwen3-8B has hidden size 4096, intermediate size 12288, 36 layers, 32 attention heads, and 8 KV heads. Its 252
quantized projections reduce to four exact `(K, N)` classes:

```text
K      N      modules                              count
4096   1024   k_proj, v_proj                       72
4096   4096   q_proj, o_proj                       72
4096   12288  gate_proj, up_proj                   72
12288  4096   down_proj                            36
```

Laguna S 2.1 has hidden size 3072, dense intermediate size 12288, MoE intermediate size 1024, 48 layers, 48 attention
heads, 8 KV heads, 256 experts, and 10 routed experts per token. The full dense checkpoint and the
GPTQModel-compatible checkpoint expose the following unique quantizable `(K, N)` classes:

```text
K      N      role
1024   3072   expert down_proj
3072   48     g_proj attention variant
3072   72     g_proj attention variant
3072   1024   k_proj, v_proj, expert gate_proj/up_proj
3072   6144   q_proj attention variant
3072   9216   q_proj attention variant
3072   12288  dense gate_proj/up_proj
6144   3072   o_proj attention variant
9216   3072   o_proj attention variant
12288  3072   dense down_proj
```

The Laguna compatibility checkpoint quantizes layers 0 and 1 and leaves layers 2-47 dense, but those first two layers
contain every unique quantized shape above, including all 256 experts plus the shared expert. The dense router
`(3072, 256)` is not a GPTQ qweight and is therefore outside the Amplin W4 kernel contract.

The widths 48 and 72 invalidate a blanket `N % 64 == 0` assumption. Amplin must have a correct masked tail. All
listed `K` values are divisible by group size 128, but they span 8 through 96 quantization groups, so the V0
one-half-warp-per-group schedule cannot remain fixed at 512 threads.

Expanded acceptance matrix:

```text
correctness:
  every unique Qwen3 and Laguna (K, N)
  M = 1, 4, 16
  FP16 and BF16
  independent FP32 dequantized reference

timing:
  every unique shape at M = 1, 4, 16
  representative dense-attention, dense-MLP, and MoE-expert shapes at M = 64, 256
  raw Amplin versus the best legal existing path, with layout/setup time reported separately

profiling:
  M = 1 single-request decode
  M = 16 continuous-batch decode
  M = 256 chunked-prefill/token-GEMM
```

`M` is the flattened activation-row/token count. For decode it represents concurrent sequences; for prefill it may
represent batch times token chunk. Laguna expert projections also need the small-`M` points because routed token
counts per expert remain sparse even when the model-wide batch is large.

Design consequence: Amplin is allowed to become a dispatched family. A canonical-layout bandwidth kernel may be
best for small `M`; a separate Ampere Tensor Core schedule and possibly a different offline weight layout may be
best for medium/large `M`. The V0 row-independent GEMV schedule may be used as a batched correctness baseline, but
re-reading the whole weight matrix for every row is not accepted as the final batched design.

Next action: generalize the raw operator to arbitrary listed `K`, masked `N` tails, and `M > 1`, using a bounded
group-stride schedule as the correctness baseline. Then run the complete shape matrix before making performance
claims.

### 2026-07-24 — Success: generalized real-shape and batched correctness baseline

Revision before change: `b465349b`

Implementation:

- Preserved the successful V0 512-thread kernel for `M=1`, `K=4096`, and `N % 16 == 0`.
- Added a general canonical-layout group-stride kernel for all positive group-128-aligned `K`, arbitrary positive
  `N`, and flattened activation-row count `M`.
- Uses at most eight warps per CTA. Warps stride over pairs of quantization groups, each half-warp still owns one
  group and 16 output columns, and the CTA reduces only eight warp partials.
- Uses 128 threads for the Laguna expert-down `K=1024` case and 256 threads for the other target K values.
- Maps one CTA to one activation row and one 16-column output tile. This is a correctness baseline for batching; it
  deliberately does not claim cross-row weight reuse.
- Masks the final output tile, including the real Laguna widths 48 and 72.
- Accepts contiguous activation tensors with two or more dimensions, flattens every prefix dimension into `M`, and
  restores the same prefix shape on output.
- Derives `K` from the input, validates canonical qweight `[K/8, N]` and scales `[K/128, N]`, checks grid and int32
  indexing limits, retains exact runtime `sm_80` gating, and dispatches on the current CUDA stream.
- Expanded `tests/kernels/test_amplin.py` with the 14 exact Qwen3-8B and Laguna S 2.1 shape classes. Every shape runs
  `M=1,4,16` in both FP16 and BF16 against an independent FP32 dequantized reference.

The new extension source fingerprint is `a355d56ff743a703`.

Validation command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
pytest -q tests/kernels/test_amplin.py -x
```

Result:

```text
21 passed in 9.86s

covered dtypes:       FP16, BF16
covered M:            1, 4, 16
covered unique K,N:   14
Qwen3 shape classes:  4
Laguna shape classes: 10
tail widths:          48, 72
additional checks:    real TorchLinear GPTQ pack, exact repeat, 3-D prefix,
                      non-default stream, invalid layout/dtype/device/empty input
FP16 error bound:     max abs <= 2e-3 for every matrix case
BF16 error bound:     max abs <= 2e-2 for every matrix case
```

Additional checks:

```text
ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py
  passed

pytest -q tests/test_extension_load_api.py
  14 passed in 8.90s

git diff --check
  passed
```

The first combined test invocation reached the JIT compilation message but its command-output wrapper returned
without a pytest summary. An isolated follow-up process for the first CUDA case exited zero, and the subsequent full
21-test process exited zero. This was treated as inconclusive tool output, not as a kernel failure.

V0 regression smoke:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_vs_marlin.py \
  --dtype fp16 --warmup 20 --iters 50 --rounds 2 \
  --json-out /tmp/amplin-generalized-v0-regression.json
```

```text
path         p50 us  batch event mean us  wall mean us
Amplin raw    29.696                17.152        17.237
Marlin raw    41.984                18.299        18.060
raw speedup: 1.067x by batch event; correctness values unchanged
```

This short run is a regression smoke, not a new performance gate. It confirms that the original square fast-path
dispatch still reaches the retained V0 kernel and remains numerically unchanged.

Decision: retain the generalized kernel as the real-shape/batching correctness baseline. Do not integrate a backend
or infer batched performance from it.

Next action: generalize the matched benchmark and profiler harnesses to `(M,K,N)`, run every real shape at
`M=1,4,16`, then profile representative `M=1,16,256` regimes. The resulting crossover points will decide where a
canonical small-M kernel ends and an Ampere Tensor Core/repacked schedule begins.

### 2026-07-24 — Success: real-model timing harness exposes baseline crossover

Revision before change: `1a7c9f3f`

Implementation:

- Generalized `scripts/benchmark_amplin_vs_marlin.py` helpers and CLI from fixed `M=1,K=4096` to explicit
  group-aligned `(M,K,N)`.
- Generalized `scripts/profile_amplin_vs_marlin.py` range names, construction, correctness, and metadata to explicit
  `(M,K,N)`.
- Added `scripts/benchmark_amplin_model_shapes.py` with the four exact Qwen3-8B and ten exact Laguna S 2.1 shape
  classes.
- Reuses one canonical qweight/scale tensor and one Marlin repack across all requested M values for a shape/dtype.
- Validates every timed path against an independently unpacked FP32 dequantized matmul before timing.
- Reports raw pre-resolved operators only, per-call CUDA event p50/p95, batched CUDA-event mean, synchronized wall
  mean, requested byte rate, max absolute error, and speedup versus raw Marlin.
- Distinguishes the current Amplin row-independent requested bytes from the minimum GPTQ payload Marlin may reuse.
- Optionally times a resident dense FP16/BF16 matmul as a clearly labeled compute ceiling; its weight storage is four
  times W4 and it is not a GPTQ inference path.
- Marks `N=48` and `N=72` as not legal for Marlin because Marlin repacking requires `N % 64 == 0`. The harness does
  not silently pad those real Laguna shapes or invent a Marlin comparison.

Static validation:

```text
ruff check scripts/benchmark_amplin_vs_marlin.py \
  scripts/profile_amplin_vs_marlin.py \
  scripts/benchmark_amplin_model_shapes.py
  passed

git diff --check
  passed
```

Representative smoke command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 --shape 1024x3072 --shape 3072x72 \
  --dtype both --m-values 1,4,16 \
  --warmup 5 --iters 10 --rounds 1 \
  --json-out /tmp/amplin-model-shapes-smoke.json
```

Raw batched-event smoke:

```text
model/role                dtype  M   Amplin us  Marlin us  speedup
Qwen3 4096->12288 MLP     FP16   1      34.304     29.082   0.848x
Qwen3 4096->12288 MLP     FP16   4     110.182     27.443   0.249x
Qwen3 4096->12288 MLP     FP16  16     404.890     29.798   0.074x
Qwen3 4096->12288 MLP     BF16   1      35.021     27.955   0.798x
Qwen3 4096->12288 MLP     BF16   4     112.845     28.467   0.252x
Qwen3 4096->12288 MLP     BF16  16     418.304     31.744   0.076x

Laguna 1024->3072 expert   FP16   1      10.752     26.112   2.429x
Laguna 1024->3072 expert   FP16   4      14.336     26.112   1.821x
Laguna 1024->3072 expert   FP16  16      32.358     26.214   0.810x
Laguna 1024->3072 expert   BF16   1      11.878     27.546   2.319x
Laguna 1024->3072 expert   BF16   4      14.131     30.106   2.130x
Laguna 1024->3072 expert   BF16  16      33.280     27.955   0.840x

Laguna 3072->72 tail       FP16   1      12.800        n/a      n/a
Laguna 3072->72 tail       FP16   4      13.107        n/a      n/a
Laguna 3072->72 tail       FP16  16      13.312        n/a      n/a
Laguna 3072->72 tail       BF16   1      12.288        n/a      n/a
Laguna 3072->72 tail       BF16   4      12.902        n/a      n/a
Laguna 3072->72 tail       BF16  16      13.107        n/a      n/a
```

All smoke correctness checks passed. The largest reported max absolute error was `0.0006812` FP16 and `0.0052134`
BF16 for Marlin; Amplin's largest was `0.0004879` FP16 and `0.0039055` BF16.

Dynamic profiler-range smokes also passed:

```text
both paths: M=16, K=3072, N=1024, FP16, 2 launches/range
tail path:  M=16, K=3072, N=72,   BF16, 2 launches/range
CUDA profiler API bracketing: passed for both processes
```

Classification: timing-harness success and expected performance failure of the row-independent schedule outside its
small-M niche. This is not a failed correctness prototype and is not reverted:

- The small Laguna expert weight wins strongly at M=1 and M=4, showing that Marlin launch/resource cost dominates
  for sparse per-expert token counts.
- The same expert crosses behind Marlin by M=16.
- Qwen's large up projection already needs a better small-M schedule at M=1, and independent row CTAs become
  categorically unsuitable at M=4 and M=16.
- The tiny N=72 weight remains almost flat from M=1 through M=16, consistent with cache reuse, but needs comparison
  with the actual legal non-Marlin fallback before a backend decision.

The ten-iteration smoke is not a performance gate. Its purpose is to verify the harness and make the dispatch
problem explicit.

Next action: run the full 14-shape, both-dtype, `M=1,4,16` matrix with gate-quality warmup and repeated rounds.
Summarize wins, losses, and crossover classes before selecting Nsight targets.

### 2026-07-24 — Success: first full real-model crossover matrix

Revision: `4b9bc5fa`

Pre-run target state:

```text
physical index: 1
UUID:           GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855
PCI:            00000000:2B:00.0
device:         NVIDIA PG506-232, sm_80, 124 SM
utilization:    0%
memory used:    0 MiB
temperature:    34 C
SM clock:       210 MHz idle
memory clock:   1593 MHz
power state:    P0
```

Command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all --dtype both --m-values 1,4,16 \
  --warmup 50 --iters 100 --rounds 3 \
  --json-out /tmp/amplin-model-shapes-full-gpu1.json
```

The JSON report is 138,115 bytes and contains 156 timing rows: 84 Amplin rows for every
`14 shapes x 3 M x 2 dtypes`, plus 72 Marlin rows for the 12 shapes legal under `N % 64 == 0`. Each path has 300
per-call event samples and three lower-overhead batched-event/wall rounds. The complete process took 23.64 seconds.

Legal-shape win counts using raw batched CUDA-event means:

```text
dtype  M   Amplin faster  clears 5% gate  legal shapes
FP16   1       6 / 12          6 / 12          12
FP16   4       3 / 12          3 / 12          12
FP16  16       0 / 12          0 / 12          12
BF16   1       9 / 12          9 / 12          12
BF16   4       3 / 12          3 / 12          12
BF16  16       0 / 12          0 / 12          12
```

Per-shape speedup matrix (`Marlin time / Amplin time`):

```text
abbrev  shape/role
Qkv     Qwen3 4096->1024 KV projection
Qqo     Qwen3 4096->4096 Q/O projection
Qup     Qwen3 4096->12288 MLP up
Qdown   Qwen3 12288->4096 MLP down
Lexp    Laguna 1024->3072 expert down
Lkv     Laguna 3072->1024 KV/expert up
Lq6     Laguna 3072->6144 Q projection
Lq9     Laguna 3072->9216 Q projection
Ldup    Laguna 3072->12288 dense up
Lo6     Laguna 6144->3072 O projection
Lo9     Laguna 9216->3072 O projection
Ldd     Laguna 12288->3072 dense down

dtype M   Qkv    Qqo    Qup    Qdown  Lexp   Lkv    Lq6    Lq9    Ldup   Lo6    Lo9    Ldd
FP16  1  2.690  0.681  0.860  0.616  2.796  2.510  1.363  1.173  0.974  1.567  0.917  0.684
FP16  4  1.420  0.706  0.266  0.416  1.500  1.891  0.643  0.451  0.328  0.574  0.514  0.375
FP16 16  0.655  0.197  0.068  0.077  0.875  0.771  0.196  0.116  0.123  0.158  0.134  0.090
BF16  1  2.979  1.667  0.826  0.562  2.931  2.043  1.260  1.107  1.051  1.457  1.069  0.776
BF16  4  1.475  0.618  0.242  0.228  2.490  1.688  0.521  0.385  0.299  0.531  0.364  0.510
BF16 16  0.614  0.186  0.069  0.070  0.927  0.736  0.215  0.136  0.085  0.167  0.177  0.101
```

Extremes:

```text
FP16 best:  2.796x, Laguna expert down, M=1, Amplin 8.277 us
FP16 worst: 0.068x, Qwen3 MLP up, M=16, Amplin 349.327 us
BF16 best:  2.979x, Qwen3 KV projection, M=1, Amplin 7.950 us
BF16 worst: 0.069x, Qwen3 MLP up, M=16, Amplin 360.564 us
```

Marlin-ineligible Laguna tails:

```text
dtype  M   3072->48 us  3072->72 us
FP16   1       9.653        11.745
FP16   4       9.875        12.066
FP16  16      10.103        10.237
BF16   1       9.312         9.438
BF16   4       9.776         9.820
BF16  16       9.834        10.615
```

Worst numerical errors across the complete run:

```text
path         dtype  max abs    location
Amplin raw   FP16   0.0009744  Qwen3 MLP down, M=16, K=12288, N=4096
Amplin raw   BF16   0.0077991  Laguna dense down, M=16, K=12288, N=3072
Marlin raw   FP16   0.0010998  Laguna dense down, M=16, K=12288, N=3072
Marlin raw   BF16   0.0090489  Laguna dense down, M=16, K=12288, N=3072
```

All correctness gates passed.

Interpretation:

- Amplin V0's square win does not generalize uniformly to non-square M=1. K, N, output-tile count, and dtype matter.
- Three compact-output/low-K classes survive through M=4 in both dtypes: Qwen KV, Laguna expert down, and Laguna
  KV/expert up.
- No row-independent schedule survives M=16. Cross-row weight reuse is a hard dispatch boundary, not a minor tuning
  opportunity.
- M=1 has real canonical-layout wins beyond the original square point, especially bandwidth-light outputs and
  sparse experts. A specialized small-M family remains justified.
- The stable 9-12 us N=48/72 tail measurements fill a real coverage hole, but require a legal Torch/fallback
  comparator before production routing.

Decision: retain the correctness baseline and the original square fast path. Do not route M>=16 to this baseline.
Repeat the complete matrix once from an idle device before finalizing profiler targets; the large FP16/BF16
differences on some M=1 shapes need a stability check.

### 2026-07-24 — Failure: three-round batch-event mean is not a stable routing statistic

Revision: `67a58692`

The complete matrix was repeated with the identical command and workload, writing
`/tmp/amplin-model-shapes-repeat-gpu1.json`. The target was again idle before launch: 0% utilization, 0 MiB used,
34 C, 210 MHz idle SM clock, 1593 MHz memory clock, and P0. The repeat process took 23.51 seconds and all numerical
checks passed.

Stable high-level findings:

```text
dtype  M   run-1 Amplin wins  repeat Amplin wins
FP16   4          3 / 12             3 / 12
FP16  16          0 / 12             0 / 12
BF16   4          3 / 12             3 / 12
BF16  16          0 / 12             0 / 12
```

The same three compact classes win at M=4, and no class wins at M=16. The extreme cases are also stable:

```text
combined FP16 best:  2.897x Laguna expert down M=1
combined BF16 best:  3.064x Laguna expert down M=1
combined FP16 worst: 0.071x Qwen3 MLP up M=16
combined BF16 worst: 0.070x Qwen3 MLP up M=16
```

However, six borderline cells changed winner:

```text
shape/dtype/M                         run 1   repeat
Qwen 4096->4096 FP16 M=1             0.681x   1.531x
Laguna 9216->3072 FP16 M=1           0.917x   1.125x
Laguna expert-down BF16 M=16         0.927x   1.020x
Laguna 3072->12288 BF16 M=1          1.051x   0.965x
Laguna 6144->3072 BF16 M=1           1.457x   0.606x
Laguna 9216->3072 BF16 M=1           1.069x   0.951x
```

Inspection shows that the stored `batch_event_mean_us` can be dominated by one perturbed round:

```text
shape/path                            run  p50 us  batch mean us  wall mean us
Qwen 4096->4096 FP16 Amplin           1    29.696        32.355        14.572
Qwen 4096->4096 FP16 Amplin           2    31.744        16.773        14.648
Laguna 6144->3072 BF16 Amplin         1    29.696        16.445        16.247
Laguna 6144->3072 BF16 Amplin         2    31.744        39.567        16.245
Laguna expert-down BF16 M=4 Marlin    1    50.176        27.382        22.544
Laguna expert-down BF16 M=4 Marlin    2    54.272        37.550        25.951
```

The current `TimingStats` serializes only the mean of three batch-event and wall rounds, so the individual samples
cannot be inspected after the process exits. Averaging the two run-level means reduces noise but does not repair the
missing per-round evidence.

Classification: benchmark aggregation failure, not a kernel correctness failure. The broad M=4/M=16 crossover is
valid, but borderline M=1 routing decisions are suspended.

Resolution:

1. Serialize every batched-event and wall-time round.
2. Add median, p95, min, and max fields while retaining mean for compatibility.
3. Make the batched-event median the primary speedup statistic.
4. Require at least five rounds for gate-quality model-shape runs.
5. Rerun the complete matrix and report both median and spread before choosing profiler targets.

### 2026-07-24 — Success: auditable five-round median timing

Revision before change: `947dac75`

Implementation:

- `TimingStats` now serializes every batched CUDA-event round and every synchronized wall-time round.
- Added median, mean, p95, min, and max for both batched-event and wall distributions.
- Retained the previous mean fields for report compatibility.
- Changed speedup and effective-byte-rate calculations to the batched-event median.
- Added batch median, mean, min-max range, wall median, and wall mean to ASCII reports.
- Changed the real-model matrix default from three to five rounds.
- Added `gate_quality_round_count` and `primary_statistic` metadata to JSON reports. Shorter explicit runs remain
  valid smoke tests but are marked non-gating.

Validation:

```bash
ruff check scripts/benchmark_amplin_vs_marlin.py \
  scripts/benchmark_amplin_model_shapes.py \
  scripts/profile_amplin_vs_marlin.py

PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model qwen3-8b --shape 4096x4096 \
  --dtype fp16 --m-values 1,4 \
  --warmup 5 --iters 10 --rounds 5 \
  --json-out /tmp/amplin-median-smoke.json
```

Result:

```text
ruff: passed
git diff --check: passed
all correctness checks: passed

M=1 Amplin batch rounds us:
  18.6368, 18.3296, 18.5344, 18.3296, 18.4320
  median=18.4320, mean=18.4525, range=18.3296-18.6368

M=1 Marlin batch rounds us:
  27.8528, 27.6480, 26.8288, 26.7264, 27.2384
  median=27.2384, mean=27.2589, range=26.7264-27.8528
  median speedup=1.478x

M=4 Amplin batch median=43.4176 us
M=4 Marlin batch median=27.4432 us
median speedup=0.632x
```

The M=4 Amplin wall rounds contained one 63.37 us sample followed by four 43.99-44.08 us samples. The new wall
median remains representative and the exact outlier is preserved in JSON rather than hidden in a mean.

Result: timing aggregation success. The report now contains enough evidence to audit every gate. Next action:
rerun the full matrix with five rounds and use median speedups plus min-max spread to select profiler targets.

### 2026-07-24 — Success: five-round median real-model matrix

Revision: `0403432c`

The target was idle before launch: 0% utilization, 0 MiB used, 34 C, 210 MHz idle SM clock, 1593 MHz memory clock,
and P0.

Command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all --dtype both --m-values 1,4,16 \
  --warmup 50 --iters 100 --rounds 5 \
  --json-out /tmp/amplin-model-shapes-median5-gpu1.json
```

The complete process took 29.79 seconds. The 248,955-byte JSON contains 156 timing rows and all five raw
batch-event/wall samples per row. All correctness gates passed.

Median speedup results:

```text
dtype  M   median wins  median >=1.05  >=1.05 in >=3/5 rounds  >=1.05 in 5/5 rounds
FP16   1      8 / 12        8 / 12             8 / 12                 5 / 12
FP16   4      3 / 12        3 / 12             3 / 12                 3 / 12
FP16  16      0 / 12        0 / 12             0 / 12                 0 / 12
BF16   1      7 / 12        7 / 12             7 / 12                 5 / 12
BF16   4      3 / 12        3 / 12             3 / 12                 3 / 12
BF16  16      0 / 12        0 / 12             0 / 12                 0 / 12
```

Strong M=1 wins and their per-round gate votes:

```text
shape                                      FP16 median/votes  BF16 median/votes
Qwen 4096->1024 KV                            2.631x 5/5         2.947x 5/5
Laguna 3072->1024 KV/expert up                2.235x 5/5         2.283x 5/5
Laguna 6144->3072 O projection                1.419x 5/5         1.764x 5/5
Qwen 4096->4096 Q/O                           1.549x 5/5         1.540x 5/5
Laguna 1024->3072 expert down                  2.996x 5/5         3.001x 4/5
Laguna 3072->6144 Q projection                 1.372x 4/5         1.384x 5/5
```

The BF16 expert-down miss was one 27.32 us Amplin round; its other four rounds were 7.47-7.64 us.

Five-of-five clear M=4 wins in both dtypes:

```text
shape                                      FP16 median  BF16 median
Qwen 4096->1024 KV                            1.408x       1.433x
Laguna 1024->3072 expert down                  2.034x       2.071x
Laguna 3072->1024 KV/expert up                 1.690x       1.678x
```

Stable losses selected for contrast:

```text
shape/M                                    FP16 median  BF16 median
Qwen 4096->12288 MLP up, M=1                 0.835x       0.807x
Qwen 12288->4096 MLP down, M=1               0.546x       0.558x
Qwen 4096->12288 MLP up, M=16                0.067x       0.070x
Qwen 12288->4096 MLP down, M=16               0.071x       0.075x
```

The M=4 result is now especially strong evidence: the same three shapes clear 5% in all five rounds and every other
shape fails it in all five rounds. M=16 has zero majority wins.

The report preserves isolated system/clock perturbations rather than allowing them to control the median. Examples:

```text
Marlin Qwen MLP-up FP16 M=16:
  median=23.398 us, range=23.378-91.013 us
Amplin Laguna expert-down BF16 M=1:
  median=7.547 us, range=7.465-27.320 us
Amplin Laguna 3072->48 FP16 M=4:
  median=9.974 us, range=9.943-37.089 us
```

Marlin-ineligible tail medians remain compact:

```text
dtype  M   3072->48 median us  3072->72 median us
FP16   1          9.748               9.779
FP16   4          9.974              10.220
FP16  16         10.240              10.332
BF16   1          9.441               9.554
BF16   4          9.728               9.984
BF16  16          9.964              10.148
```

Worst numerical errors are unchanged from the earlier deterministic run:

```text
Amplin FP16 0.0009744
Amplin BF16 0.0077991
Marlin FP16 0.0010998
Marlin BF16 0.0090489
```

Decision:

- Retain the three compact M<=4 classes as clear canonical-layout candidates.
- Treat other M=1 majority wins as research candidates until their round behavior is explained by profiles.
- Reject the row-independent schedule for every M=16 class.
- Use Qwen KV M=1 as the stable-win profile, Qwen MLP-up M=1 as the stable-loss profile, Laguna expert M=4/M=16
  as a crossover profile, and Qwen MLP-up M=16/M=256 as the row-reuse/Tensor-Core profile.

Next action: add M=64 and M=256 timing for representative Qwen dense and Laguna expert shapes, then run Nsight
Systems captures on the selected regimes.

### 2026-07-24 — Failure: row-independent schedule at M=64 and M=256

Revision: `ac707b74`

This experiment intentionally extended the retained correctness baseline beyond its M<=4 candidate range. It asked
whether its one-row-per-output-tile schedule merely crossed over gradually or became structurally unsuitable once
real batching made weight and activation reuse important.

The target was idle before launch: physical GPU 1
(`GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`) reported 0% utilization, 0 MiB used, 34 C, a 210 MHz idle SM clock,
a 1593 MHz memory clock, and P0.

Command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 \
  --shape 1024x3072 \
  --dtype both \
  --m-values 64,256 \
  --warmup 50 --iters 100 --rounds 5 \
  --json-out /tmp/amplin-model-shapes-m64-m256-gpu1.json
```

Artifact:

```text
size:   27,331 bytes
sha256: 1bdc2343aeec5d3ff5b1eb8318ca0298641fb0b0aa2a8838df8664748071664c
rows:   16
primary statistic: median batched CUDA-event time across five rounds
```

All independent FP32-dequant correctness gates passed. The batched timing medians were:

```text
shape/dtype                         M   Amplin us  Marlin us  speedup
Qwen 4096->12288 MLP up, FP16      64    1402.143     44.308   0.032x
Qwen 4096->12288 MLP up, FP16     256    5647.114    132.864   0.024x
Qwen 4096->12288 MLP up, BF16      64    1438.751     44.104   0.031x
Qwen 4096->12288 MLP up, BF16     256    5810.391    135.096   0.023x
Laguna 1024->3072 expert down, FP16 64      91.771     24.084   0.262x
Laguna 1024->3072 expert down, FP16 256    355.144     22.804   0.064x
Laguna 1024->3072 expert down, BF16 64      95.529     25.252   0.264x
Laguna 1024->3072 expert down, BF16 256    369.746     24.556   0.066x
```

The Amplin medians were stable across rounds. Examples:

```text
Qwen MLP up FP16 M=64:   1385.953-1404.058 us
Qwen MLP up FP16 M=256:  5625.907-5659.924 us
Laguna expert FP16 M=64:   91.668-93.983 us
Laguna expert BF16 M=256: 369.285-377.027 us
```

The failure mechanism is visible in the schedule's traffic accounting. It assigns a distinct CTA to every
activation row and N tile, so requested W4/scales traffic grows approximately with M even though all rows share the
same resident weights. The measured requested-byte rates near 1.1-1.2 TB/s therefore represent repeated reads, not
useful matrix-level weight reuse. Marlin's batched path amortizes those weights and increasingly uses Ampere Tensor
Cores, while this Amplin baseline remains a scalar/SIMT collection of independent GEMVs.

Decision:

- Reject the row-independent group-stride kernel for M>=16; M=64/256 makes the rejection categorical.
- Do not try to rescue batching by changing only CTA size or warp count.
- Split Amplin into at least two families:
  - compact canonical-layout GEMV/small-M kernels for the already demonstrated M<=4 winners and N tails;
  - a packed Ampere-native W4A16 GEMM family for larger M, designed around `mma.sync`/Tensor Core fragments,
    activation tiles, and weight reuse across multiple rows.
- Treat offline/runtime weight transformation as a first-class design variable. Canonical GPTQ int32 remains the
  serialization source, not a constraint on the execution layout.

Result: performance failure of the current batching schedule, with correctness preserved. This is a successful
design discriminator and is retained in the journal as a separate commit. Next action: use Nsight Systems to verify
launch structure and kernel-time dominance, then use targeted Nsight Compute captures to quantify occupancy,
memory traffic, scheduler stalls, and Tensor Core utilization before defining the batched execution layout.

### 2026-07-24 — Success: Nsight Systems regime classification

Revision: `759ddc06`

Tool:

```text
NVIDIA Nsight Systems 2024.6.2.225
```

The captures used CUDA profiler API start/stop and NVTX ranges, so imports, JIT loading, case construction, Marlin
repacking, correctness references, and warmup were outside the recorded interval. Each measured range issued 100
raw operator calls on physical GPU 1. The command pattern was:

```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx \
  --sample=none \
  --force-overwrite=true \
  --output=/tmp/amplin-nsys-20260724-0WbFnK/<case> \
  --env-var=PYTHONPATH=/root/GPT-QModel-Ultra-2,CUDA_DEVICE_ORDER=PCI_BUS_ID,\
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855,TORCH_CUDA_ARCH_LIST=8.0 \
  python scripts/profile_amplin_vs_marlin.py \
  --path <path> --dtype fp16 --m <M> --k <K> --n <N> \
  --warmup 20 --launches 100 --cuda-profiler-api
```

Reports were inspected with:

```bash
nsys stats \
  --report cuda_gpu_kern_gb_sum,cuda_kern_exec_sum,nvtx_kern_sum,nvtx_gpu_proj_sum,nvtx_pushpop_sum \
  --timeunit usec \
  /tmp/amplin-nsys-20260724-0WbFnK/<case>.nsys-rep
```

All six captures passed their independent FP32-dequant numerical gates. Every raw call mapped to exactly one kernel
launch; no CUDA memory operation was captured inside a measured range.

Kernel-only medians:

```text
case/path                              grid       block  kernel median us
Qwen KV M=1, Amplin                    64         512            7.743
Qwen KV M=1, Marlin                   124         128           19.872
Qwen MLP up M=1, Amplin               768         512           31.408
Qwen MLP up M=1, Marlin               124         128           22.016
Laguna expert down M=4, Amplin         768         128           11.936
Laguna expert down M=4, Marlin         124         128           10.880
Laguna expert down M=16, Amplin       3072         128           30.016
Laguna expert down M=16, Marlin        124         128           11.135
Qwen MLP up M=256, Amplin           196608         256         5497.197
Qwen MLP up M=256, Marlin              124         256          128.703
Laguna 3072->72 tail M=1, Amplin          5         256           10.304
```

The stable Qwen KV result is a genuine kernel win: the canonical-layout V0 kernel is 2.57x faster in the timeline
despite launching only 64 CTAs on a 124-SM device. The Qwen MLP-up loss is also a genuine kernel result: increasing
N from 1024 to 12288 expands V0 from 64 to 768 CTAs and makes it 1.43x slower than Marlin.

The Laguna expert M=4 benchmark win needs a more precise label. The five-round CUDA-event benchmark was
11.131 us Amplin versus 22.651 us Marlin, but the kernel-only timeline was 11.936 us Amplin versus 10.880 us
Marlin. Nsight's 100-call projected GPU spans were 15.041 us/call Amplin versus 33.924 us/call Marlin, and its
host NVTX ranges were 17.075 us/call versus 34.985 us/call. Therefore the measured raw-operator throughput win is
caused by the submission/dispatch path and inter-launch gaps, not by faster Amplin device execution. Kernel work
and operator throughput must remain separate gates.

At M=16, the general Amplin kernel grows to 3072 CTAs and 30.016 us while Marlin remains a 124-CTA launch at
11.135 us. At Qwen M=256 the structural failure is extreme: independent row/N16 tiles create 196,608 CTAs and a
5.497 ms median kernel, versus Marlin's 124 CTAs and 128.703 us. This validates the repeated-weight-traffic
diagnosis without relying on host timing.

The N=72 tail is a different regime. Its five-CTA grid is deliberately underfilled but provides a legal, compact
10.304 us path for a shape Marlin cannot repack. Optimizing it requires reducing fixed launch/underfill cost rather
than adding large-M reuse machinery.

Report fingerprints:

```text
a929ea806254b3f993a957239a1ce152aee4bcae579670b5f3fdb5159134f5d5  qwen-kv-m1-fp16.nsys-rep
8b7940d5a400566d3fe0f889d0d963391ecf08eb5a113002e0214e7d2366241e  qwen-mlp-up-m1-fp16.nsys-rep
bdb5ebdd1f51fb04cca3d1ed78b4124367e3e3ea370dd8e8371bc56cb892c8c3  laguna-expert-down-m4-fp16.nsys-rep
97b4d2102b0cbb63a7de52b901d601027efbb38c337b5ac3e5f5092a312e8957  laguna-expert-down-m16-fp16.nsys-rep
4be05d1fbf916a64565a1563706a011d10d3cb121af37dfefbe85eca64fc4616  qwen-mlp-up-m256-fp16.nsys-rep
e4cc36371c40915f7c54787ce41efd9d279157871502a58c5ed7ac64d2df70cc  laguna-router-tail-m1-n72-fp16.nsys-rep
```

Decision:

- Keep the Qwen KV M=1 V0 path as the primary device-kernel win for counter profiling.
- Profile Qwen MLP-up M=1 as the same-layout N-scaling loss.
- Treat the Laguna M=4 result as an operator-submission win, not a kernel win.
- Use Laguna M=16 and Qwen M=256 to quantify the row-independent CTA/traffic explosion.
- Keep N=48/72 in a compact tail family.
- Do not use Nsight-instrumented host ranges as benchmark replacements; continue using warmed CUDA-event medians
  for gates and use Nsight Systems to explain launch structure.

Result: timeline classification success. Next action: collect targeted Nsight Compute counters for the two M=1
V0 regimes, the general-kernel M=16 collapse, the M=256 batched collapse, and the N=72 underfilled tail.

### 2026-07-24 — Failure: Nsight Compute Marlin kernel-name filter

Revision: `2a069fe4`

The first Qwen KV M=1 Amplin SOL/launch/occupancy capture succeeded, but the matching Marlin command used:

```text
--kernel-name regex:marlin::Marlin
```

Nsight Compute completed the application with no kernel result:

```text
==WARNING== No kernels were profiled.
Available Kernels:
1. Marlin
```

Cause: with the default function-name matching basis, this Nsight Compute build normalizes the templated Marlin
symbol to the function name `Marlin`. The namespace-qualified expression therefore did not match. No CUDA kernel
was replayed or measured, and no performance conclusion is drawn from this command.

Result: profiler-filter failure. Retry with the profiler-reported exact filter `--kernel-name Marlin`.

### 2026-07-24 — Success: Nsight Compute bottleneck classification

Revision: `b501ec8f`

Tool:

```text
NVIDIA Nsight Compute 2025.3.1.0
```

All captures used one measured launch bracketed by CUDA profiler start/stop after 20 warmup launches. The first pass
collected `SpeedOfLight`, `LaunchStats`, and `Occupancy`; the Qwen KV Amplin follow-up collected
`MemoryWorkloadAnalysis`, `SchedulerStats`, `WarpStateStats`, and `InstructionStats`. Explicit tensor-pipe metrics
were then collected for Qwen MLP-up M=256. The command form was:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
ncu \
  --profile-from-start off \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --kernel-name <exact-filter> \
  --launch-count 1 \
  --export /tmp/amplin-ncu-20260724-LH6zXt/<case> \
  --force-overwrite \
  python scripts/profile_amplin_vs_marlin.py \
  --path <path> --dtype fp16 --m <M> --k <K> --n <N> \
  --warmup 20 --launches 1 --cuda-profiler-api
```

NCU replay changes absolute duration, so warmed event and Nsight Systems medians remain the performance gates. The
counter values and resource comparisons below are the purpose of these captures.

Qwen KV M=1, K=4096, N=1024:

```text
metric                              Amplin V0       Marlin
grid x block                        64 x 512        124 x 128
registers/thread                    32              94
static shared/block                 1.15 KiB        0
dynamic shared/block                0               166.91 KiB
waves/SM                            0.13            1.00
theoretical occupancy               100.00%          6.25%
achieved occupancy                   25.93%          6.25%
NCU duration                         11.97 us        21.54 us
compute throughput                   14.17%           4.06%
memory throughput                    16.29%           4.20%
DRAM throughput                       7.51%           4.20%
```

The Amplin detail pass found zero local-memory spilling, 74.52% L1 hit rate, 27.07% L2 hit rate, and only
180.94 GB/s memory throughput. Schedulers had no eligible warp in 67.60% of cycles, with 4.07 active but only
0.45 eligible warps per scheduler. Long-scoreboard waits consumed 6.2 of the 12.6 average cycles between issued
instructions, approximately 49.1% of the issue interval. This is an underfilled latency/scoreboard regime, not a
DRAM-bandwidth ceiling.

Qwen MLP-up M=1, K=4096, N=12288:

```text
metric                              Amplin V0       Marlin
grid x block                        768 x 512       124 x 128
waves/SM                              1.55            1.00
achieved occupancy                   80.76%           6.20%
NCU duration                         35.90 us         25.60 us
compute throughput                   59.21%           27.50%
memory throughput                    68.14%           41.97%
DRAM throughput                      30.30%           41.97%
L1/TEX throughput                    80.43%           15.95%
```

The same V0 resource footprint is no longer underfilled. It becomes a balanced compute/memory workload with one
full wave plus a 272-CTA partial wave. Increasing occupancy cannot repair this case; total scalar decode/reduction
work and traffic must fall.

Qwen MLP-up M=256:

```text
metric                              Amplin general  Marlin
grid x block                        196608 x 256     124 x 256
waves/SM                              198.19          1.00
registers/thread                       32           255
dynamic shared/block                    0           166.91 KiB
achieved occupancy                    98.61%         12.50%
NCU duration                           6.26 ms       147.87 us
compute throughput                    86.79%         56.92%
memory throughput                     98.61%         24.19%
L1/TEX throughput                     98.68%         25.45%
DRAM throughput                        2.16%         15.89%
HMMA/tensor instructions                   0       6,291,456
tensor-pipe active                         0%          59.06%
```

This kernel is already highly occupied and saturates the L1/TEX path while doing redundant row-independent work.
Its low 2.16% DRAM throughput does not make the canonical layout optimal: resident lines are repeatedly consumed
through L1 for separate rows. Marlin's much lower occupancy is not a liability here because its packed execution
layout feeds 6.29 million HMMA instructions and reuses weights across an M tile.

Laguna K=3072, N=72 tail:

```text
grid x block:             5 x 256
waves/SM:                 0.01
achieved occupancy:       12.53%
compute throughput:        0.50%
memory throughput:         0.57%
NCU replay duration:      20.99 us
Nsight Systems median:    10.304 us
```

Only five output tiles exist. More warps inside each CTA cannot fill the GPU; splitting K would require an extra
reduction mechanism and must beat an approximately 10 us single-launch baseline.

Tensor-layout conclusion:

- Canonical GPTQ `[K/8, N]` int32 is a viable serialization and direct-GEMV source layout. Its N-contiguous words
  give simple coalesced scalar loads and already win important M=1 shapes without repacking.
- It is not an Ampere-native batched execution layout. It neither matches `ldmatrix`/`mma.sync` fragment
  consumption nor exposes cross-row reuse, and the current scalar unpack/FMA path issues zero tensor instructions.
- Amplin should therefore preserve canonical int32 at the file/API boundary while allowing a derived execution
  layout selected by regime. The batched pack must be designed around Ampere HMMA fragment order and group-128
  scale application, not around historical GPTQ word order.
- Marlin proves one effective Ampere mapping, but its 166.91 KiB shared-memory footprint and 94/255-register
  kernels are comparison points, not architectural requirements for Amplin.

First implementation hypothesis:

1. Add an isolated small-N V1 for M=1, K=4096, N=1024.
2. Change the output tile from 16 columns/512 threads to 8 columns/256 threads, mapping four group quarters per
   warp. This doubles the grid from 64 to 128 CTAs while keeping total decoded weights, arithmetic, and threads
   constant.
3. Retain V0 unchanged for N>=4096, where total work rather than insufficient CTA distribution dominates.
4. Gate V1 on FP16 and BF16 correctness plus five-round raw timing against both retained V0 evidence and Marlin.
5. Treat the N=48/72 tail and M>=16 Tensor-Core family as separate designs.

Report fingerprints:

```text
1b37c1760f420981ec94210f0cabfb7de9ba531dcbbeb192cd023fcac483c0af  qwen-kv-m1-amplin-sol.ncu-rep
a63d112af04c36a9dfc4bb9ac19d866d14e4aea89c5fbd55dc041f90cb90794c  qwen-kv-m1-marlin-sol.ncu-rep
b81443e610742672fb997fd51afc7e478f77cd47c3aad9b33b084e971070afb3  qwen-kv-m1-amplin-detail.ncu-rep
c4e13ca4ebf2e9db215588ba7ba99e5876cabdccf6445bdf71dd12892b2eb0f0  qwen-mlp-up-m1-amplin-sol.ncu-rep
2e7609ea67c55c43db1bed8deab17366eab17c6695923d4ee978971ef9aaf4e4  qwen-mlp-up-m1-marlin-sol.ncu-rep
973f08df30513ad0e3de0840895df233739e496af7aff31fb568a0ca2004385c  qwen-mlp-up-m256-amplin-sol.ncu-rep
2a494148f8fee1395346da59fae04c2f619c7cb15477ea8b19f06754ea57d580  qwen-mlp-up-m256-marlin-sol.ncu-rep
5fd125766ca1e8d8ced58bce12bc88b2ad927b540cb3fcfb8f1f48873cfc3078  qwen-mlp-up-m256-amplin-tensor.ncu-rep
53a5bd00d1d1768b7df4cd0d44a3d4e01a43d2f6ec6cf7a577d3e34c312cea8e  qwen-mlp-up-m256-marlin-tensor.ncu-rep
3f100fcfd124358f0525d2d52b35402153e52d59260ea67c7d2c7f622a3f317a  laguna-tail-n72-m1-amplin-sol.ncu-rep
```

Result: counter-classification success. Next action: implement and gate the isolated Qwen KV small-N tile
hypothesis without changing V0, the general fallback, serialization, or backend selection.

### 2026-07-24 — Failure: Qwen KV 8-column small-N tile

Revision: `5c3ecf3a`

Hypothesis:

- Route only M=1, K=4096, N=1024 to a new V1 kernel.
- Replace V0's 16-column/512-thread CTA with an 8-column/256-thread CTA.
- Map four group quarters per warp, reducing 32 group contributions with 8 warps.
- Double the grid from 64 to 128 CTAs while keeping canonical int32, total decoded weights, total useful threads,
  and arithmetic constant.
- Leave V0 and the general fallback byte-for-byte unchanged for every other shape.

The experimental JIT source fingerprint was `443c7beb7a949e05`.

Correctness command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
pytest -q tests/kernels/test_amplin.py
```

Result:

```text
JIT compile: 29 seconds
tests:       21 passed in 38.59 seconds
FP16/BF16 Qwen KV numerical gates: passed
all other real-shape, batch, tail, repeat, and stream gates: passed
```

Performance command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model qwen3-8b \
  --shape 4096x1024 \
  --shape 4096x4096 \
  --shape 4096x12288 \
  --dtype both --m-values 1 \
  --warmup 50 --iters 100 --rounds 5 \
  --json-out /tmp/amplin-small-n-v1-qwen-gate.json
```

Artifact:

```text
size:   21,026 bytes
sha256: ba8e7de5b7047f0dc1e563264b88ca992367f6e95361b99349642f76c0d1c71c
```

Qwen KV five-round batched-event medians:

```text
dtype  retained V0 us  experimental V1 us  latency change  V1/Marlin
FP16          8.694            11.878          +36.6%         1.803x
BF16          7.946            10.281          +29.4%         2.051x
```

Experimental V1 raw rounds:

```text
FP16: 11.909, 11.878, 12.534, 11.848, 11.878 us
BF16: 10.281, 10.260, 10.342, 10.260, 13.138 us
```

The result rejects the initial interpretation of the NCU grid-underfill rule. V0 launches `64 * 16 = 1024`
warps; V1 launches `128 * 8 = 1024` warps. V1 spreads the same number of warps across more SMs but does not create
additional independent warp work to hide the measured long-scoreboard latency. It also changes each group's
contiguous qweight access from 16 lanes/64 bytes to 8 lanes/32 bytes and halves the warps resident in an occupied
CTA. More CTAs alone are not a useful objective when total warps and useful memory operations remain constant.

Decision:

- Reject and fully remove V1.
- Retain the original 16-column/512-thread V0 for Qwen KV.
- Do not convert profiler occupancy suggestions directly into launch-geometry changes without checking total
  device warps and transaction geometry.
- Future M=1 work must reduce dependency depth, decoded instructions, or useful bytes rather than only redistribute
  the same work.
- Continue with the separate Ampere HMMA execution-layout design for M>=16, where the missing cross-row reuse and
  zero tensor-pipe activity are categorical.

Result: performance failure with correctness preserved. The experimental kernel and route were reverted before
commit; only this measurement and conclusion remain.

### 2026-07-24 — Success: reversible Ampere HMMA execution layout

Revision: `f5ce8f86`

This step defines a derived execution layout without changing GPTQ quantization values, checkpoint serialization,
the retained canonical GEMV kernels, or backend selection.

Canonical GPTQ word addressing:

```text
logical code:       q[k, n]
canonical tensor:   qweight[K/8, N], int32
canonical word:     qweight[k / 8, n]
canonical nibble:   (word >> (4 * (k % 8))) & 0xf
```

Canonical order is appropriate when a GEMV warp holds K/group fixed and spans adjacent N: its lanes read adjacent
int32 words. Ampere FP16/BF16 HMMA instead consumes a KxN B fragment in column-major form, so each output channel
needs contiguous K values before `ldmatrix.trans`/`mma.sync`.

Amplin W4-N64-K128 physical layout:

```text
shape: [ceil(N / 64), K / 128, 64, 16], int32

tile_n      = n / 64
group       = k / 128
n_in_tile   = n % 64
k_word      = (k % 128) / 8
nibble      = k % 8

word_index  = packed[tile_n, group, n_in_tile, k_word]
```

Each group-128/N64 W4 tile is exactly 4096 contiguous bytes. For a fixed output channel, its 16 K words are
contiguous. Scales use `[ceil(N / 64), K / 128, 64]`, so the scale vector for the same CTA/group is contiguous.
This is the source layout for dequantizing a column-major shared-memory B tile.

Padding:

- Padded qweight words use the signed int32 representation of `0x88888888`, so every nibble is GPTQ's symmetric
  logical zero 8.
- Padded scales are zero.
- N divisible by 64 has no byte overhead. This covers all Qwen/Laguna classes except N=48 and N=72.
- N=48 would pad by 33.3%; N=72 would pad by 77.8%. Those shapes remain on the canonical tail family unless a
  batched-tail measurement justifies a separate pack.

Implementation:

- Added `pack_hmma_qweight` / `unpack_hmma_qweight`.
- Added `pack_hmma_scales` / `unpack_hmma_scales`.
- Added `pack_hmma_weights` to validate the canonical W4/group-128 pair together.
- Exposed `HMMA_K_TILE=128`, `HMMA_N_TILE=64`, and `HMMA_PACKED_K_WORDS=16`.
- The operations preserve the input device and dtype, produce contiguous tensors, and are explicit utilities only.

Validation:

```bash
ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py
git diff --check

PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
pytest -q tests/kernels/test_amplin.py
```

Result:

```text
ruff:       passed
diff check: passed
pytest:     36 passed in 9.29 seconds
```

The new 14-shape parameterized test covers every exact Qwen3-8B and Laguna S 2.1 `(K,N)` class. It verifies:

- packed shape and contiguity;
- bit-exact qweight round trip, including negative int32 words;
- bit-exact FP16 scale round trip;
- direct physical-index correspondence at first and last group/channel boundaries;
- `0x88888888`/zero-scale padding for N=48 and N=72;
- invalid dtype, rank, group alignment, scale shape, and logical-N rejection.

All 21 pre-existing kernel correctness/batch/stream/negative tests passed unchanged after the layout tests.

Decision:

- Freeze W4-N64-K128 as the first Amplin batched-layout contract.
- Keep canonical `[K/8,N]` as the on-disk/API source and direct-GEMV layout.
- Derive W4-N64-K128 during explicit post-initialization, not per forward.
- Prototype a raw M>=16 HMMA operator that consumes only this derived layout.
- Target a 64x64 output CTA first: one 16 KiB FP16/BF16 A group tile, one 16 KiB dequantized B group tile,
  8 warps, and FP32 accumulation. This leaves room for staging/double buffering below Marlin's measured
  166.91 KiB dynamic shared-memory footprint.
- Do not claim performance until the raw operator executes HMMA instructions, passes independent FP32-dequant
  checks, and is timed on Qwen MLP-up plus Laguna expert shapes at M=16/64/256.

Result: layout-contract success. Next action: add the explicit raw HMMA operator and its correctness-first kernel
prototype behind exact sm_80 and packed-layout validation.

### 2026-07-24 — Success: first raw HMMA correctness and performance baseline

Revision: `7be2ed66`

The first batched kernel is deliberately simple enough to audit:

```text
operator:             gptqmodel_amplin::gemm_hmma
input layout:         contiguous [..., M, K] FP16/BF16, flattened M divisible by 16
weight layout:        W4-N64-K128 int32 plus tiled same-dtype scales
output tile/CTA:      M16 x N64
threads:              128 (4 warps)
MMA primitive:        WMMA m16n16k16, A row-major, B column-major
accumulation:         FP32
static shared:        4 KiB A + 16 KiB dequantized B + 4 KiB FP32 C = 24 KiB
group pipeline:       synchronous load/dequantize, 8 K16 MMA steps, barrier
grid:                 (N / 64, M / 16)
```

Each CTA reads one packed 4 KiB W4 group/N64 tile, dequantizes it to a column-major shared B tile, shares one
M16/K128 A tile across four warps, and accumulates four M16/N16 fragments. There is no per-forward repack,
workspace, atomic, or second kernel. The raw contract rejects non-sm_80 devices, M not divisible by 16, N not
divisible by 64, incompatible packed shapes, dtype/device mismatches, and non-contiguous tensors.

The JIT source fingerprint is `3b265c774b4ab246`.

Correctness:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
pytest -q tests/kernels/test_amplin.py

pytest -q tests/test_extension_load_api.py
```

Result:

```text
Amplin tests:          40 passed in 9.24 seconds
extension API tests:   14 passed in 9.04 seconds
```

New HMMA tests cover Qwen MLP-up M=16 and Laguna expert-down M=16/64 in both FP16 and BF16, exact repeats, output
shape/dtype/device, finite values, independent FP32-dequant references, and negative contract boundaries. The
worst matrix errors in the performance run were 0.0007186 FP16 and 0.0058625 BF16, within the existing 0.002/0.02
gates.

The real-shape harness gained an opt-in `--hmma` path. Packing occurs once before correctness and timing, so the
reported forward latency includes only the raw operator. It reports separate speedups versus scalar Amplin and
Marlin and preserves all five raw event/wall rounds.

Performance command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 \
  --shape 1024x3072 \
  --dtype both \
  --m-values 16,64,256 \
  --warmup 50 --iters 100 --rounds 5 \
  --hmma \
  --json-out /tmp/amplin-hmma-v0-real-shapes.json
```

Artifact:

```text
size:   63,223 bytes
sha256: fb89860b9b00bba59080602a911593caefd38404b24426839893dfc8c638760c
```

Five-round batched-event medians:

```text
shape/dtype                    M  scalar us  HMMA us  Marlin us  HMMA/scalar  HMMA/Marlin
Qwen MLP up FP16              16    354.181  188.140     23.450      1.883x       0.125x
Qwen MLP up FP16              64   1402.240  575.488     43.581      2.437x       0.076x
Qwen MLP up FP16             256   5702.060 1697.874    128.522      3.358x       0.076x
Qwen MLP up BF16              16    362.854  180.982     25.016      2.005x       0.138x
Qwen MLP up BF16              64   1452.460  517.939     43.284      2.804x       0.084x
Qwen MLP up BF16             256   5848.900 1676.134    130.314      3.490x       0.078x
Laguna expert down FP16       16     26.993   38.461     21.524      0.702x       0.560x
Laguna expert down FP16       64     90.726   48.660     22.026      1.864x       0.453x
Laguna expert down FP16      256    354.028  135.485     22.006      2.613x       0.162x
Laguna expert down BF16       16     27.617   35.604     21.023      0.776x       0.590x
Laguna expert down BF16       64     95.027   48.445     21.555      1.962x       0.445x
Laguna expert down BF16      256    371.036  131.215     22.661      2.828x       0.173x
```

Interpretation:

- This is a successful correctness-first Tensor Core family: it removes 1.9-3.5x of the scalar baseline at the
  larger real batches and demonstrates that the new packed layout drives WMMA correctly.
- It is not competitive with Marlin and is not eligible for routing.
- M=16 Laguna is too small for this schedule: only 48 CTAs launch, and staging/WMMA overhead exceeds scalar GEMV.
- M=64/256 repeatedly read the same W4 tile for every M16 row tile. At M=256 the weight is consumed 16 times.
- The synchronous group loop cannot overlap A/W4 loads or nibble dequantization with HMMA.
- The 16x64 CTA does not exploit the primary reason batching helps: reuse one dequantized B tile across multiple
  M16 rows within the CTA.

Decision:

- Retain HMMA V0 as an explicit raw correctness/performance baseline.
- Keep M=16 on V0 only for research; do not replace scalar/Marlin paths.
- Profile HMMA V0 before tuning to measure tensor-pipe activity, occupancy, memory, and dequantization stalls.
- Prototype an M64xN64 CTA for M divisible by 64:
  - 8 warps;
  - two M16 accumulators per warp;
  - one B fragment reused for two A fragments;
  - one W4 dequantized B tile reused across four M16 rows;
  - approximately 16 KiB A + 16 KiB B + 16 KiB C = 48 KiB static shared memory.
- Preserve the M16xN64 kernel as the exact fallback for M=16/32.

Result: raw HMMA baseline success with a failed Marlin performance gate. Next action: capture V0 counters at Qwen
M=256 and Laguna M=16/256, then implement the M64xN64 reuse hypothesis as a separate experimental variant.

### 2026-07-24 — Success: HMMA V0 Nsight Compute classification

Revision: `b3064d6a`

The bounded profiler harness now accepts `--path hmma` and `--path all`. It derives the W4-N64-K128 qweight and
scale tensors before warmup and before the CUDA-profiler range, resolves the raw `gemm_hmma` operator directly,
applies the independent FP32-dequant correctness gate, and captures only requested forward launches. It rejects
HMMA profiles unless M is divisible by 16 and N is divisible by 64.

Profiler setup failures, all before a workload kernel launched:

1. `ncu --capture-range cudaProfilerApi ...` failed because Nsight Compute 2025.3 does not accept the Nsight
   Systems `--capture-range` option.
2. Replacing that option with `--profile-from-start off` exposed that `ncu` requires an explicit interpreter for
   a Python script.
3. Invoking the interpreter directly made `scripts/` the import root and failed with
   `ModuleNotFoundError: gptqmodel`.

The working invocation uses the absolute interpreter, an explicit repository `PYTHONPATH`, and
`--profile-from-start off`; the harness brackets the measured launch with `cudaProfilerStart/Stop`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
ncu --target-processes all --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
  --metrics sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tensor_op_hmma.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
  --export <report> /root/vm314t/bin/python scripts/profile_amplin_vs_marlin.py \
  --path hmma --dtype fp16 --m <M> --k <K> --n <N> \
  --warmup 20 --launches 1 --cuda-profiler-api
```

All three captures completed 15 replay passes and passed the FP32-dequant gate:

```text
shape                         M  grid CTAs  waves/SM  duration us  HMMA inst  tensor active
Qwen MLP up       4096x12288 256       3072      4.13     1968.608    6291456          4.319%
Laguna expert down 1024x3072  16         48      0.06       57.664      24576          1.517%
Laguna expert down 1024x3072 256        768      1.03      155.168     393216          3.996%
```

`duration` is profiler replay time, not the warmed five-round benchmark latency. The tensor-active column is the
explicit `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` metric. The section's elapsed-cycle
tensor-pipe values were 4.23%, 0.56%, and 3.12%, respectively.

Resource and scheduler evidence:

```text
shape/M                  regs  static smem  theor occ  achieved occ  no eligible  active/eligible warps/sched
Qwen MLP up M256           70     24576 B      37.50%        35.15%        67.92%              5.62 / 0.61
Laguna expert down M16     70     24576 B      37.50%         6.25%        88.47%              0.98 / 0.12
Laguna expert down M256    70     24576 B      37.50%        30.84%        69.48%              4.92 / 0.57
```

Memory evidence:

```text
shape/M                  memory SOL  DRAM SOL  L1/TEX hit  L2 hit
Qwen MLP up M256             83.39%     1.19%       12.34%  88.97%
Laguna expert down M16       11.09%     1.18%       10.81%  48.35%
Laguna expert down M256      61.98%     0.57%       22.88%  91.61%
```

Reports:

```text
/tmp/amplin-ncu-hmma-20260724-f0vBVD/qwen-mlp-up-m256-hmma-v0.ncu-rep
sha256 20425a55217fde923f17dc92a4967ffac1e9f9cb5c19732ecdb81c1fde34314f

/tmp/amplin-ncu-hmma-20260724-f0vBVD/laguna-expert-m16-hmma-v0.ncu-rep
sha256 474999c5d0cdde677f8992eb9f925b00ed9f6d2060f61826a762cd348775e309

/tmp/amplin-ncu-hmma-20260724-f0vBVD/laguna-expert-m256-hmma-v0.ncu-rep
sha256 ec28c618425d4444252be18d6b234367735ec23478ac162704ae7ba89f9d490a
```

Interpretation:

- HMMA emission is correct. Qwen V0 executes 6,291,456 HMMA instructions, the same count measured for Marlin at
  this shape, but its tensor pipe is active only 4.319% versus Marlin's earlier 59.06%.
- Qwen and Laguna M=256 are not DRAM-bandwidth limited. They have low DRAM SOL, high L2 hit rates, and roughly 68%
  scheduler cycles with no eligible warp. The synchronous load, nibble-dequantize, shared-store, barrier, and MMA
  sequence leaves long dependency gaps around little tensor work.
- Laguna M=16 is independently underfilled: 48 CTAs cover only 38.7% of 124 SMs for one partial wave, producing
  6.25% achieved occupancy and 88.47% no-eligible cycles.
- V0 rereads and dequantizes each packed B tile for every M16 row tile. At M=256 that is 16 reads/dequantizations.
  This is the first scheduling cost to remove before adding asynchronous stages.

Decision:

- Keep V0 and its three profiles as the control.
- Implement M64xN64 only for M divisible by 64; retain V0 for M=16/32 and all current validation.
- Require the M64 kernel to reduce packed-B traffic fourfold, preserve exact output/error gates, and beat V0 on
  both Qwen MLP-up and Laguna expert-down at M=64/256 before considering pipelining.

Result: profiler-classification success after three documented pre-launch tooling failures. Next action: prototype
the M64xN64 four-row weight-reuse CTA and compare it against both V0 and Marlin.

### 2026-07-24 — Failure: unconditional M64 weight-reuse routing

Revision: `a8fea5c4`

Hypothesis: for every M divisible by 64, make one CTA own M64xN64 so the packed W4-N64-K128 B tile is loaded and
dequantized once for four M16 row tiles instead of four times.

Prototype:

```text
kernel:                amplin_gptq_w4_group128_gemm_hmma_m64_reuse_v1_kernel
output tile/CTA:       M64 x N64
threads/warps:         256 / 8
work/warp:             two M16xN16 accumulator fragments
B reuse:               one B fragment feeds two A fragments per warp
static shared A:       64 x 128 x 2 bytes = 16 KiB
static shared B:       64 x 128 x 2 bytes = 16 KiB
static shared C:       8 per-warp 16x16 FP32 conversion buffers = 8 KiB
total static shared:   40 KiB
grid:                  (N / 64, M / 64)
```

The per-warp C scratch is reused between the warp's two accumulator fragments. This avoids the initially planned
16 KiB full M64xN64 FP32 output tile and lowers static shared memory from 48 KiB to 40 KiB. The retained
`gemm_hmma_v0` raw operator forces the original M16xN64 control so both schedules can be timed in the same process.
Packing remains outside correctness and timing ranges.

The new correctness matrix adds Qwen MLP-up and Laguna expert-down at M=64 and M=256, in FP16 and BF16. Both the
selected schedule and forced V0 are checked independently against FP32 dequantization; selected-schedule repeats
must be exact.

Validation:

```text
JIT fingerprint:       8dbf6390264f2e1e
focused HMMA tests:    7 passed, 36 deselected
complete Amplin tests: 43 passed in 9.26 seconds
extension API tests:   14 passed in 8.95 seconds
Ruff:                  passed
diff check:            passed
```

Five-round gate:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 \
  --shape 1024x3072 \
  --dtype both \
  --m-values 64 \
  --warmup 50 --iters 100 --rounds 5 \
  --hmma \
  --json-out /tmp/amplin-hmma-m64-v1-gate.json
```

Artifact:

```text
size:   30,668 bytes
sha256: 5aa0ca7f5ff76105dd55756e031f8fc9bb195981b4bae3ce49f395ee0a3864ea
```

Five-round batched-event medians:

```text
shape/dtype                    M  V0 us  M64 reuse us  reuse/V0  reuse/Marlin  max abs
Qwen MLP up FP16              64 575.785       414.351    1.390x        0.104x 0.0006819
Qwen MLP up BF16              64 518.093       429.763    1.206x        0.100x 0.0052497
Laguna expert down FP16       64  48.148        70.349    0.684x        0.322x 0.0003778
Laguna expert down BF16       64  48.548        70.062    0.693x        0.315x 0.0026821
```

The correctness errors are identical between V0 and M64 reuse for every row. The performance split follows grid
coverage, not model identity:

```text
shape                    N64 tiles  V0 CTAs at M64  reuse CTAs at M64  selected GPU SMs
Qwen 4096x12288                192              768                 192               124
Laguna 1024x3072                48              192                  48               124
```

Qwen retains at least one CTA per SM and benefits from fourfold B reuse. Laguna collapses from 192 CTAs to 48,
covering only 38.7% of the 124 SMs, so its saved dequantization work cannot offset whole-device underfill.

Decision:

- Reject `M % 64 == 0` as a sufficient routing rule.
- Do not use model names or a fixed CUDA index in schedule selection.
- Preserve the correct M64 kernel and explicit V0 control for the next experiment.
- Gate M64 reuse on runtime work coverage: M divisible by 64 and
  `(N / 64) * (M / 64) >= selected_device_sm_count`.
- This predicts V0 for Laguna M=64, reuse for Laguna M=256, and reuse for Qwen M=64/256. Confirm every prediction
  with five-round FP16/BF16 measurements before retaining the route.

Result: correctness success but unconditional-routing performance failure. This historical commit intentionally
captures the failed raw scheduling policy; no production backend routes to Amplin. Next action: add and test the
runtime one-CTA-per-SM wave guard.

### 2026-07-24 — Success: selected-device wave gate for M64 reuse

Revision: `504ff29e`

The retained schedule selection is hardware-derived:

```text
m64_grid_ctas = (logical_n / 64) * (flattened_m / 64)

use M64 reuse when:
  flattened_m is divisible by 64
  and m64_grid_ctas >= selected CUDA device SM count

otherwise:
  use M16 V0
```

The selected device is established by `CUDAGuard` from the input tensor, and its runtime
`cudaDeviceProp::multiProcessorCount` supplies the SM count. The policy does not use a fixed CUDA index, PCI order,
model name, or hard-coded 124-SM assumption. Exact sm_80 remains a separate capability gate.

The benchmark and profiler metadata now report the expected selected schedule, M64 grid CTA count, and selected
device SM count. Requested-byte accounting follows the same coverage rule, so an M-divisible-by-64 fallback is
accounted as M16 V0 rather than incorrectly receiving M64's fourfold weight-traffic reduction.

JIT and correctness:

```text
JIT fingerprint:       cd98c81a119e27d0
focused HMMA tests:    7 passed, 36 deselected
complete Amplin tests: 43 passed in 8.78 seconds
Ruff:                  passed
diff check:            passed
```

Final five-round command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 \
  --shape 1024x3072 \
  --dtype both \
  --m-values 64,256 \
  --warmup 50 --iters 100 --rounds 5 \
  --hmma \
  --json-out /tmp/amplin-hmma-wave-gate-v1-final.json
```

Artifact:

```text
size:   63,316 bytes
sha256: d957588b9fa778b3376f0732c72645c11fe82e69bc52d3431ba7c8fe10fa664e
```

Five-round batched-event medians:

```text
shape/dtype                    M  selected       V0 us  selected us  selected/V0  selected/Marlin
Qwen MLP up FP16              64  M64 reuse     702.362      506.696        1.386x            0.104x
Qwen MLP up FP16             256  M64 reuse    2132.480     1705.380        1.250x            0.091x
Qwen MLP up BF16              64  M64 reuse     622.387      529.551        1.175x            0.093x
Qwen MLP up BF16             256  M64 reuse    2099.330     1691.873        1.241x            0.093x
Laguna expert down FP16       64  M16 V0         56.678       56.484        1.003x            0.461x
Laguna expert down FP16      256  M64 reuse     160.492      130.171        1.233x            0.188x
Laguna expert down BF16       64  M16 V0         56.023       56.760        0.987x            0.374x
Laguna expert down BF16      256  M64 reuse     155.781      131.871        1.181x            0.204x
```

The host was more heavily loaded than the earlier isolated M64 failure capture, so absolute times rose across
scalar, HMMA, and Marlin paths. The primary same-process selected/V0 ratio remains decisive: every reuse branch
wins 1.175-1.386x, while the intended fallback matches V0 within 1.3%. All numerical gates pass; the worst selected
errors are 0.0007186 FP16 and 0.0058625 BF16.

Bounded one-launch Nsight Systems routing proof:

```text
shape                 predicted  observed kernel             grid       block  trace duration
Qwen MLP up M64       M64 reuse  m64_reuse_v1                192x1      256      518.269 us
Laguna expert M64     M16 V0     hmma_v0                      48x4      128       62.592 us
Laguna expert M256    M64 reuse  m64_reuse_v1                 48x4      256      132.095 us
```

Trace artifacts:

```text
/tmp/amplin-nsys-wave-gate-20260724-Zcy6GF/qwen-m64-selected.nsys-rep
sha256 a3e53e5a3e4ebede7cb310b167272e09cd137feedd2090e8826fe9f7ef0d15d7

/tmp/amplin-nsys-wave-gate-20260724-Zcy6GF/laguna-m64-selected.nsys-rep
sha256 79edb859eccdf029736792accddc4d854a0ff4bfe6711d37a894fa900e8f1805

/tmp/amplin-nsys-wave-gate-20260724-Zcy6GF/laguna-m256-selected.nsys-rep
sha256 247de3ae2367a6b2c3aefadd7b5a9bcd44986f24d9a5e78abf2f447e1a0d917f
```

Decision:

- Retain the M64xN64 kernel and one-CTA-per-SM runtime selection rule as the first successful batched Amplin
  schedule.
- Retain the explicit M16 V0 control for A/B measurement and underfilled shapes.
- Do not route a production backend to either HMMA schedule: selected Amplin remains only 0.091-0.461x Marlin.
- Profile selected M64 reuse at Qwen M=256 and Laguna M=256. Compare tensor-pipe activity, registers, occupancy,
  eligible warps, and W4/shared traffic with V0 before choosing between `cp.async` staging, a leaner
  dequantization path, or a different CTA/warp decomposition.

Result: schedule-selection success. Next action: Nsight Compute the retained M64 reuse kernel and use its measured
stall/resource split to order the next Ampere-specific prototype.

### 2026-07-24 — Success: M64 reuse Nsight Compute classification

Revision: `204fb013`

Two 15-pass Nsight Compute captures profile the selected M64 reuse kernel at M=256. Both pass the independent
FP32-dequant gate before the bounded one-launch profiler range.

Compared with the earlier M16 V0 captures:

```text
metric                              Qwen V0  Qwen M64  Laguna V0  Laguna M64
grid CTAs                              3072       768         768         192
threads/CTA                             128       256         128         256
registers/thread                         70        64          70          64
static shared/CTA                    24 KiB    40 KiB      24 KiB      40 KiB
waves/SM                                4.13      1.55        1.03        0.39
theoretical occupancy                  37.50%    50.00%      37.50%      50.00%
achieved occupancy                     35.15%    41.91%      30.84%      19.64%
NCU duration                        1968.608  1589.184     155.168     132.864 us
HMMA instructions                    6291456   6291456      393216      393216
tensor active, active-cycle metric      4.319%    5.988%      3.996%      4.548%
memory SOL                             83.39%    72.78%      61.98%      54.80%
DRAM SOL                                1.19%     1.49%       0.57%       1.30%
L1/TEX hit                              12.34%     7.46%      22.88%       7.38%
L2 hit                                  88.97%    88.24%      91.61%      89.19%
no eligible                             67.92%    74.92%      69.48%      80.25%
active warps/scheduler                    5.62      6.70        4.92        3.15
eligible warps/scheduler                  0.61      0.59        0.57        0.29
```

The M64 kernel executes exactly the same HMMA count as V0 while reading/dequantizing each packed B tile four times
less often. Its duration improves, registers fall from 70 to 64, and Qwen achieved occupancy rises. Tensor-pipe
activity improves only modestly, however, and eligible-warps-per-scheduler does not improve. Laguna trades too much
grid concurrency for reuse but still wins because it removes enough staging work.

M64 warp-stall mix, in average warp cycles per issued instruction:

```text
stall reason             Qwen M64  Laguna M64
MIO throttle                  8.98        1.38
short scoreboard              6.97        6.55
long scoreboard               3.19        3.73
barrier                        1.93        0.54
wait                           1.35        1.31
math-pipe throttle             1.03        0.41
selected                       1.00        1.00
total cycles/issued           26.71       15.94
```

Nsight classifies Qwen's MIO-throttle stall as 33.6% of cycles between issued instructions and recommends fewer,
wider shared-memory operations. It classifies Laguna's short-scoreboard stall as 41.1%, primarily dependencies on
shared-memory/MIO operations. DRAM SOL remains negligible for both. This rejects a DRAM-bandwidth explanation and
places shared staging/dequantization ahead of adding more global-memory concurrency.

Reports:

```text
/tmp/amplin-ncu-m64-reuse-20260724-Yv3jAM/qwen-mlp-up-m256-m64-v1.ncu-rep
sha256 955d4167f7788ec1eef7b5290594de7a13d3d7de77dc64e8ee50f4053cf8117d

/tmp/amplin-ncu-m64-reuse-20260724-Yv3jAM/laguna-expert-m256-m64-v1.ncu-rep
sha256 3a53697dca693fd597f24fc0a52eb6174e47cebe4e7308abaf2d119cf1f40c79
```

Decision:

- Keep the M64 schedule and wave gate unchanged as the control.
- Prototype wide B dequant stores before `cp.async`: for each aligned 16-byte destination corresponding to one
  canonical int32 W4 word, construct four FP16x2/BF16x2 pairs and issue one 128-bit shared store instead of eight
  scalar shared stores.
- Preserve the exact W4-N64-K128 physical layout and WMMA shared-B view. This isolates shared-MIO instruction
  pressure without changing qweight bytes, tile geometry, barriers, or weight reuse.
- Require all 43 correctness tests and same-process M64 selected/V1 timing on Qwen/Laguna M=64/256. If the wide
  store does not improve every branch that actually selects M64, revert it before trying asynchronous A staging.

Result: bottleneck-classification success. Next action: implement the isolated 128-bit B-dequant shared-store
prototype and apply the same real-shape batch gate.

### 2026-07-24 — Success: 128-bit B-dequant shared stores

Revision: `53228824`

The retained M64 reuse schedule previously converted each canonical int32 W4 word into eight FP16 or BF16 values
and issued eight scalar shared-memory stores. The V2 prototype keeps the W4-N64-K128 physical layout, M64xN64 CTA,
WMMA shared-B view, barriers, and runtime wave gate unchanged, but builds four packed FP16x2 or BF16x2 values and
issues one aligned `uint4` store for the same 16-byte destination.

The destination alignment follows directly from the layout:

```text
shared B base                           16-byte aligned
column stride = K tile * 2 bytes        256 bytes
packed-K word stride = 8 * 2 bytes       16 bytes
one canonical int32 W4 word             8 dequantized values = 16 bytes
```

The automatic `gemm_hmma` route selects the wide-store V2 kernel only where the existing M64 wave gate selects
reuse. M16 V0 remains byte-for-byte unchanged. A new explicit `gemm_hmma_m64_v1` operator forces the retained
scalar-store M64 control, including on an intentionally underfilled grid, so V2/V1 comparisons do not depend on
separate builds or changing the selection policy.

JIT and validation:

```text
JIT fingerprint:                 1830ca26acd0ad7c
focused HMMA tests:              7 passed, 36 deselected in 36.01 seconds
complete Amplin tests:           43 passed in 10.13 seconds
extension-loading API tests:     14 passed in 9.18 seconds
Ruff:                            passed
diff check:                      passed
```

The correctness matrix runs selected V2, forced M16 V0, and forced M64 V1 against the independent FP32-dequant
reference for both FP16 and BF16. It covers M=16, 64, and 256 across the exact Qwen3-8B and Laguna S 2.1 shapes.
All outputs retain the previous error bounds, and selected V2 and scalar-store V1 have identical numerical error.
The negative contract checks also prove that the forced M64 control rejects M not divisible by 64.

Five-round same-process command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 \
  --shape 1024x3072 \
  --dtype both \
  --m-values 64,256 \
  --warmup 50 --iters 100 --rounds 5 \
  --hmma \
  --json-out /tmp/amplin-hmma-wide-b-v2-ab-gate.json
```

Artifact:

```text
size:   82,866 bytes
sha256: 7327df5d673a1fbf50d2729e2f309a6de819a7aba0723e8168ce4fd4cf86c9cd
```

Five-round batched-event medians for branches where the automatic route actually selects M64 reuse:

```text
shape/dtype                    M    V2 us    V1 us  V2/V1
Qwen MLP up FP16              64  505.692  506.890  1.0024x
Qwen MLP up FP16             256 1682.432 1704.500  1.0131x
Qwen MLP up BF16              64  515.441  528.077  1.0245x
Qwen MLP up BF16             256 1670.697 1691.310  1.0123x
Laguna expert down FP16      256  128.901  130.488  1.0123x
Laguna expert down BF16      256  131.256  132.383  1.0086x
```

Every selected-reuse branch improves in this run, by 0.24-2.45%. Laguna M=64 correctly remains on M16 V0; its
forced M64 result is retained only as an underfilled research control. V2 remains far slower than Marlin, so this
result does not change backend routing.

Nsight Compute captured Qwen MLP up FP16 M=256 scalar-store V1 and wide-store V2 back-to-back with the same
15-pass section/metric set used for the earlier M64 classification:

```text
metric                                  M64 V1      M64 V2       change
NCU duration                         1584.928 us  1567.200 us    1.0113x
executed SASS instructions            210763776    206438400    -2.052%
memory SOL                                72.81%        72.18%   -0.63 pp
DRAM SOL                                   1.54%         1.57%   +0.03 pp
compute SOL                               22.34%        21.94%   -0.40 pp
tensor active, elapsed-cycle metric        5.29%         5.35%   +0.06 pp
L1/TEX hit                                 7.52%         7.86%   +0.34 pp
L2 hit                                    87.91%        87.42%   -0.49 pp
achieved occupancy                        41.88%        41.94%   +0.06 pp
active warps/scheduler                      6.70          6.70    unchanged
eligible warps/scheduler                    0.59          0.58   -0.01
no eligible                               74.96%        75.21%   +0.25 pp
cycles per issued instruction              26.77          27.01   +0.24
MIO throttle, cycles/issued                  8.96           8.87   -0.09
short scoreboard, cycles/issued              6.96           7.29   +0.33
long scoreboard, cycles/issued               3.28           3.19   -0.09
barrier, cycles/issued                       1.91           2.02   +0.11
registers/thread                              64             64    unchanged
static shared/CTA                         40 KiB         40 KiB    unchanged
```

Reports:

```text
/tmp/amplin-ncu-wide-b-v2-20260724-RsdUaE/qwen-mlp-up-m256-m64-v1.ncu-rep
sha256 f395ba1797ef7d65e1c9d043e0a32e811c511394a23f38006486dc019d31a523

/tmp/amplin-ncu-wide-b-v2-20260724-RsdUaE/qwen-mlp-up-m256-m64-v2.ncu-rep
sha256 880a9b007c700a5152164a1a1e3223d34ecb3b59ca90f0639895e50adde20ece
```

The 2.05% instruction reduction and 1.13% profiled-duration improvement confirm that the wide operation was
generated and removed real shared-store work. The main dependency problem did not move: eligible warps remain
approximately 0.6 per scheduler, no-eligible cycles slightly increase, and short-scoreboard stalls worsen. The
change is therefore a narrow implementation improvement rather than a schedule breakthrough.

Decision:

- Retain the aligned 128-bit B store as M64 V2 because it passes every correctness gate and improves all six
  selected-reuse real-shape/dtype branches in the same-process test.
- Retain explicit M64 V1 and M16 V0 controls for subsequent A/B profiling.
- Do not alter production backend routing; V2 is still not competitive with Marlin.
- Prototype asynchronous A-tile global-to-shared staging next. Keep B dequantization synchronous, expose V2 as an
  explicit control, and measure whether overlapping A traffic changes short-scoreboard stalls or eligible warps.

Result: narrow shared-store success. Next action: isolate `cp.async` A staging on the M64 V2 schedule and repeat
the Qwen/Laguna multi-batch correctness, same-process timing, and Nsight gate.

### 2026-07-24 — Success: aligned A-vector loads plus `cp.async` staging

Revision: `936aba55`

The M64 V2 kernel copied its 64x128 activation tile into shared memory one FP16/BF16 scalar per instruction. The
first V3 prototype instead divides the 16 KiB tile into 1,024 aligned 16-byte chunks. Each of 256 threads issues
four `cp.async.cg.shared.global` operations, commits the group, performs the unchanged W4 B-tile dequantization
while A is in flight, waits for the A group, and then enters the existing CTA barrier and WMMA loop.

Alignment is guaranteed by the current contract:

```text
PyTorch allocation base                   at least 16-byte aligned
activation row stride = K * 2 bytes       K divisible by 128, so divisible by 256 bytes
group stride = 128 * 2 bytes              256 bytes
per-thread copy granularity                16 bytes = 8 FP16/BF16 values
shared A base                              explicitly 32-byte aligned
```

The automatic route uses V3 only when the selected-device wave gate already selects M64 reuse. The following
controls remain directly callable:

```text
gemm_hmma_v0                     M16 V0
gemm_hmma_m64_v1                scalar A stores + scalar B stores
gemm_hmma_m64_v2                scalar A stores + 128-bit B stores
gemm_hmma_m64_v2_sync_a128      synchronous 128-bit A load/store + 128-bit B stores
gemm_hmma                       selected V3: 128-bit cp.async A + 128-bit B stores
```

The inline Ampere PTX is compile-time guarded. A non-sm_80 code-generation pass receives a synchronous 16-byte
copy fallback so a mixed-architecture fatbin can still compile; the existing selected-device runtime gate
continues to reject every architecture except exact sm_80 before launch.

#### Avoiding a false causal result

The first same-process and Nsight comparisons used scalar-A V2 as the control. V3 appeared 1.23-1.30x faster in
the five-round real-shape timing matrix, while Nsight showed 53-55% fewer executed SASS instructions. That result
combined two changes: 16-byte vectorization and asynchronous issue. It could not establish that overlap, rather
than width alone, was responsible.

The synchronous A128 control was therefore added before accepting the prototype. `cuobjdump` confirms distinct
sm_80 instruction paths:

```text
synchronous A128: LDG.E.128.CONSTANT + STS.128
asynchronous A:   LDGSTS.E.BYPASS.128 + LDGDEPBAR + DEPBAR.LE
```

This control separates three effects without changing tile layout, qweight bytes, B dequantization, WMMA work,
barriers, registers, shared-memory allocation, grid, or block size.

#### Correctness and build validation

```text
final JIT fingerprint:           7389b07ba5772721
focused HMMA tests:              7 passed, 36 deselected in 37.71 seconds
complete Amplin tests:           43 passed in 9.66 seconds
extension-loading API tests:     14 passed in 7.73 seconds
Ruff:                            passed
diff check:                      passed
```

For both FP16 and BF16, the HMMA matrix compares selected V3, synchronous A128, scalar-A V2, scalar-B V1, and
M16 V0 with the independent FP32-dequant reference at exact Qwen3-8B and Laguna S 2.1 M=16/64/256 shapes. All
outputs retain the same error as their controls and repeated selected launches remain bitwise stable. Negative
tests prove every forced M64 control rejects M not divisible by 64.

#### Five-round real-shape timing

Command:

```bash
PYTHONPATH=/root/GPT-QModel-Ultra-2 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 \
  --shape 1024x3072 \
  --dtype both \
  --m-values 64,256 \
  --warmup 50 --iters 100 --rounds 5 \
  --hmma \
  --json-out /tmp/amplin-hmma-async-a-v3-sync-control-final.json
```

Artifact:

```text
size:   122,221 bytes
sha256: 5b3b7f3313f285b55ef6990a04909595873c4cf7c2b6a70f3cdfdb1e7728f7b7
```

Five-round batched-event medians for branches where automatic routing selects M64:

```text
shape/dtype                    M  scalar-A V2 us  sync-A128 us  async V3 us  sync/V2  V3/sync  V3/Marlin
Qwen MLP up FP16              64          501.740       401.644      385.587    1.249x   1.042x      0.134x
Qwen MLP up FP16             256         1687.780      1393.183     1346.468    1.211x   1.035x      0.115x
Qwen MLP up BF16              64          513.956       410.378      406.528    1.252x   1.009x      0.123x
Qwen MLP up BF16             256         1667.240      1392.179     1351.383    1.198x   1.030x      0.116x
Laguna expert down FP16      256          128.492       103.506      100.065    1.241x   1.034x      0.253x
Laguna expert down BF16      256          130.662       106.609      103.383    1.226x   1.031x      0.232x
```

The synchronous 128-bit copy accounts for most of the improvement: it wins 1.198-1.252x over scalar-A V2.
With copy width matched, `cp.async` still wins every selected real-shape/dtype branch by 1.009-1.042x. Laguna M=64
continues to select M16 V0 because the M64 grid has only 48 CTAs on the runtime 124-SM device; its forced M64
controls remain research-only.

All numerical gates pass. The worst selected max-absolute errors are 0.0007186 FP16 and 0.0058625 BF16. Despite
the material improvement, V3 remains only 0.115-0.253x Marlin on selected branches and remains ineligible for
production backend routing.

#### Width-matched Nsight Compute classification

Final-source 15-pass captures compare synchronous A128 and `cp.async` at M=256:

```text
metric                                  Qwen sync  Qwen async  Laguna sync  Laguna async
NCU duration us                          1296.960    1253.888      101.856        98.528
async speedup                                           1.034x                     1.034x
executed SASS instructions               92178432    92215296      6271488       6280704
HMMA instructions                         6291456     6291456       393216        393216
memory SOL                                  84.18%      83.63%       67.17%        67.21%
DRAM SOL                                     1.77%       1.89%        1.41%         1.75%
compute SOL                                 11.86%      12.20%       10.25%        10.62%
tensor active, active-cycle metric            7.35%       7.59%        6.29%         6.62%
tensor active, elapsed-cycle metric           6.47%       6.66%        5.13%         5.30%
achieved occupancy                            42.63%      42.64%       20.54%        20.54%
active warps/scheduler                          6.82        6.82         3.28          3.28
eligible warps/scheduler                        0.234       0.242        0.177         0.182
issue-active cycles                            13.46%      13.90%       12.57%        13.24%
cycles per issued instruction                  50.66       49.10        26.11         24.75
MIO throttle, cycles/issued                     17.50       15.11         2.14          1.71
short scoreboard, cycles/issued                 17.24       19.67        15.01         15.03
long scoreboard, cycles/issued                   4.92        3.66         3.38          2.59
barrier, cycles/issued                           6.44        5.73         1.63          1.42
registers/thread                                   64          64           64            64
static shared/CTA                              40 KiB      40 KiB       40 KiB        40 KiB
```

The width-matched kernels execute almost the same instruction count; async adds only the explicit dependency
management overhead. The approximately 3.4% profiled speedup is therefore not another vector-width artifact.
`cp.async` improves eligible and issue-active warps, tensor activity, total cycles per issued instruction, MIO
throttle, long-scoreboard stalls, and barrier stalls. It does not solve the dominant dependency: short-scoreboard
stalls remain approximately 15-20 cycles per issued instruction and slightly increase on Qwen.

Final reports:

```text
/tmp/amplin-ncu-async-a-v3-20260724-69q52B/qwen-mlp-up-m256-final-sync-a128.ncu-rep
sha256 c324dacac1bfe65246b278f253d711763f1c89a163d49d2aec1d2f0fac28c4cd

/tmp/amplin-ncu-async-a-v3-20260724-69q52B/qwen-mlp-up-m256-final-async-a.ncu-rep
sha256 9a1b5987168507b6208d7f293c2024ecd9760a46d4da8fc4bf90410080ff5703

/tmp/amplin-ncu-async-a-v3-20260724-69q52B/laguna-expert-m256-final-sync-a128.ncu-rep
sha256 31f5f1d546bcb65eebe0defd641166b7cd2c1ff1665604b7de3153738c79bbc8

/tmp/amplin-ncu-async-a-v3-20260724-69q52B/laguna-expert-m256-final-async-a.ncu-rep
sha256 56a41410ed7dca2164b884a13da7082c88723ad3a55900c15429b18b7bc61415
```

The initial scalar-A comparison reports are retained as evidence of the combined vector-width plus asynchronous
effect, but are not used for the causal `cp.async` conclusion:

```text
qwen-mlp-up-m256-m64-v2.ncu-rep
sha256 c01a53895e47a919cea6508c112dc85ba87f5eb6d768b8574fcf858c062733c1

qwen-mlp-up-m256-m64-v3.ncu-rep
sha256 2931962b554b299d58758171d884978944389bd1bb3ee81317cb84a1fb85fb44

laguna-expert-m256-m64-v2.ncu-rep
sha256 e5cf15856e4552e336a28e44ba46aaad0d4afa85bbe810ee6e6a572ace591ec0

laguna-expert-m256-m64-v3.ncu-rep
sha256 ba6c6920c89850066c0d02da62068931821ce01106657da9ac340e9827f8cf81
```

Decision:

- Retain aligned 128-bit A copying as a major M64 improvement.
- Retain `cp.async` A staging as selected V3 because it passes every correctness gate and improves all six
  selected real-shape/dtype branches against the width-matched control.
- Retain synchronous A128, scalar-A V2, scalar-B V1, and M16 V0 as explicit controls.
- Keep the runtime M64 wave gate and all production backend routing unchanged.
- Target the remaining shared-load/WMMA dependency next, not more global-memory bandwidth. Prototype two
  independently live A fragments per K step so both M16 accumulator rows can have their shared loads issued
  before HMMA; measure the register/occupancy cost before considering a direct `ldmatrix`/`mma.sync` rewrite.

Result: combined vector-copy and Ampere asynchronous-staging success. Next action: isolate fragment-level
instruction parallelism against V3 and reject it if the register cliff outweighs reduced short-scoreboard stalls.

### 2026-07-24 — Failure/no-op: two source-level live A fragments

Revision: `83adcc84`

The next prototype tested whether explicitly keeping two WMMA A fragments live would expose more independent
shared-memory work to Ampere. V3 used one source-level A fragment in this order for every K16 step:

```text
load B
load A row 0
HMMA row 0
load A row 1
HMMA row 1
```

The candidate declared separate A0 and A1 fragments and expressed:

```text
load B
load A row 0
load A row 1
HMMA row 0
HMMA row 1
```

Automatic M64 selection used the candidate while a forced `gemm_hmma_m64_v3` operator preserved the exact V3
`cp.async` path as its control. The extension API, Python wrapper, correctness matrix, timing harness, and profiler
were temporarily wired for this comparison. The candidate JIT fingerprint was `f5dea44b0d3091c9`.

#### Correctness and pilot timing

The candidate compiled for `sm_80` and passed the focused Qwen/Laguna HMMA matrix:

```text
7 passed, 36 deselected in 36.45 seconds
```

This covers FP16 and BF16 at exact Qwen MLP-up 4096x12288 and Laguna expert-down 1024x3072 shapes, with
M=16/64/256 and an independent FP32-dequant reference. The forced V3 control also passed the M64 contract check.

The first direct-script benchmark invocation omitted `PYTHONPATH=.` and stopped at import time with
`ModuleNotFoundError: No module named 'gptqmodel'`; no CUDA kernel ran. The corrected pilot command was:

```bash
PYTHONPATH=. \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 \
TORCH_CUDA_ARCH_LIST=8.0 \
python scripts/benchmark_amplin_model_shapes.py \
  --device cuda:0 \
  --model all \
  --shape 4096x12288 \
  --shape 1024x3072 \
  --dtype both \
  --m-values 64,256 \
  --warmup 30 --iters 100 --rounds 3 \
  --hmma \
  --json-out /tmp/amplin-hmma-dual-a-v4-pilot.json
```

Artifact:

```text
size:   135,817 bytes
sha256: b928fb8cf7c4a03f9d342f706ec5fd795066c2313e263e6e26148b1947c37191
```

Three-round batched-event medians for the six branches where the runtime wave gate actually selected M64 are:

```text
shape/dtype                    M  candidate us   V3 us  candidate/V3
Qwen MLP up FP16              64       518.554  511.560       0.9865x
Qwen MLP up FP16             256      1805.343 1811.159       1.0032x
Qwen MLP up BF16              64       508.232  512.625       1.0086x
Qwen MLP up BF16             256      1938.719 1895.895       0.9779x
Laguna expert down FP16      256       126.956  127.713       1.0060x
Laguna expert down BF16      256       132.526  128.911       0.9727x
```

The candidate has three small apparent wins and three losses, including 2.21% and 2.73% regressions on the two
M=256 BF16 cases. Laguna M=64 is intentionally excluded because its 48-CTA M64 grid does not fill the 124-SM
device and automatic routing correctly retained M16 V0. All numerical gates passed; the worst candidate
max-absolute errors were 0.0007186 FP16 and 0.0058625 BF16.

#### Generated-code result

The timing spread is not evidence that the source rewrite created different hardware behavior. `cuobjdump`
reported identical resources for V3 and the candidate:

```text
dtype  V3 registers/thread  candidate registers/thread  static shared/CTA
FP16                    64                          64              40 KiB
BF16                    68                          68              40 KiB
```

More importantly, hashing only the 64-bit SASS instruction/control words from each forced symbol produced
identical hashes within each dtype:

```text
FP16 V3        1a0172ebd6a9f2fb3ae8752e5126f2bb93ff51d3e0d308f8afdbe0733f7011ba
FP16 candidate 1a0172ebd6a9f2fb3ae8752e5126f2bb93ff51d3e0d308f8afdbe0733f7011ba
BF16 V3        3062230f14fefc86b5efe1e009d0e70efd4a639aa6cace9302d00e7ab9d3d797
BF16 candidate 3062230f14fefc86b5efe1e009d0e70efd4a639aa6cace9302d00e7ab9d3d797
```

Inspection around the tensor-core loop confirms that NVCC already schedules B, A0, and A1 `LDSM` operations
ahead of the dependent `HMMA.16816.F32` instructions in V3. The extra C++ fragment objects neither increase
register pressure nor change a single emitted instruction. A five-round gate or Nsight Compute comparison would
only profile two names for the same executable instruction stream, so neither was run.

#### Revert validation and decision

All candidate implementation and harness changes were removed. The retained source is byte-for-byte V3 again:

```text
restored JIT fingerprint:       7389b07ba5772721
focused restored HMMA tests:    7 passed, 36 deselected in 7.50 seconds
tracked source diff after revert: empty
```

Decision:

- Reject the two-live-A source rewrite as a compiler-normalized no-op.
- Treat its mixed three-round timing deltas as noise, not as candidate speedups or regressions.
- Retain selected V3, every explicit earlier control, the M64 SM-wave gate, and production routing unchanged.
- Do not spend a direct `ldmatrix`/`mma.sync` rewrite merely recreating the same instruction order already emitted
  by WMMA.
- Make the next prototype attack work that still exists in SASS: shared B materialization and the group barrier.
  Design a native Ampere lane layout that expands GPTQ int4 values into the B fragment registers consumed by
  `mma.sync`, avoiding the shared-B store/read round trip. Establish the lane mapping with a small correctness
  microkernel before integrating the full Qwen/Laguna GEMM.

Result: failed source-level ILP experiment with a useful compiler finding. Next action: prototype a register-fed
W4-to-MMA B-fragment mapping whose packed int32 layout is designed around Ampere lane ownership.

### 2026-07-24 — Success: Ampere lane-native W4 B-fragment mapping

Revision: `a68378c2`

The next layout removes an avoidable mismatch between canonical GPTQ words and Ampere
`mma.sync.aligned.m16n8k16.row.col` operand ownership. For one K16xN16 W4 tile:

```text
quantized values: 16 * 16 = 256
information size: 256 * 4 bits = 1,024 bits = 128 bytes
warp storage:     32 lanes * one int32/lane = 128 bytes
```

Each lane belongs to one four-thread quad. `quad = lane / 4` identifies an output column in each N8 fragment,
while `thread = lane % 4` identifies two K values in each K8 half. One lane word therefore contains exactly the
eight codes consumed by that lane across the two N8 B fragments:

```text
nibble  B fragment  output column     K within K16
0       0           quad               2 * thread
1       0           quad               8 + 2 * thread
2       1           quad + 8           2 * thread
3       1           quad + 8           8 + 2 * thread
4       0           quad               2 * thread + 1
5       0           quad               8 + 2 * thread + 1
6       1           quad + 8           2 * thread + 1
7       1           quad + 8           8 + 2 * thread + 1
```

This nibble order is chosen for direct expansion into the two 32-bit registers required by each native B
fragment, not for canonical GPTQ row order. The complete reversible execution layout is:

```text
[ceil(N/64), K/128, 8 K16 steps, 4 N16 warp tiles, 32 lane words] int32
```

It contains the same number of int32 values as canonical `[K/8,N]` or the earlier
`[N64,K128,64,16]` execution layout whenever N is divisible by 64. N=48 and N=72 retain the existing logical-zero
N64 padding rule. Scales remain `[N64,K128,64]`; this checkpoint does not invent a redundant scale format.

#### Mapping proof

The new `mma_lane_tile` research operator is deliberately only one warp and one K16xN16 output tile:

- A is copied to a 512-byte shared tile and loaded with one `ldmatrix.x4`.
- Every lane loads one packed int32 W4 word.
- The word expands directly into two register-resident B fragments and applies the two output-channel scales.
- Two native `mma.sync.m16n8k16` operations produce the N16 result with FP32 accumulation.
- The documented accumulator lane mapping writes the 16x16 output.

FP16 and BF16 both match an independent FP32-dequant matrix product and repeat bit-for-bit. Contract tests reject
wrong A, qweight, scale, and dtype shapes. The candidate JIT fingerprint is `45c0fe3114b13bd6`.

Generated sm_80 code confirms that the proof exercises the intended path:

```text
dtype  SASS instructions  HMMA.16816  LDSM  STS  registers/thread  static shared
FP16                 104           2     1    1                28          512 B
BF16                 104           2     1    1                28          512 B
```

The single shared store and `LDSM` belong to A. There is no shared-B store/load round trip and no CTA-wide
post-dequant barrier.

#### Real-shape and repository validation

```text
selected lane-word ownership test:     1 passed, 45 deselected
FP16/BF16 native tile tests:            2 passed, 44 deselected
all 14 Qwen/Laguna layout round trips: 14 passed, 32 deselected
complete Amplin suite:                 47 passed in 10.13 seconds
extension-loading API suite:           14 passed in 8.92 seconds
Ruff:                                  passed
diff check:                            passed
```

The 14 layout cases are the exact Qwen3-8B and Laguna S 2.1 KxN classes already frozen in this journal, including
4096x12288, 12288x4096, 1024x3072, 3072x48, 3072x72, 3072x9216, and 12288x3072. Every unpacked int32 word is
bit-identical to its canonical input. All N-divisible-by-64 cases have byte-identical qweight size.

Decision:

- Retain the reversible Ampere lane-native pack/unpack transform and one-warp native-MMA correctness operator.
- Make no performance or backend-routing claim from a one-tile proof.
- Keep selected V3 and every CPU/non-sm_80 fallback unchanged.
- Integrate the layout only as a forced full-GEMM research path first. Preserve V3 as the exact control.
- Begin with the current M64xN64 schedule and exact Qwen/Laguna M=64/256 gates. Remove shared B and its group
  barrier, but keep A staging and the output schedule fixed so the layout effect remains attributable.

Result: layout and native-fragment correctness success. Next action: build a forced register-fed M64 GEMM against
the retained V3 control, then accept or revert it on real-model batched timing and generated-resource evidence.

### 2026-07-24 — Success: full M64 register-fed B fragments

Revision: `9b66dc86`

The first full-GEMM use of the Ampere lane layout is intentionally a forced research operator. It preserves the
selected V3 path as an exact control and changes only how B reaches `mma.sync`:

```text
property                    retained V3                 register-fed candidate
CTA tile                    M64 x N64 x K128            M64 x N64 x K128
threads / warps             256 / 8                     256 / 8
warp tile                   M32 x N16                   M32 x N16
A path                      16 KiB cp.async shared      16 KiB cp.async shared
B path                      expand to 16 KiB shared     one int32/lane -> registers
C path                      8 KiB shared scratch        direct accumulator stores
static shared / CTA         40 KiB                      16 KiB
accumulation                FP32                        FP32
architecture gate           exact sm_80                 exact sm_80
```

Every warp directly loads its lane words from
`[N64,K128,8 K16 steps,4 N16 warp tiles,32 lanes]`, expands symmetric GPTQ U4 codes into two FP16 or BF16 B
fragments, applies the unchanged `[N64,K128,64]` scales, and executes four
`mma.sync.aligned.m16n8k16.row.col.f32` instructions per K16 step. The public research operator requires M
divisible by 64 and logical N divisible by 64. It does not alter automatic backend selection, V3, M16 V0, CPU
behavior, or any non-sm_80 fallback.

The benchmark harness retains a separately forced `gemm_hmma_m64_v3` operator, so comparisons do not depend on
the runtime wave selector. It also reports a lane-path requested-byte model. With the current M32xN16 warp tile,
the two M-warps load each qweight lane word twice and scales are loaded eight times relative to unique values:

```text
(M / 64) * (2 * canonical qweight bytes + 8 * canonical scale bytes)
+ (N / 64) * activation bytes
+ output bytes
```

#### Correctness failure found before timing

The first full candidate compiled as JIT fingerprint `141fcca9ad84134e`, but failed every M64/M256 numerical
case. No timing was run. Reusing the fast dequantizer in the one-warp proof isolated the bug from indexing and
CTA scheduling: the new bit constants decoded unsigned U4, while symmetric GPTQ requires `code - 8`.

The corrected exact transforms use the FP16 subtract/add constants `0x64086408` and `0xd480d480`, and the BF16
subtract constant `0x43084308`. Fingerprint `28074a8797367d34` then passed the microkernel and all six batched
full-GEMM cases in both dtypes. Removing the now-unused scalar proof helper produced the final fingerprint
`cb8b588383462512`.

This was an implementation bug inside the same full-GEMM prototype, not a failed performance candidate. It is
recorded because the failure demonstrated that a U4 lane layout alone does not encode GPTQ's symmetric zero
point.

#### Five-round real-model performance gate

The gate used physical GPU 1 by UUID, logical `cuda:0`, exact sm_80 code, 50 warmups, 200 timed iterations, five
rounds, and CUDA-event batch medians. Both paths ran in the same process over the exact Qwen3-8B MLP-up
4096x12288 and Laguna S 2.1 expert-down 1024x3072 shapes:

```text
shape/dtype                    M    V3 us  register-B us  speedup  register-B/Marlin
Qwen MLP up FP16              64   321.193        203.018   1.582x              0.213x
Qwen MLP up FP16             256  1082.593        699.930   1.547x              0.189x
Qwen MLP up BF16              64   337.039        202.573   1.664x              0.214x
Qwen MLP up BF16             256  1101.655        699.899   1.574x              0.187x
Laguna expert down FP16       64    50.852         34.780   1.462x              0.647x
Laguna expert down FP16      256    84.035         56.504   1.487x              0.427x
Laguna expert down BF16       64    49.659         33.526   1.481x              0.705x
Laguna expert down BF16      256    87.741         56.412   1.555x              0.380x
```

Every branch improves, including the deliberately underfilled Laguna M64 control. Candidate and V3 max-absolute
errors are identical in every row. Worst errors remain 0.0007186 FP16 and 0.0058625 BF16.

Artifacts:

```text
/tmp/amplin-mma-lane-m64-pilot.json
size 156248 bytes
sha256 f1da6d86c1c36d3eb3c64efcd92e3560c87679a32325e3ac4d50601114250a93

/tmp/amplin-mma-lane-m64-gate.json
size 164213 bytes
sha256 54a01ea0b2ba9132912d4f0d8c5a02d5ef872eace91b2545deae27185bd39418
```

The result is a large improvement over Amplin V3, but it is not yet a Marlin replacement. The candidate reaches
0.187-0.705x Marlin on this matrix and therefore remains a forced research path.

#### Generated code and Nsight Compute

`cuobjdump` on the final extension confirms that removing shared B and C survives compilation:

```text
dtype  path        static SASS  HMMA  LDSM  STS  registers/thread  static shared
FP16   V3                  592    32    24   11                64         40 KiB
FP16   register-B          440    32    16    0                49         16 KiB
BF16   V3                  616    32     0   11                64         40 KiB
BF16   register-B          456    32    16    0                52         16 KiB
```

The candidate keeps all 32 static HMMA instructions while reducing static instructions by 25.7-26.0%. Its
`LDGSTS` operations and shared reads belong only to A; there are no ordinary shared stores for B or C and no
local-memory spills.

Final FP16 Nsight Compute captures used the same 15-pass section/metric set as V3. Profiler duration is replay
time and is used only for paired classification:

```text
metric                                  Qwen V3  Qwen register-B  Laguna V3  Laguna register-B
NCU duration us                        1255.488          814.496      98.720             65.792
NCU speedup                                                1.541x                           1.500x
executed SASS instructions             92215296         56045568     6280704            3615744
instruction reduction                                    39.223%                          42.431%
HMMA instructions                       6291456          6291456      393216             393216
memory SOL                                   83.87%            85.60%       67.70%              68.30%
DRAM SOL                                      1.26%             3.11%        2.56%               3.91%
compute SOL                                  12.24%            11.49%       10.70%               9.27%
tensor active, active-cycle metric            7.59%            11.75%        6.63%               9.93%
tensor active, elapsed-cycle metric           6.68%            10.31%        5.34%               8.03%
L1/TEX hit                                     4.99%            20.83%        7.38%              28.51%
L2 hit                                        88.77%            87.86%       89.15%              90.65%
achieved occupancy                            42.68%            43.20%       20.56%              20.05%
active warps/scheduler                          6.827             6.904        3.287               3.208
eligible warps/scheduler                        0.243             0.203        0.183               0.150
issue-active cycles                            13.92%            13.09%       13.30%              11.49%
cycles per issued instruction                  49.06             52.76        24.73               27.92
MIO throttle, cycles/issued                     15.16             12.65         1.68                1.92
short scoreboard, cycles/issued                 19.67             19.95        15.08               12.41
long scoreboard, cycles/issued                   3.59              6.84         2.59                6.32
barrier, cycles/issued                           5.74              5.95         1.43                1.72
registers/thread                                   64                49           64                  49
static shared/CTA                              40 KiB            16 KiB       40 KiB              16 KiB
```

Reports:

```text
/tmp/amplin-ncu-register-b-v4-20260724-Xf7vdy/qwen-mlp-up-m256-m64-v3.ncu-rep
sha256 94066e94700beba22a86b9024f9eb59bd12c75767d1c3a2fdfc9e4926171ac41

/tmp/amplin-ncu-register-b-v4-20260724-Xf7vdy/qwen-mlp-up-m256-register-b-v4.ncu-rep
sha256 cee29943818bcf97af68b4831f96ddc6b605c7851a37d61064d4c3096d9d6f62

/tmp/amplin-ncu-register-b-v4-20260724-Xf7vdy/laguna-expert-m256-m64-v3.ncu-rep
sha256 2dac82b3517edfccd899c9eacdc1773333cac8bc68c5380a5abf916302710523

/tmp/amplin-ncu-register-b-v4-20260724-Xf7vdy/laguna-expert-m256-register-b-v4.ncu-rep
sha256 c00ae6e99e8e0f4d7c74bb107f6eeb9d7c2dc94dd2910fa414c1bcd268ac666d
```

The speedup comes from doing much less work, not from higher occupancy: theoretical occupancy remains 50% and
the register allocation limit remains four CTAs/SM. Per-issued-instruction eligibility does not improve. Long
scoreboard stalls approximately double because the two M32 warps directly load the same B lane words. The
20.8-28.5% L1 hit rate shows that the second warp often finds those duplicate loads in cache, but L1/TEX remains
the dominant SOL unit.

This points to a specific next layout/schedule pair: change each warp from M32xN16 to M64xN8. Eight warps would
still cover M64xN64 and execute the same HMMA count, but each N8 B fragment would be owned by only one warp.
Packing two consecutive K16 N8 fragments into one int32/lane would retain information-optimal W4 storage, remove
the current 2x qweight loads, halve scale traffic from 8x to 4x unique values, and amortize one global lane-word
load over two K steps.

#### Validation and decision

```text
focused real-shape M16/M64/M256 and contracts: 7 passed, 40 deselected
profiler harness lane-path dry run:            passed
complete Amplin suite:                        47 passed in 9.52 seconds
extension-loading API suite:                  14 passed in 7.32 seconds
Ruff:                                         passed
Python compile check:                          passed
diff check:                                   passed
```

Decision:

- Retain the full M64 register-fed operator and forced V3 control as research paths.
- Keep automatic routing on selected V3 until a broader shape gate and backend integration decision.
- Preserve the exact sm_80 runtime gate and every fallback.
- Start an M64xN8, K32-paired lane-layout prototype as the next independently gated experiment.

Result: first full-GEMM Amplin layout success. The Ampere-native int32 order materially outperforms the earlier
shared-B design without changing quantized information or numerical behavior.

### 2026-07-24 — Failure: M64xN8 warp ownership over-duplicates A

Revision: `6a3c13a6`

The next prototype tested whether eliminating the register-B kernel's duplicate B loads was more valuable than
its M32xN16 warp-level A reuse. The candidate changed each warp to M64xN8 and paired two consecutive K16 B
fragments into one lane word:

```text
[N64, K128, 4 K32 pairs, 8 N8 warps, 32 lanes] int32
```

For each K32xN8 tile, one lane word contains the four codes needed by that lane at the first K16 in nibble
positions 0,1,4,5 and the four codes for the second K16 in positions 2,3,6,7. The existing dequantizer therefore
consumes the first fragment directly and the second after an eight-bit shift. The layout is information-optimal:
all 14 Qwen3-8B and Laguna S 2.1 shape classes round-trip bit-for-bit, including N=48/72 padding, and every
N-divisible-by-64 case has the same byte count as canonical GPTQ.

Eight warps still cover one M64xN64 CTA and execute the same number of HMMA instructions. Relative to the retained
M32xN16 register-B kernel, the intended traffic change per CTA/group was:

```text
traffic                    M32xN16 control       M64xN8 candidate
global qweight             2x unique values      1x unique values
scale loads                8x unique values      4x unique values
A ldmatrix operations      16 per K16/CTA        32 per K16/CTA
```

The candidate compiled as JIT fingerprint `9b73fac6778ed93d`. Both FP16 and BF16 passed the independent
FP32-dequant reference for exact Qwen/Laguna M=64/256 cases, contract checks passed, the profiler harness passed a
dry run, and all output errors were identical to the retained control.

#### Pilot rejection

The three-round pilot used physical GPU 1 by UUID, 30 warmups, 100 timed iterations, and paired same-process CUDA
event medians:

```text
shape/dtype                    M  M32xN16 us  M64xN8 us  candidate/control
Qwen MLP up FP16              64      203.458     394.914              0.515x
Qwen MLP up FP16             256      702.802    1355.364              0.519x
Qwen MLP up BF16              64      216.161     420.461              0.514x
Qwen MLP up BF16             256      702.633    1353.725              0.519x
Laguna expert down FP16       64       34.796      55.964              0.622x
Laguna expert down FP16      256       56.304     105.445              0.534x
Laguna expert down BF16       64       37.970      64.287              0.591x
Laguna expert down BF16      256       70.922     139.090              0.510x
```

Every row loses by 37.8-49.0%, so the prototype fails before the five-round gate.

Artifact:

```text
/tmp/amplin-k32-n8-m64-pilot.json
size 182433 bytes
sha256 293665158032ef26dd5dc11485b55958503b7513b3d584cdb0ff2aa26ceb9eab
```

#### Generated-code and Nsight explanation

The layout does reduce B-side instructions, but it exactly doubles warp-level A `LDSM` instructions:

```text
dtype  path       static SASS  HMMA  LDSM  LDG  registers/thread  static shared
FP16   M32xN16            440    32    16   18                49         16 KiB
FP16   M64xN8             400    32    32   13                52         16 KiB
BF16   M32xN16            456    32    16   18                52         16 KiB
BF16   M64xN8             408    32    32   13                49         16 KiB
```

A paired 15-pass Qwen FP16 M=256 Nsight Compute capture confirms that fewer total instructions do not compensate
for the shared-A dependency pressure:

```text
metric                                  M32xN16 control  M64xN8 candidate
NCU duration us                                 814.880          1570.688
candidate/control                                                  0.519x
executed SASS instructions                    56045568          49551360
instruction reduction                                             11.587%
HMMA instructions                              6291456           6291456
memory SOL                                        85.66%            86.86%
DRAM SOL                                           2.64%             1.48%
compute SOL                                       11.50%             5.61%
tensor active, active-cycle metric                11.75%             6.06%
tensor active, elapsed-cycle metric               10.32%             5.35%
L1/TEX hit                                         20.83%             5.00%
L2 hit                                             88.57%            89.19%
achieved occupancy                                 43.20%            42.74%
active warps/scheduler                              6.919             6.839
eligible warps/scheduler                            0.203             0.097
issue-active cycles                                13.09%             5.97%
cycles per issued instruction                      52.85            114.51
MIO throttle, cycles/issued                         12.59             41.92
short scoreboard, cycles/issued                     19.94             48.56
long scoreboard, cycles/issued                       6.94              6.59
barrier, cycles/issued                               5.93             11.57
registers/thread                                       49                52
static shared/CTA                                  16 KiB            16 KiB
```

Reports:

```text
/tmp/amplin-ncu-k32-n8-failure-20260724-XdKEVT/qwen-m256-n16-control.ncu-rep
size 984529 bytes
sha256 2a55345352d13d278b27259d0d01b7969daf1fc6c77319e84db49694268f838c

/tmp/amplin-ncu-k32-n8-failure-20260724-XdKEVT/qwen-m256-k32-n8-candidate.ncu-rep
size 984965 bytes
sha256 a4446bf803bfadbdcff9141d90af00e1ad3a5633ecf3167afd5c902dc579bfee
```

The candidate executes 11.6% fewer instructions and keeps the exact HMMA count, but eligible warps and tensor
activity roughly halve. MIO-throttle and short-scoreboard cycles more than double. The lower L1 hit rate confirms
that duplicate B loads were removed; it also confirms that those duplicate loads were mostly cheap cache hits.
On this schedule, doubling shared-A fragment loads is much more expensive.

All K32/N8 layout, kernel, operator, test, benchmark, and profiler changes were removed. The retained
M32xN16 register-B source is restored exactly to commit `6a3c13a6`.

Decision:

- Reject M64xN8 warp ownership even though its int32 layout is reversible and information-optimal.
- Treat A/B reuse balance, not qweight byte count alone, as a first-class Ampere layout constraint.
- Keep the successful M32xN16 register-B kernel and its 2x cache-resident B loads.
- Test compressed W4 sharing next: cooperatively stage only the 4 KiB packed lane words per group so paired M32
  warps can share global loads without materializing 16 KiB dequantized B or doubling A `LDSM`.

Result: failed warp-tile/layout pairing. A more compact int32 order is not automatically a faster Ampere
execution layout when it destroys warp-local A reuse.

### 2026-07-24 — Failure: compressed-W4 shared staging adds MIO pressure

Revision: `ffa1bef9`

The next prototype preserved the successful M64xN64 CTA, M32xN16 warp tile, register-fed dequantization, and
lane-native `[N64, K128, 8, 4, 32]` int32 order. It changed only how paired M32 warps obtained the same B lane
words. Instead of independently loading the packed words from global memory and relying on L1, 256 threads
cooperatively copied the information-optimal 4 KiB W4 group into shared memory:

```text
per CTA/group                  direct register-B       compressed-W4 staging
packed B global loads          2x unique words         1x unique words
packed B shared allocation     0 KiB                   4 KiB
dequantized B allocation       0 KiB                   0 KiB
A shared allocation            16 KiB                  16 KiB
A ldmatrix operations          unchanged               unchanged
```

Each thread issued one additional 16-byte `cp.async` per group. The paired M warps then loaded the same packed
word from shared memory before the unchanged register dequantization and HMMA sequence. Unlike the rejected
M64xN8 schedule, this design did not alter A reuse or the serialized Amplin layout.

The candidate compiled as JIT fingerprint `121b28ab2ff3a509`. FP16 and BF16 passed the independent FP32-dequant
reference on exact Qwen3-8B and Laguna S 2.1 M=64/256 cases, contract checks passed, and the profiler harness
passed its dry run:

```text
focused real-shape and contract suite:  7 passed, 40 deselected
profiler harness dry run:                passed
```

#### Pilot rejection

The three-round pilot used physical GPU 1 by UUID, 30 warmups, 100 timed iterations, and paired same-process CUDA
event medians. The Qwen shape was K=4096, N=12288; the Laguna shape was K=1024, N=3072:

```text
model/dtype       M  direct us  staged us  direct/staged
Qwen FP16        64    203.325    213.023          0.954x
Qwen FP16       256    702.054    722.729          0.971x
Laguna FP16      64     38.636     36.966          1.045x
Laguna FP16     256     70.984     73.677          0.963x
Qwen BF16        64    202.926    214.057          0.948x
Qwen BF16       256   1027.953   1229.343          0.836x
Laguna BF16      64     36.465     35.922          1.015x
Laguna BF16     256     71.250     73.748          0.966x
```

Only the two underfilled Laguna M=64 rows win, by 1.5-4.5%. The other six rows lose by 2.9-16.4%, including both
Qwen M values and both dtypes, so the prototype fails before the five-round all-shape gate.

Artifact:

```text
/tmp/amplin-shared-w4-m64-pilot.json
size 182767 bytes
sha256 db4b9521180cffa94f475834d96ff40847f6c2cadb18ce0fe13ec56356ae8dbf
```

#### Generated-code and Nsight explanation

The generated code retains all tensor-core work and the successful A schedule:

```text
dtype  path        static SASS  HMMA  LDSM  LDG  LDGSTS  registers/thread  static shared
FP16   direct              440    32    16   10       7                49         16 KiB
FP16   staged              448    32    16    2       8                48         20 KiB
BF16   direct              456    32    16   10       7                52         16 KiB
BF16   staged              464    32    16    2       8                48         20 KiB
```

The staged path also adds ordinary shared reads of the compressed lane words. Those reads are not included in
the `LDSM` column, which counts only `ldmatrix` operations used for A.

A paired 15-pass Qwen FP16 M=256 Nsight Compute capture shows that staging achieves its intended latency and
reuse effects, but replaces cheap L1-resident duplicate loads with more expensive shared-memory issue pressure:

```text
metric                                  direct control  staged candidate
NCU duration us                                811.040           831.072
direct/staged                                                       0.976x
executed SASS instructions                   56045568          57225216
instruction increase                                               2.105%
HMMA instructions                             6291456           6291456
memory SOL                                       85.47%             84.45%
DRAM SOL                                          2.005%              1.769%
compute SOL                                      11.474%             11.510%
tensor active, active-cycle metric               11.744%             11.481%
tensor active, elapsed-cycle metric              10.296%             10.114%
L1/TEX hit                                        20.816%              6.107%
L2 hit                                            88.883%             89.743%
achieved occupancy                                43.149%             48.849%
active warps/scheduler                             6.908               7.813
eligible warps/scheduler                           0.203               0.259
issue-active cycles                               13.088%             13.064%
cycles per issued instruction                     52.782              59.806
MIO throttle, cycles/issued                        12.648              29.068
short scoreboard, cycles/issued                    19.995              13.993
long scoreboard, cycles/issued                      6.823               2.767
barrier, cycles/issued                              5.961               8.706
registers/thread                                      49                  48
static shared/CTA                                 16 KiB              20 KiB
local-memory spills                                    0                   0
```

Reports:

```text
/tmp/amplin-ncu-shared-w4-failure-20260724-nObKZg/qwen-m256-direct-control.ncu-rep
size 994135 bytes
sha256 4d5ddddab5410521fe5031ff8cf057f4fd7c03c1cae62ef590bf30badb4a9012

/tmp/amplin-ncu-shared-w4-failure-20260724-nObKZg/qwen-m256-shared-w4-candidate.ncu-rep
size 994576 bytes
sha256 697f0f44d835fc27942daef4397fc8ae81d668025515bad41220f2735b72a975
```

The lower L1 hit rate confirms that duplicate global B reads were removed. Occupancy, eligible warps, and both
scoreboard metrics improve. Nevertheless, dynamic instructions rise 2.1%, MIO-throttle cycles more than double,
barrier pressure increases, and cycles per issued instruction rise. Direct duplicate loads are sufficiently
cache-resident that Ampere's L1 is a better exchange mechanism here than explicitly staging packed W4 in shared
memory.

All compressed-W4 kernel, operator, test, benchmark, and profiler changes were removed. The retained M32xN16
register-B source is restored exactly to commit `ffa1bef9`.

Restored implementation validation:

```text
complete Amplin suite:  47 passed in 9.44 seconds
Ruff:                   passed
diff check:             passed
```

Decision:

- Reject compressed-W4 shared staging for the current CTA geometry.
- Keep the direct register-B loads and let paired M32 warps reuse packed words through L1.
- Preserve the successful lane-native int32 order, A reuse, exact sm_80 gate, and all fallbacks.
- Test warp-level scale broadcast next. Within each warp, only one lane per four-lane quad needs to load each
  scale; `__shfl_sync` can broadcast it to the other three lanes without changing A or B ownership. This reduces
  scale loads from 8x to 2x each unique value across the two M warps, while avoiding shared memory.

Result: failed cache-management experiment. Removing duplicate global loads is not useful when the replacement
consumes more MIO and synchronization capacity than Ampere's cache-resident loads.

### 2026-07-24 — Failure: warp-quad scale broadcast does not reduce L1 transactions

Revision: `361827e8`

The next prototype kept the retained M64xN64 CTA, M32xN16 warp tile, direct register-B loads, A staging, and
lane-native int32 layout unchanged. It targeted only scale-load lane redundancy. In the retained kernel, the four
lanes of each MMA quad load the same two FP16/BF16 scale values. The two M32 warps covering the same N16 tile
repeat those loads:

```text
schedule                       scale lane requests per unique value
direct register-B control                                         8x
one loader per quad plus shuffle                                  2x
```

The candidate allowed only lane zero of each four-lane quad to issue the two 16-bit loads. It packed both scale
bit patterns into one uint32 and used one width-four `__shfl_sync` to broadcast the pair. It allocated no shared
memory and did not change B ownership, A `LDSM`, HMMA, output stores, or serialized weight data.

The candidate compiled as JIT fingerprint `99560133765f2570`. Both dtypes matched the independent FP32-dequant
reference on exact Qwen3-8B and Laguna S 2.1 M=64/256 cases, the malformed-M contract passed, and the bounded
profiler path passed:

```text
focused real-shape and contract suite:  7 passed, 40 deselected
profiler harness dry run:                passed
```

#### Pilot rejection

The three-round paired pilot used physical GPU 1 by UUID, 30 warmups, 100 timed iterations, and same-process CUDA
event medians. The Qwen shape was K=4096, N=12288; the Laguna shape was K=1024, N=3072:

```text
model/dtype       M  direct us  broadcast us  direct/broadcast
Qwen FP16        64    335.596       336.138             0.998x
Qwen FP16       256    701.850       700.826             1.001x
Laguna FP16      64     34.478        34.243             1.007x
Laguna FP16     256     56.443        56.422             1.000x
Qwen BF16        64    202.394       203.039             0.997x
Qwen BF16       256    698.573       703.918             0.992x
Laguna BF16      64     33.249        33.239             1.000x
Laguna BF16     256     56.371        56.566             0.997x
```

The result splits four nominal wins and four losses. All wins are at most 0.69%, while the largest loss is 0.76%.
There is no consistent model, dtype, or batch branch benefit, so the candidate fails before the five-round gate.

Artifact:

```text
/tmp/amplin-scale-broadcast-m64-pilot.json
size 173558 bytes
sha256 19a3187b597f7a53aba673b35067436c33ea643610851243daae528f228245f9
```

#### Generated-code and Nsight explanation

`cuobjdump` confirms that predicating scale loads does not remove either static load instruction. It adds one
shuffle plus predicate and bit-packing support:

```text
dtype  path       static SASS  HMMA  LDSM  LDG  SHFL  registers/thread  static shared
FP16   direct             440    32    16   10     0                49         16 KiB
FP16   broadcast          448    32    16   10     1                49         16 KiB
BF16   direct             456    32    16   10     0                52         16 KiB
BF16   broadcast          464    32    16   10     1                52         16 KiB
```

A paired 15-pass Qwen FP16 M=256 Nsight Compute capture isolates the effect:

```text
metric                                  direct control  broadcast candidate
NCU duration us                                814.272              816.736
direct/broadcast                                                       0.997x
executed SASS instructions                   56045568             56973312
instruction increase                                                  1.655%
HMMA instructions                             6291456              6291456
memory SOL                                       85.281%               85.415%
DRAM SOL                                          1.876%                1.857%
compute SOL                                      11.449%               11.631%
tensor active, active-cycle metric               11.748%               11.725%
tensor active, elapsed-cycle metric              10.273%               10.269%
L1/TEX hit                                        20.834%               20.835%
L2 hit                                            89.093%               88.940%
achieved occupancy                                43.209%               43.188%
active warps/scheduler                             6.906                 6.909
eligible warps/scheduler                           0.203                 0.210
issue-active cycles                               13.088%               13.273%
cycles per issued instruction                     52.770                52.052
MIO throttle, cycles/issued                        12.642                12.829
short scoreboard, cycles/issued                    19.988                20.348
long scoreboard, cycles/issued                      6.788                 5.729
barrier, cycles/issued                              5.945                 6.015
registers/thread                                      49                    49
static shared/CTA                                 16 KiB                16 KiB
```

Reports:

```text
/tmp/amplin-ncu-scale-broadcast-20260724-OHJ02D/qwen-m256-direct-control.ncu-rep
size 994279 bytes
sha256 66e7b2e518ed0e4cf0d1bd19bd60dedb388ce9bea79d7f52da241a15906068b1

/tmp/amplin-ncu-scale-broadcast-20260724-OHJ02D/qwen-m256-scale-broadcast-candidate.ncu-rep
size 994483 bytes
sha256 c4d07a8da6bfa0adba496c358a1ca42b34b330925739a9143dc5c7bfbf5a18e6
```

Long-scoreboard pressure falls and eligible-warps/issue-active metrics improve slightly, but neither the L1 hit
rate nor memory SOL changes materially. Ampere already coalesces the four identical lane addresses into the same
memory transactions. The candidate therefore reduces active lanes on the two load instructions without reducing
transactions, then adds 1.66% dynamic instructions and a shuffle dependency. That trade is neutral to negative.

All scale-broadcast kernel, operator, test, benchmark, and profiler changes were removed. The retained direct
register-B source is restored exactly to commit `361827e8`.

Restored implementation validation:

```text
complete Amplin suite:  47 passed in 9.24 seconds
Ruff:                   passed
diff check:             passed
```

Decision:

- Reject warp-quad scale broadcast for FP16/BF16 scales on this sm_80 schedule.
- Count coalesced memory transactions, not per-lane source expressions, when estimating scale traffic.
- Keep direct scale loads, direct cache-resident B loads, the lane-native int32 order, and the current A reuse.
- Test one-warp M64xN16 ownership next. Four warps would cover the same M64xN64 CTA, keep the same total A
  `LDSM` and HMMA count, and halve B/scale warp loads without shared exchange. The explicit risk is fewer active
  warps and more accumulator registers per thread; generated resources and real M=64/256 gates must decide it.

Result: failed lane-level load optimization. Ampere's coalescer had already removed the physical scale-transaction
redundancy that the source code appeared to expose.

### 2026-07-24 — Failure: one-warp M64xN16 ownership underfills low-K grids

Revision: `f35628ec`

The next prototype removed paired M-warp B ownership without increasing total A fragment loads. The retained
schedule assigns each warp M32xN16, so two warps cover M64 for one N16 tile. The candidate assigned one warp the
full M64xN16 tile:

```text
property                         M32xN16 control       one-warp M64xN16
CTA tile                         M64xN64xK128          M64xN64xK128
threads / warps                  256 / 8               128 / 4
A fragments per warp/K16         2                     4
B fragments per warp/K16         2                     2
HMMA per warp/K16                4                     8
total A LDSM per CTA/K16         16                    16
total HMMA per CTA/K16           32                    32
qweight warp loads per CTA       2x unique values      1x unique values
scale lane requests per CTA      8x unique values      4x unique values
```

Four warps still produce every element of M64xN64, use the existing lane-native int32 words, and preserve the
16 KiB `cp.async` A tile. Each warp holds eight FP32 accumulator fragments and applies one B pair to four
independently loaded M16 A fragments. No shared-B exchange, new weight format, or output reduction is involved.

The candidate compiled as JIT fingerprint `e0a3113bab5cce4a`. FP16 and BF16 matched the independent
FP32-dequant reference for exact Qwen3-8B and Laguna S 2.1 M=64/256 cases, the malformed-M contract passed, and
the profiler path completed:

```text
focused real-shape and contract suite:  7 passed, 40 deselected
profiler harness dry run:                passed
local-memory spills:                     0
```

#### Pilot rejection

The three-round pilot used physical GPU 1 by UUID, 30 warmups, 100 timed iterations, and paired same-process CUDA
event medians. The Qwen shape was K=4096, N=12288; the Laguna shape was K=1024, N=3072:

```text
model/dtype       M  direct us  deep-M us  direct/deep-M
Qwen FP16        64    203.448    206.029           0.987x
Qwen FP16       256    702.464    691.773           1.015x
Laguna FP16      64     34.693     33.935           1.022x
Laguna FP16     256     56.627     60.396           0.938x
Qwen BF16        64    203.018    209.971           0.967x
Qwen BF16       256    700.447    699.853           1.001x
Laguna BF16      64     33.423     33.946           0.985x
Laguna BF16     256     56.494     60.703           0.931x
```

The candidate wins three rows, but one is a 0.08% tie. It loses five rows and regresses both real Laguna batching
cases by 6.65-7.45% in latency. Qwen FP16 M=256 gains 1.55%, showing that deeper per-warp work can amortize B
loads on a sufficiently large grid, but the schedule fails the cross-model batch gate.

Artifact:

```text
/tmp/amplin-deep-m-m64-pilot.json
size 177988 bytes
sha256 6d8bf8e9ce5ab74c24a1c9c3e720aee723f513567bc361d0cd784c0a24113879
```

#### Generated-code result

Each candidate warp has twice the tensor and shared-A program, while the launch has half as many warps:

```text
dtype  path       static SASS/warp  HMMA/warp  LDSM/warp  registers/thread  static shared
FP16   direct                  440         32         16                49         16 KiB
FP16   deep-M                  544         64         32                64         16 KiB
BF16   direct                  456         32         16                52         16 KiB
BF16   deep-M                  560         64         32                64         16 KiB
```

Both kernels retain a 50% theoretical occupancy ceiling. The direct block is limited to four 256-thread CTAs by
register allocation; the candidate can place eight 128-thread CTAs. Equal theoretical occupancy therefore
requires enough grid CTAs to populate twice as many candidate blocks.

#### Paired Nsight Compute classification

On Qwen FP16 M=256, the 768-CTA grid contains enough work for the four-warp blocks. The candidate reduces
instructions by 26.56% and profiles 1.014x faster:

```text
metric                                  direct control  deep-M candidate
block threads                                      256               128
grid CTAs                                          768               768
NCU duration us                                814.848           803.488
direct/deep-M                                                       1.014x
executed SASS instructions                   56045568          41158656
instruction reduction                                            26.562%
HMMA instructions                             6291456           6291456
memory SOL                                       85.544%            85.663%
DRAM SOL                                          1.910%             1.570%
compute SOL                                      11.484%            10.427%
tensor active, active-cycle metric               11.746%            11.896%
tensor active, elapsed-cycle metric              10.305%            10.427%
L1/TEX hit                                        20.817%             5.024%
L2 hit                                            89.654%            90.533%
achieved occupancy                                43.194%            37.301%
active warps/scheduler                             6.907              5.969
eligible warps/scheduler                           0.203              0.117
issue-active cycles                               13.086%             9.736%
cycles per issued instruction                     52.784             61.310
MIO throttle, cycles/issued                        12.621              7.155
short scoreboard, cycles/issued                    20.007             42.165
long scoreboard, cycles/issued                      6.720              5.225
barrier, cycles/issued                              5.903              2.661
registers/thread                                      49                 64
static shared/CTA                                 16 KiB             16 KiB
```

The lower L1 hit rate confirms that paired B loads are gone. Total work falls enough to produce a narrow win even
though active/eligible warps and issue rate fall and short-scoreboard cycles double from the longer per-warp A
dependency chain.

Laguna FP16 M=256 launches only 192 CTAs. Halving warps per CTA therefore halves the available grid warps before
the hardware can exploit the candidate's eight-CTA residency:

```text
metric                                  direct control  deep-M candidate
block threads                                      256               128
grid CTAs                                          192               192
NCU duration us                                 65.376            69.856
direct/deep-M                                                       0.936x
executed SASS instructions                    3615744           2658816
instruction reduction                                            26.466%
HMMA instructions                              393216            393216
memory SOL                                       68.203%            63.430%
DRAM SOL                                          1.647%             1.484%
compute SOL                                       9.262%             7.531%
tensor active, active-cycle metric                9.951%             9.687%
tensor active, elapsed-cycle metric               8.015%             7.531%
L1/TEX hit                                        28.508%            15.960%
L2 hit                                            90.633%            90.583%
achieved occupancy                                20.062%            10.089%
active warps/scheduler                             3.211              1.602
eligible warps/scheduler                           0.151              0.088
issue-active cycles                               11.514%             8.251%
cycles per issued instruction                     27.891             19.411
MIO throttle, cycles/issued                         1.909              0.052
short scoreboard, cycles/issued                    12.412             12.719
long scoreboard, cycles/issued                      6.268              2.703
barrier, cycles/issued                              1.730              0.222
registers/thread                                      49                 64
static shared/CTA                                 16 KiB             16 KiB
```

Despite executing 26.47% fewer instructions, candidate achieved occupancy falls from 20.1% to 10.1%, active
warps halve, and tensor/compute activity declines. This is a grid-population failure, not a qweight bandwidth
failure.

Reports:

```text
/tmp/amplin-ncu-deep-m-20260724-DWJqhT/qwen-m256-direct-control.ncu-rep
size 1013580 bytes
sha256 5f954232bff14e0cb8cca46009059afecabae929923aeb26ef02bc316e54921d

/tmp/amplin-ncu-deep-m-20260724-DWJqhT/qwen-m256-deep-m-candidate.ncu-rep
size 1015250 bytes
sha256 b62e546c14fa7f3f1830991b285ad09177b54962833557fc49c259c4f67b3cb0

/tmp/amplin-ncu-deep-m-20260724-DWJqhT/laguna-m256-direct-control.ncu-rep
size 1013510 bytes
sha256 f37b5b6c56fad4f6c6b947c058ab8ba5745f6763141ba73f87ee3340c0cca0de

/tmp/amplin-ncu-deep-m-20260724-DWJqhT/laguna-m256-deep-m-candidate.ncu-rep
size 1015513 bytes
sha256 027b13a87f8b1ffb56f82aa4fc4aaee11bd4e59291445c7b742bdd0240127c89
```

All deep-M kernel, operator, test, benchmark, and profiler changes were removed. The retained M32xN16
register-B source is restored exactly to commit `f35628ec`.

Restored implementation validation:

```text
complete Amplin suite:  47 passed in 14.99 seconds
Ruff:                   passed
diff check:             passed
```

Decision:

- Reject one-warp M64xN16 as a general Qwen/Laguna schedule.
- Do not route a 1.5% Qwen FP16-only gain when real low-K M=256 batches lose more than 6%.
- Keep eight warps, M32xN16 ownership, the lane-native int32 layout, and cache-resident paired B loads.
- Test Ampere `cp.async` double buffering across K128 groups next. Two 16 KiB A buffers can overlap the next
  group's global-to-shared copy with current register-B HMMA work while retaining the successful warp geometry.
  The current register limit should keep the same 50% theoretical occupancy despite growing shared A to 32 KiB.

Result: failed warp-ownership schedule with a useful conditional result. Deeper per-warp ILP can win on a large
Qwen grid, but halving warps per CTA makes low-K expert batching underfill the GPU.

### 2026-07-24 — Failure: whole-K128 double-buffered A staging

Revision before experiment: `2b784b0f`

The next prototype kept the successful lane-native int32 weight format, direct register-B dequantization,
M64xN64 CTA, eight warps, and M32xN16 warp ownership. It changed only the activation pipeline:

```text
property                         direct register-B       double-A candidate
A shared stages                                  1                        2
static A shared/CTA                         16 KiB                   32 KiB
initial action                 copy K128, wait, sync     copy K128, wait, sync
steady state                   copy, wait, sync, HMMA    copy next K128 during current HMMA
CTA barriers over G groups                         2G                        G
weight/scale layout               unchanged lane-native int32 / packed FP16 or BF16
```

The candidate alternated two 16 KiB A buffers. After priming group zero, every group issued the next group's
16-byte-per-lane `cp.async` copies before dequantization and HMMA, then used one `wait_all` plus CTA barrier to
both make the next stage visible and prove that all warps had finished reading the current stage. This directly
tested whether Ampere could hide activation-copy latency behind the retained register-B tensor work without
changing weight ownership.

The isolated operator compiled as JIT fingerprint `e5cbe81f0756c279`. Both dtypes matched the independent
FP32-dequant reference on exact Qwen3-8B and Laguna S 2.1 M=64/256 cases, the malformed-M contract passed, and
the profiler harness completed:

```text
focused real-shape and contract suite:  7 passed, 40 deselected
profiler harness dry run:                passed
local-memory spills:                     0
```

#### Pilot rejection

The three-round pilot used physical GPU 1 by UUID, 30 warmups, 100 timed iterations, and paired same-process CUDA
event medians. It covered Qwen3-8B MLP-up K=4096, N=12288 and Laguna S 2.1 expert-down K=1024, N=3072 at both
M=64 and M=256:

```text
model/dtype       M  direct us  double-A us  direct/double-A
Qwen FP16        64    279.378      293.151             0.953x
Qwen FP16       256   1023.611     1031.946             0.992x
Laguna FP16      64     42.220       40.612             1.040x
Laguna FP16     256     69.325       71.229             0.973x
Qwen BF16        64    396.964      439.142             0.904x
Qwen BF16       256   1031.567     1040.558             0.991x
Laguna BF16      64     42.066       41.544             1.013x
Laguna BF16     256     69.253       69.427             0.997x
```

Only the two underfilled Laguna M=64 rows win. The candidate loses or ties all six deeper/larger cases, including
a 9.60% Qwen BF16 M=64 regression. Candidate and direct max-absolute errors are identical in every row. The
prototype therefore fails before the five-round all-shape gate.

Artifact:

```text
/tmp/amplin-double-a-m64-pilot.json
size 177980 bytes
sha256 0f968ba725e63d3bc6f09b73aef6a5de326e3c6ab03ce1bec4dff87d67109875
```

#### Generated-code result

`cuobjdump` confirms that the two-stage pipeline survives compilation, but its control and address state are
expensive:

```text
dtype  path      static SASS  HMMA  LDSM  LDGSTS  BAR  DEPBAR  registers/thread  static shared
FP16   direct            440    32    16        7    3       2                49         16 KiB
FP16   double-A          584    32    16       12    4       4                63         32 KiB
BF16   direct            456    32    16        7    3       2                52         16 KiB
BF16   double-A          600    32    16       12    4       4                62         32 KiB
```

Both paths still have a 50% theoretical occupancy ceiling and four resident 256-thread CTAs per SM. The direct
kernel is register-limited at four blocks; the candidate reaches both the four-block register and shared-memory
limits. There are no local-memory spills, but double buffering raises the FP16 allocation by 14 registers/thread
and static code size by 32.7%.

#### Paired Nsight Compute classification

The paired 15-pass Qwen FP16 M=256 capture shows real overlap: short- and long-scoreboard pressure fall and more
warps become eligible. The additional asynchronous-copy bookkeeping, MIO issue pressure, and longer combined
copy/read handoff outweigh it:

```text
metric                                      direct control  double-A candidate
NCU duration us                                    812.832             820.608
direct/double-A                                                           0.991x
executed SASS instructions                       56045568            58877952
instruction increase                                                     5.054%
HMMA instructions                                 6291456             6291456
memory SOL                                           85.573%              85.354%
compute SOL                                          11.487%              11.961%
tensor active, active-cycle metric                   11.741%              11.596%
L1/TEX hit                                            20.814%              20.596%
L2 hit                                                89.248%              89.590%
achieved occupancy                                    43.210%              42.223%
active warps/scheduler                                 6.913                6.758
eligible warps/scheduler                               0.203                0.225
issue-active cycles                                   13.085%              13.575%
cycles per issued instruction                         52.827               49.785
MIO throttle, cycles/issued                            12.608               19.023
short scoreboard, cycles/issued                        19.953               11.420
long scoreboard, cycles/issued                          6.841                4.148
barrier, cycles/issued                                  5.945                7.920
wait, cycles/issued                                     1.711                1.931
registers/thread                                          49                   63
static shared/CTA                                     16 KiB               32 KiB
```

The candidate collapses two synchronization points per group into one, but that remaining barrier now waits for
both next-stage copy completion and every warp's current-stage consumers. Fewer barrier instructions therefore
do not mean fewer barrier-stall cycles. The extra live pointers, stage state, and asynchronous-copy control also
increase dynamic instructions by 5.05%.

Laguna FP16 M=64 has only 48 CTAs and eight K128 groups. Dependency latency matters more than steady-state issue
capacity on this underfilled grid, so the scoreboard reduction produces the isolated win:

```text
metric                                      direct control  double-A candidate
NCU duration us                                     42.688              38.720
direct/double-A                                                           1.102x
executed SASS instructions                         903936              951936
instruction increase                                                     5.310%
HMMA instructions                                   98304               98304
memory SOL                                           26.198%              28.985%
compute SOL                                           3.562%               4.128%
tensor active, active-cycle metric                    8.385%               9.251%
L1/TEX hit                                            28.509%              28.509%
L2 hit                                                79.470%              78.518%
achieved occupancy                                    12.385%              12.254%
active warps/scheduler                                 1.980                1.991
eligible warps/scheduler                               0.117                0.140
issue-active cycles                                    9.700%              11.392%
cycles per issued instruction                         20.411               17.480
MIO throttle, cycles/issued                             0.009                0.040
short scoreboard, cycles/issued                         9.744                6.784
long scoreboard, cycles/issued                          4.885                4.647
barrier, cycles/issued                                  1.104                1.383
wait, cycles/issued                                     1.600                1.835
registers/thread                                          49                   63
static shared/CTA                                     16 KiB               32 KiB
```

Reports:

```text
/tmp/amplin-ncu-double-a-failure-20260724-RE4Tsv/qwen-m256-direct-control.ncu-rep
size 1009622 bytes
sha256 31ba2a70a873c5a2a02bd8b3af6f5c5228df870d2d8873136949fccdb6bcce7d

/tmp/amplin-ncu-double-a-failure-20260724-RE4Tsv/qwen-m256-double-a-candidate.ncu-rep
size 1011991 bytes
sha256 0e062310f79f81daf986cdae2c33b06db5304ac265b4b1177be06c37340a3499

/tmp/amplin-ncu-double-a-failure-20260724-RE4Tsv/laguna-m64-direct-control.ncu-rep
size 1009625 bytes
sha256 feab40ab186c43d4d53fca26887a2ab632a74f20eb68c6e7d156bc586117615f

/tmp/amplin-ncu-double-a-failure-20260724-RE4Tsv/laguna-m64-double-a-candidate.ncu-rep
size 1012239 bytes
sha256 1cf78b24df2760ffc2373fc9fd2751965ef7e6d159826c82bd1105167bf12c03
```

All double-A kernel, operator, test, benchmark, and profiler changes were removed. The retained single-stage
register-B source is restored exactly to commit `2b784b0f`.

Restored implementation validation:

```text
restored JIT fingerprint:       cb8b588383462512
complete Amplin suite:          47 passed in 9.44 seconds
extension-loading API suite:    14 passed in 7.62 seconds
Ruff:                           passed
diff check:                     passed
```

Decision:

- Reject whole-K128 double buffering as a general Qwen/Laguna schedule.
- Keep one 16 KiB A stage; four resident direct CTAs provide better latency hiding than carrying a second full
  A stage and its address/control state.
- Preserve the lane-native int32 weight format and direct register-B path.
- Test direct register-fed A fragments next. Loading the native A operand from global/L1 may trade four-way
  duplicate activation requests for removal of all `cp.async`, shared A, `ldmatrix`, and CTA barriers; the
  experiment must first prove the native lane mapping independently before a full GEMM gate.

Result: failed full-tile asynchronous pipeline. Ampere does overlap the next activation tile, but the successful
register-B kernel is already latency-hidden enough that whole-K128 double-buffer state and rendezvous cost more
than the saved scoreboard time on production-sized Qwen and batched Laguna shapes.

### 2026-07-24 — Success: direct global-to-register A fragment map

Revision before experiment: `a7604488`

The next proof isolated the native `mma.sync.aligned.m16n8k16.row.col` A operand. The retained tile control copies
one 16x16 low-precision A tile to 512 bytes of shared memory, executes `ldmatrix.x4`, and feeds four 32-bit A
registers per lane to HMMA. The new forced `mma_lane_tile_global_a` proof instead loads those four registers
directly from global memory.

The obvious mapping was wrong. Giving each lane the contiguous eight-element row segment used as its
`ldmatrix.x4` shared-memory address compiled as fingerprint `341cbeff187b1748`, but both dtype tests failed:

```text
FP16 mismatched elements:  220 / 256
FP16 greatest abs error:   0.139329
BF16 mismatched elements:   80 / 256
BF16 greatest abs error:   0.139085
```

This proved that `ldmatrix.x4` redistributes four logical 8x8 matrices across lane registers rather than merely
vector-loading the row segment supplied by each lane. An identity W4 B tile then made the HMMA output expose the
exact A permutation. For a lane's conventional `ldmatrix` address row `r` and a register's even matrix column
`c`, the pair that the native A operand actually expects is:

```text
source_row    = floor(r / 4) + 4 * bit(c, 3) + 8 * bit(c, 1)
source_column = 2 * (r mod 4) + 8 * bit(c, 2)
```

Here `c = 8 * floor(lane / 16) + 2 * register_index`. The 32 lanes and four registers/lane cover every aligned
two-element pair in the 16x16 A tile exactly once. Loading the inverse-map pair into each native register produced
the same output bits as the shared-`ldmatrix` control in both FP16 and BF16. The final proof JIT fingerprint is
`a8b2c258413ca6df`.

#### Generated-code proof

`cuobjdump` confirms that the direct map removes the intended shared-memory machinery while preserving both
tensor instructions:

```text
dtype  path      static SASS  HMMA  LDSM  LDG  STS  BAR  registers/thread  static shared
FP16   shared-A           72     2     1    4    1    1                26          512 B
FP16   global-A           80     2     0    7    0    0                28            0 B
BF16   shared-A           72     2     1    4    1    1                26          512 B
BF16   global-A           80     2     0    7    0    0                28            0 B
```

The compiler combines some of the four aligned pair loads, so the complete kernel has three more static `LDG`
instructions than the control, not four. It uses two more registers/thread, no shared memory, no `LDSM`, no
shared store, no barrier, and no local-memory spills.

Validation:

```text
focused shared/global A tile and contract suite:  3 passed, 44 deselected
complete Amplin suite:                            47 passed in 39.26 seconds
extension-loading API suite:                      14 passed in 7.31 seconds
Ruff:                                             passed
diff check:                                       passed
```

Decision:

- Retain the direct-A native-fragment map as an explicit one-warp proof.
- Do not make a performance claim from a one-CTA tile launch.
- Build a forced M64xN64 direct-A/register-B kernel next, keeping the same output ownership and lane-native int32
  B layout so the only performance trade is four N-warps loading each A fragment from global/L1 versus removing
  the CTA's `cp.async`, 16 KiB shared A tile, 16 `LDSM` instructions/warp, and two barriers per K128 group.
- Gate the full path on exact Qwen3-8B and Laguna S 2.1 M=64/256 FP16/BF16 shapes before considering routing.

Result: native-fragment correctness success. Ampere HMMA A operands can be fed directly from ordinary aligned
global loads with a deterministic inverse-`ldmatrix` lane map; whether L1 can cheaply absorb the four-way
activation duplication is now an empirical full-kernel question.

### 2026-07-24 — Success: full M64 direct-A and register-B path

Revision before experiment: `4f08c2aa`

The full prototype kept the successful M64xN64 CTA, 256 threads, M32xN16 warp ownership, lane-native int32 W4
layout, packed scales, register dequantization, and output stores. It changed only the A operand:

```text
property                         shared-A control              direct-A candidate
CTA / warp tile                  M64xN64 / M32xN16             M64xN64 / M32xN16
threads / CTA                    256                           256
A path                           cp.async -> shared -> LDSM    global/L1 -> four native registers
A copies per CTA                 1x logical tile               4x logical tile across N warps
B / scale layout                 identical lane-native int32 / packed low precision
B requested copies               2x logical W4 tile            2x logical W4 tile
synchronization                  cp.async waits + CTA barriers none
```

Each warp invokes the proven inverse-`ldmatrix` map for its two M16 A fragments at every K16 step. Four N warps
therefore request the same A pairs, but no A state crosses a warp and the kernel uses no shared memory,
`cp.async`, `LDSM`, or CTA barrier. The operator is exposed only as the forced research path
`mma_lane_m64_global_a`; automatic Amplin selection and every non-sm80 fallback remain unchanged.

The isolated extension compiled as JIT fingerprint `61a5b4645d957542`. Focused Qwen3-8B/Laguna S 2.1
M=64/256 FP16/BF16 execution and malformed-M tests passed:

```text
focused real-shape and contract suite:  7 passed, 40 deselected
profiler harness dry run:                passed
```

#### Generated code

The direct-A kernel removes more control instructions than its additional global loads add:

```text
dtype  path       static SASS  HMMA  LDSM  LDG  LDGSTS  BAR  DEPBAR  registers  shared
FP16   shared-A           440    32    16   10       7    3       2         49  16 KiB
FP16   direct-A           392    32     0   74       0    0       0         64       0
BF16   shared-A           456    32    16   10       7    3       2         52  16 KiB
BF16   direct-A           408    32     0   74       0    0       0         64       0
```

Both variants retain a four-CTA/SM, 50% theoretical occupancy ceiling. The direct-A candidate reaches the exact
64-register/thread limit for four 256-thread CTAs, has no local-memory spills, and removes the shared-memory
limit entirely.

SASS artifact:

```text
/tmp/amplin-global-a-61a5b4645d957542.sass
size 2957561 bytes
sha256 265d0f4c0fbc9fc84403f999d2679969b5c79c513d5c07504fb713e742e59176
```

#### Contaminated pilot and clean gate

The initial three-round pilot was numerically valid but not a performance gate. Physical GPU 1 was also running
a two-GPU Qwen3-8B quantization process, which varied between zero and more than 80% SM utilization on the
selected device. Seven of eight candidate medians won, while Qwen BF16 M=256 ranged from 791 to 1859 us and
appeared to lose 2.7%. The competing process was neither stopped nor moved, and Amplin stayed pinned to physical
GPU 1. The pilot was retained only to document why its timing was rejected:

```text
/tmp/amplin-global-a-m64-pilot.json
size 182676 bytes
sha256 2287ea1974d256806ac15d8b20d5ab17a614c212b48f686247c070d53005ae86
```

After that process completed and released its CUDA contexts, `nvidia-smi` reported zero utilization and zero
allocated memory on physical GPU 1. The clean gate used 50 warmups, 200 timed launches per round, five alternating
rounds, and the median batched CUDA-event time:

```text
model/dtype       M  shared-A us  direct-A us  shared/direct  direct/Marlin
Qwen FP16        64      203.141      200.110          1.015x          0.215x
Qwen FP16       256      701.660      690.002          1.017x          0.188x
Laguna FP16      64       34.519       30.080          1.148x          0.769x
Laguna FP16     256       56.438       55.506          1.017x          0.418x
Qwen BF16        64      202.604      199.997          1.013x          0.214x
Qwen BF16       256      699.105      687.923          1.016x          0.193x
Laguna BF16      64       33.326       30.239          1.102x          0.777x
Laguna BF16     256       56.289       55.404          1.016x          0.465x
```

All eight comparisons win, including the previously noisy Qwen BF16 M=256 case. Candidate and control have
identical max-absolute error in every row. Batch ranges are narrow: Qwen direct-A spans at most 3.95 us and
Laguna direct-A at most 1.17 us across the five rounds.

Gate artifact:

```text
/tmp/amplin-global-a-m64-gate.json
size 191413 bytes
sha256 9b305f271570c572acaa7e553149f844e739d17d18f3360d06a1e5caa5f057db
```

#### Uncontended Nsight Compute classification

Clean 15-pass Qwen3-8B MLP-up M=256 captures in both dtypes confirm that direct-A exchanges shared-memory
dependencies for cache-resident global-load dependencies:

```text
metric                                      FP16 shared  FP16 direct  BF16 shared  BF16 direct
NCU duration us                                 810.816      800.576      814.240      797.600
shared/direct                                    1.013x                     1.021x
executed SASS instructions                     56045568     49680384     59197440     52838400
instruction reduction                            11.357%                     10.742%
HMMA instructions                               6291456      6291456      6291456      6291456
memory SOL                                         85.680%       86.191%       85.770%       86.264%
compute SOL                                        11.502%       12.212%       12.161%       12.223%
tensor active, active-cycle metric                 11.740%       11.944%       11.743%       11.948%
L1/TEX hit                                          20.816%       87.396%       20.838%       87.404%
L2 hit                                              88.905%       88.979%       89.201%       88.239%
achieved occupancy                                  43.164%       44.375%       43.174%       44.506%
active warps/scheduler                               6.911         7.111         6.914         7.122
eligible warps/scheduler                             0.203         0.222         0.212         0.233
issue-active cycles                                 11.502%       10.351%       12.161%       11.019%
cycles per issued instruction                       52.804        60.196        50.023        56.712
LG throttle, cycles/issued                            3.361        24.537         3.185        23.572
long scoreboard, cycles/issued                        6.845        31.520         6.439        29.009
MIO throttle, cycles/issued                          12.616         0.149        12.995         0.143
short scoreboard, cycles/issued                      19.968         0.053        18.015         0.050
barrier, cycles/issued                                5.918         0.000         5.455         0.000
wait, cycles/issued                                   1.711         1.597         1.647         1.622
registers/thread                                         49            64            52            64
static shared/CTA                                    16 KiB             0        16 KiB             0
```

The four-way A requests are served effectively by L1: its hit rate rises from about 20.8% to 87.4%. Direct-A
eliminates essentially all MIO, short-scoreboard, and barrier pressure while raising achieved occupancy and
eligible warps slightly. Its new limit is explicit: frequent global instructions fill the LG queue, and A/B
dependencies raise long-scoreboard stalls. Despite more cycles between issued instructions, 10.7-11.4% fewer
dynamic instructions and no CTA rendezvous reduce total duration.

Reports:

```text
/tmp/amplin-ncu-global-a-final-20260724-c0YAzF/qwen-fp16-m256-shared-a.ncu-rep
size 1001905 bytes
sha256 bca5fac9ff9b4cd7b9617d2ee3577f6fcdc6356d2100b666b05045be594bf032

/tmp/amplin-ncu-global-a-final-20260724-c0YAzF/qwen-fp16-m256-global-a.ncu-rep
size 1000980 bytes
sha256 a45cc0946974e6a4fbf7c7658ff8e4533600e72fcce0a474d647e044e16b9857

/tmp/amplin-ncu-global-a-final-20260724-c0YAzF/qwen-bf16-m256-shared-a.ncu-rep
size 1002187 bytes
sha256 a7a7cff20a6dfcaeef5acbd9c7e353eec7ac8e0fffde46f6c04ad73d78471b7a

/tmp/amplin-ncu-global-a-final-20260724-c0YAzF/qwen-bf16-m256-global-a.ncu-rep
size 1001367 bytes
sha256 c0979e428fc3c3d5afc570f78e05334f75e110c81f17d1d748f661fcf35f5595
```

Final validation:

```text
complete Amplin suite:             47 passed in 9.20 seconds
extension-loading API suite:       14 passed in 6.93 seconds
torch.ops JIT extension suite:     29 passed in 6.51 seconds
Ruff:                              passed
diff check:                        passed
```

Decision:

- Retain `mma_lane_m64_global_a` as a forced exact-sm80 research path.
- Do not route it automatically: it consistently beats the shared-A Amplin control, but remains
  0.188-0.777x Marlin on this first complete schedule.
- Preserve the lane-native int32 W4 and packed-scale layouts; the winning change proves that a conventional
  shared-`ldmatrix` A pipeline is not automatically optimal on Ampere.
- Use the measured LG/long-scoreboard ceiling to test M32xN64 CTAs with four M16xN32 warps next. That ownership
  halves A duplication from 4x to 2x, but over equivalent M64 output it also doubles logical W4 requests from
  2x to 4x and scale requests from 8x to 16x. It removes shared state and doubles the M-grid CTA count so low-K
  Laguna batches preserve the total grid-warp count that defeated deep-M.

Result: full direct-A success. Ampere's L1 can absorb the four-way duplicate activation requests cheaply enough
that deleting `cp.async`, shared A, `LDSM`, and CTA barriers wins every clean Qwen/Laguna batch gate.

### 2026-07-24 — Success: M32xN64 direct-A schedule

Revision before experiment: `c6aadc27`

The next forced prototype changed output ownership without changing the lane-native int32 W4 layout. One
M32xN64 CTA now has four M16xN32 warps and 128 threads. Each warp loads one native A fragment, two adjacent
N16 lane-word fragments, and four packed scales per K16 step. Two M32 CTAs cover the same output as one prior
M64 CTA:

```text
property                         M64 direct-A control       M32 direct-A candidate
CTA / warp tile                  M64xN64 / M32xN16         M32xN64 / M16xN32
warps / threads per CTA          8 / 256                   4 / 128
M-grid CTAs                      M / 64                    M / 32
total grid warps                 M*N / 512                 M*N / 512
A requests per logical element   4x                        2x
W4 requests over M64 output      2x                        4x
scale requests over M64 output   8x                        16x
shared / local memory            0 / 0                     0 / 0
automatic routing                unchanged                 forced research path only
```

This corrects the preceding section's initial prediction: M32 does not preserve the M64 path's B reuse. It
deliberately trades twice as many W4 and scale requests for half as many duplicate A requests, finer CTA
granularity, and a different per-warp dependency graph. The total grid-warp count is unchanged because twice as
many CTAs have half as many warps.

The operator is exposed as `mma_lane_m32_global_a`. The shared validation/launch implementation is templated on
block M, but the new launch is still gated to exact sm_80, FP16/BF16, W4 group-128 symmetric GPTQ, M divisible
by 32, K divisible by 128, and N divisible by 64. Automatic Amplin selection, canonical serialization, Marlin,
CPU, and every non-sm80 fallback remain unchanged.

The isolated extension compiled as JIT fingerprint `1c06f8b0075daa38`. The per-warp generated code keeps the
same 32 HMMA instructions while shifting work from A loads to a second pair of B fragments:

```text
dtype  path  static SASS/warp  HMMA/warp  LDG/warp  registers/thread  static/local
FP16   M64                392         32        74                64           0/0
FP16   M32                440         32        52                55           0/0
BF16   M64                408         32        74                64           0/0
BF16   M32                472         32        52                56           0/0
```

SASS artifact:

```text
/tmp/amplin-m32-global-a-1c06f8b0075daa38.sass
size 3178695 bytes
sha256 3b126af91fa8a17b989a0fd56e6612dac130df5c87395c539976b42f1b817b7f
```

#### Pilot and clean five-round gate

The uncontended three-round pilot won all eight representative comparisons and was retained as a screening
artifact:

```text
/tmp/amplin-m32-global-a-pilot.json
size 211683 bytes
sha256 0bb2685ddb4faa39212493bab47f6f73519e59ccb2f0f05240cd7f9f5e09f43d
```

The clean gate used physical GPU 1 by UUID, 50 warmups, 200 timed launches per round, five alternating rounds,
and median batched CUDA-event time. `nvidia-smi` showed zero utilization and zero allocated memory on that GPU
immediately after the run:

```text
model/dtype       M  M64 direct us  M32 direct us  M64/M32  M32/Marlin
Qwen FP16        64        200.192        120.556    1.661x       0.359x
Qwen FP16       256        688.154        401.582    1.714x       0.319x
Laguna FP16      64         30.218         18.161    1.664x       1.251x
Laguna FP16     256         55.603         33.674    1.651x       0.657x
Qwen BF16        64        200.054        120.525    1.660x       0.359x
Qwen BF16       256        690.995        401.275    1.722x       0.329x
Laguna BF16      64         30.351         17.930    1.693x       1.294x
Laguna BF16     256         55.465         33.167    1.672x       0.675x
```

All eight M32 comparisons win by 1.651-1.722x and have exactly the same max-absolute error as M64. Candidate
ranges are narrow: the largest spread is 4.50 us on the 401.58 us Qwen FP16 M=256 result. The Laguna M=64
Marlin wins are also separated under the full round ranges, not just their medians:

```text
dtype  M32 range us       Marlin range us
FP16   18.063-18.217      22.231-25.272
BF16   17.925-17.956      22.241-25.462
```

This is the first Amplin Tensor-Core schedule in the prototype series to beat Marlin on a real extracted model
shape, although only for the representative Laguna K1024xN3072 M=64 regime so far.

Gate artifact:

```text
/tmp/amplin-m32-global-a-gate.json
size 221413 bytes
sha256 c887cceaf569a388bb9920aac952b70a88f4861cec092da5c79e5fc168fffc99
```

#### Paired Nsight Compute classification

Fifteen-pass Qwen3-8B MLP-up M=256 captures in both dtypes reproduce the M32/M64 speedup under replay and show
why more requested W4 traffic is still profitable:

```text
metric                                      FP16 M64    FP16 M32    BF16 M64    BF16 M32
NCU duration us                              795.712     462.144     797.152     462.976
M64/M32 NCU speedup                            1.722x                  1.722x
executed SASS instructions                  49680384    61679616    52838400    67977216
M32 instruction increase                       24.15%                  28.65%
HMMA instructions                            6291456     6291456     6291456     6291456
memory SOL                                     86.195%      79.190%      86.485%      79.638%
compute SOL                                    12.213%      22.164%      12.254%      24.565%
tensor active, active-cycle metric             11.944%      20.574%      11.946%      20.573%
L1/TEX hit                                     87.378%      78.885%      87.365%      79.007%
L2 hit                                         88.600%      83.075%      88.045%      85.164%
achieved occupancy                             44.356%      44.758%      44.491%      44.525%
active warps/scheduler                          7.104        7.116        7.118        7.165
eligible warps/scheduler                        0.221        0.336        0.233        0.406
issue-active cycles                            10.351%      22.164%      11.047%      24.565%
cycles per issued instruction                  60.159       28.365       56.684       25.654
LG throttle, cycles/issued                      24.715        1.939       23.487        2.334
long scoreboard, cycles/issued                  31.282       22.488       29.112       19.147
MIO throttle, cycles/issued                      0.148        0.053        0.143        0.046
short scoreboard, cycles/issued                  0.054        0.010        0.050        0.010
barrier, cycles/issued                           0.000        0.000        0.000        0.000
wait, cycles/issued                              1.597        1.809        1.622        1.829
registers/thread                                    64           55           64           56
block threads / grid CTAs                     256/768     128/1536      256/768     128/1536
```

M32 executes 24-29% more instructions, so this is not a byte-count shortcut. It nearly doubles tensor and
issue-active percentages, raises eligible warps by 52-74%, and cuts cycles between issued instructions by more
than half. The second B pair is cheaper than the removed A pair on this cache hierarchy and dependency graph:
LG-throttle pressure collapses and long-scoreboard pressure falls materially while occupancy and total grid
warps remain almost identical.

The separate Laguna BF16 M=64 candidate/Marlin captures preserve equal error and expose an important profiler
limit. NCU replay reports 35.200 us for M32 and 21.248 us for Marlin, reversing the stable warmed event result.
NCU replay duration is therefore not used to claim the Laguna speedup; the disjoint five-round CUDA-event ranges
above are the performance gate. The reversal is also a reason to require the broader exact-shape timing sweep
before routing this cache-sensitive schedule automatically.

Reports:

```text
/tmp/amplin-m32-ncu.7S9zpS/qwen-fp16-m256-m64.ncu-rep
size 1085435 bytes
sha256 69b3bd7126ca08cf2af09bd2f3f2d588230a25bd6f2e2ca82d1e14594e839bea

/tmp/amplin-m32-ncu.7S9zpS/qwen-fp16-m256-m32.ncu-rep
size 1086150 bytes
sha256 7c21b3319f1e66ce1f11b8b27cdf344239b37a27cded2d26a43949beb44bc33d

/tmp/amplin-m32-ncu.7S9zpS/qwen-bf16-m256-m64.ncu-rep
size 1085822 bytes
sha256 4c720b130c87fe4876067db95461030ad621b7c33891e60ad816ca25aab0ad68

/tmp/amplin-m32-ncu.7S9zpS/qwen-bf16-m256-m32.ncu-rep
size 1086778 bytes
sha256 adafee025ffa39af45222c7ffb910fd6694afd848fa26b38a0644099ee5eabf8

/tmp/amplin-m32-ncu.7S9zpS/laguna-bf16-m64-m32.ncu-rep
size 1085524 bytes
sha256 f9e2c177ee5f821163ccb4ead29fd50585b275ae8e443451e6d4118b64c207bc

/tmp/amplin-m32-ncu.7S9zpS/laguna-bf16-m64-marlin.ncu-rep
size 20889450 bytes
sha256 ddff4bc77ecb91e5123da5cc99c1a0f25a1fa3d90e69b0ae703ce759fa593e9f
```

Final validation:

```text
focused real-shape and contract suite:  7 passed, 40 deselected
profiler harness dry run:                passed
complete Amplin suite:                  47 passed in 9.72 seconds
extension-loading API suite:            14 passed in 7.63 seconds
torch.ops JIT extension suite:          29 passed in 6.55 seconds
Ruff:                                   passed
diff check:                             passed
```

The complete suite exercises the extracted Qwen3-8B and Laguna S 2.1 layout shapes, including the padded
N=48/N=72 tails, and executes representative Qwen/Laguna kernels at M=16, M=64, and M=256 in FP16 and BF16.

Decision:

- Retain `mma_lane_m32_global_a` as a forced exact-sm80 research path.
- Do not route it automatically yet. It beats M64 on every representative gate and beats Marlin at Laguna
  M=64, but only two of the legal extracted KxN shapes have full performance evidence.
- Keep the current lane-native int32 W4 layout. The result proves that an Ampere-specific GPTQ schedule can
  profitably trade W4 reuse for activation reuse and finer block ownership; minimizing qweight requests alone
  is not the right objective.
- Sweep every legal Qwen3-8B and Laguna S 2.1 KxN shape at M=32/64/256 next. Use that evidence to decide whether
  Amplin needs an M32/M64 shape selector before testing a shared or multicast B path that recovers W4 reuse.

Result: M32 schedule success. The first Amplin path to surpass Marlin does so by redesigning Ampere work
ownership around scheduler eligibility and operand cost, not by imitating Marlin's tile hierarchy.

### 2026-07-24 — Success: complete M32 real-shape crossover matrix

Revision before experiment: `dc5d08a5`

The representative M32 result justified a full performance sweep before any routing change. The benchmark
harness gained a default-off `--m32-sweep` mode that retains only M32 direct-A, legal M64 direct-A, and Marlin.
It changes neither kernels nor the default benchmark matrix; it avoids spending the gate on superseded scalar
and shared-memory controls.

The mode passed `py_compile`, Ruff, `git diff --check`, and a pinned-GPU dry run. The final matrix covered all
12 extracted shapes whose N is divisible by 64:

```text
Qwen3-8B:       4096x1024, 4096x4096, 4096x12288, 12288x4096
Laguna S 2.1:   1024x3072, 3072x1024, 3072x6144, 3072x9216,
                3072x12288, 6144x3072, 9216x3072, 12288x3072
```

The exact Laguna N=48 and N=72 shapes remain covered by bit-exact padded-layout and canonical-fallback tests,
but are not falsely reported as legal for the current N64 Tensor-Core kernel or Marlin. The performance gate
used FP16 and BF16, M=32/64/256, 50 warmups, 200 timed launches per round, five alternating rounds, and median
batched CUDA-event time: 72 M32 cases and 192 total rows.

Physical GPU 1 was selected by UUID and reported zero utilization and zero allocated memory before and after
the run. Other physical GPUs became busy during the sweep, demonstrating why PCI index alone is not a safe
hardware selection mechanism; the pinned device remained uncontended.

#### M32 versus M64

M32 wins all 48 M=64/256 comparisons, and every win is separated across the complete five-round ranges:

```text
model/role                  KxN             M64/M32 range over dtype and M
Qwen kv-proj                4096x1024                    1.678-1.769x
Qwen q/o-proj               4096x4096                    1.676-2.189x
Qwen mlp-up                 4096x12288                   1.716-1.728x
Qwen mlp-down               12288x4096                   1.087-2.231x
Laguna expert-down          1024x3072                    1.649-1.684x
Laguna kv/expert-up         3072x1024                    1.661-1.760x
Laguna q-proj-6144          3072x6144                    1.593-2.057x
Laguna q-proj-9216          3072x9216                    1.710-2.263x
Laguna dense-up             3072x12288                   1.708-1.753x
Laguna o-proj-6144          6144x3072                    1.753-1.825x
Laguna o-proj-9216          9216x3072                    1.761-1.837x
Laguna dense-down           12288x3072                   1.684-1.795x
```

The minimum is still a real 1.087x win on Qwen MLP-down BF16 M=64; the maximum is 2.263x on Laguna
K3072xN9216 FP16 M=64. M32 and M64 have exactly identical max-absolute reference error in all 48 comparisons.
The M32 ownership trade is therefore broadly superior to M64 within this lane-native direct-A kernel family,
not a specialization for the original two representative shapes.

#### M32 versus Marlin

Only four of 72 M32 cases beat Marlin. All four are the Laguna K1024xN3072 expert-down shape at M=32 or M=64:

```text
dtype   M    M32 us   Marlin us   Marlin/M32   M32 full range us   Marlin full range us
FP16   32     18.488      23.547       1.274x   18.130-19.389       23.460-36.608
FP16   64     18.335      23.306       1.271x   18.324-18.785       22.723-24.156
BF16   32     18.012      22.636       1.257x   17.812-18.847       22.292-23.613
BF16   64     18.043      23.091       1.280x   17.961-18.068       22.400-23.572
```

The ranges are disjoint, so none of these are median-only noise. The same shape loses at M=256:
0.663x FP16 and 0.686x BF16. Every other extracted shape loses in both dtypes and all three M values; their best
result is 0.878x on Laguna K3072xN1024 BF16 M=64. Overall M32/Marlin ranges from 0.162x to 1.280x.

Seven FP16 Marlin rows differ slightly in max-absolute reference error because of accumulation order; every
value remains below the independent FP16 tolerance. M32 and M64 remain exactly matched, and all 192 timed rows
passed finite/output/error validation.

Gate artifact:

```text
/tmp/amplin-m32-all-shapes-gate.json
size 462021 bytes
sha256 de201498fdcca0ab950076a96e7626e3e5f0d3b742f7d2f07ea607e56ab8eb64
```

Decision:

- Keep M32 as the preferred research schedule over M64 for every tested legal real-model shape, but do not
  connect it to automatic backend routing yet.
- Do not generalize the Laguna expert-down Marlin win into a broad claim. The measured crossover is
  K=1024, N=3072, M in {32, 64}; M=256 and all other extracted KxN classes still favor Marlin.
- Do not add an exact-model-name selector. Any future selection rule must use runtime shape, dtype, architecture,
  and measured regime rather than assuming a checkpoint identity.
- Prototype M32xN32 CTAs with two M16xN32 warps next. For N=1024, the current M=32/M=64 grids contain only
  16/32 CTAs on 124 SMs; halving CTA N doubles independent block distribution without changing total grid
  warps or logical operand requests. Add guarded N8 stores so the same experiment can execute Laguna N=48/72
  padded lane layouts instead of leaving those real shapes outside the Tensor-Core research path.

Result: full crossover-matrix success. M32 is the correct retained Amplin schedule, but the only current
Marlin crossover is a narrow low-K, moderate-N, small-batch regime; the next prototype targets block
granularity and real N tails rather than premature production routing.

### 2026-07-24 — Failure: removing N32 full-width tail predicates

Revision before experiment: `897b207c`

The M32xN32 prototype halves the CTA from four to two M16xN32 warps and from 128 to 64 threads. N32 blocks map
back into the existing `[N64,K128,K16,N16,lane]` int32 storage, so serialization and nibble ownership do not
change. For a full N64 output region, two N32 CTAs launch the same total four warps and request the same logical
A, W4, and scale data as one M32xN64 CTA; only block distribution changes.

The operator `mma_lane_m32_n32_global_a` accepts M divisible by 32 and N divisible by 8. It maps every N32 block
to the correct half of a packed N64 tile and uses guarded N8 output fragments for real tails. Padded code-8 W4
values and zero scales make the unused columns inert. This is the first Tensor-Core research path that executes
the exact Laguna N=48 and N=72 shapes rather than classifying them as N64-illegal.

The first implementation compiled as JIT fingerprint `8750232e9cac009f`. Fifteen focused tests passed after a
32-second build:

```text
standard Qwen/Laguna execution at M=32/64/256:  passed
Laguna N=48 at M=32/64/256, FP16/BF16:         passed
Laguna N=72 at M=32/64/256, FP16/BF16:         passed
repeat determinism and malformed contracts:     passed
total:                                          15 passed, 40 deselected
```

#### Guarded pilot

The initial kernel used the guarded N8 stores for every N, including widths divisible by 32. Its three-round
legal-shape pilot compared Qwen K4096xN1024 and K4096xN12288 plus Laguna K1024xN3072 and K3072xN1024 at
M=32/64/256 in both dtypes. It beat M32xN64 in 22/24 rows; 21 wins had disjoint round ranges. The two losses were
the saturated Qwen MLP-up M=256 rows at 0.915x BF16 and 0.923x FP16.

The promising low-N results included:

```text
shape/dtype/M                         N64 us     guarded N32 us   N64/N32
Qwen 4096x1024 FP16 M32                65.290          47.104       1.386x
Qwen 4096x1024 BF16 M64                59.607          42.496       1.403x
Laguna 3072x1024 FP16 M64              45.947          31.293       1.468x
Laguna 3072x1024 BF16 M256             48.599          38.062       1.277x
Laguna 1024x3072 FP16 M32              18.432          13.998       1.317x
Laguna 1024x3072 BF16 M64              18.084          15.647       1.156x
```

The same path expanded the pilot Marlin crossover beyond K1024xN3072: Laguna K3072xN1024 M=64 reached 1.291x
FP16 and 1.222x BF16. Two additional Marlin ratios near 1.01 were treated as noise pending a five-round gate.

Pilot artifact:

```text
/tmp/amplin-m32-n32-legal-pilot.json
size 213744 bytes
sha256 e709520b5c5f92c661e59486c51b6e3077067b282c8a56e10d841e7df107f87e
```

The three-round tail pilot compared against canonical GPTQ Amplin because Marlin and all N64 HMMA paths are
illegal. Tail execution was not a universal acceleration:

```text
shape/dtype/M                  canonical us   guarded N32 us   canonical/N32
N48 FP16 M32                         12.206           34.232           0.357x
N48 BF16 M256                        24.863           32.512           0.765x
N72 FP16 M256                        34.734           31.416           1.106x
N72 BF16 M256                        35.359           32.614           1.084x
```

All N48 rows and N72 M=32/64 rows lose; only N72 M=256 wins in both dtypes. Correct tail support is retained as
a research capability, but the canonical fallback remains the measured choice for the losing regimes.

Tail artifact:

```text
/tmp/amplin-m32-n32-tail-pilot.json
size 58811 bytes
sha256 1de84fb94e634e6fbdca09efb7f7039bacfa28480a8b9ad1dec1b0ebfccd0e8c
```

#### Failed unguarded specialization

Generated code showed that the guarded epilogue adds 64 static instructions per warp while leaving the main
loop, HMMA count, registers, and memory resources unchanged:

```text
dtype  specialization  static SASS/warp  HMMA  LDG  STG  registers  shared/local
FP16   guarded                      512    32   52   16         55            0/0
FP16   unguarded                    448    32   52   16         55            0/0
BF16   guarded                      544    32   52   16         56            0/0
BF16   unguarded                    480    32   52   16         56            0/0
```

The next build, JIT fingerprint `a9b593348dc123a3`, dispatched N divisible by 32 to the unguarded specialization
and kept guarded stores only for true tails. All 15 focused tests still passed. The five-round gate then
falsified the instruction-count hypothesis:

```text
unguarded N32 versus N64:       13 wins / 11 losses
full-range-separated outcomes: 10 wins / 7 losses
speedup range:                  0.929-1.115x
```

Most outcomes collapse to noise around parity. The largest regression is Qwen MLP-up BF16 M=32 at 0.929x; the
largest win is Qwen KV BF16 M=32 at 1.115x, whose ranges overlap. The only substantial separated wins are
Laguna expert-down M=256 at 1.095-1.104x. Removing 64 epilogue instructions unexpectedly erased nearly all of
the guarded pilot's block-distribution gains. Static instruction count is therefore not a valid selector
between these two compiler schedules.

Final unguarded gate artifact:

```text
/tmp/amplin-m32-n32-legal-gate.json
size 223533 bytes
sha256 903ccae9c02377cc1439baf67605c091e1fc891db69dfb6f7c6de4b80390b232
```

SASS artifact:

```text
/tmp/amplin-m32-n32-a9b593348dc123a3.sass
size 3659715 bytes
sha256 552da9f4b53a2cb5ba9deb218df59d22c4d23302e998f954af1e90275b0e49f0
```

The five-round gate finished before another dual-GPU quantizer claimed physical GPUs 0 and 1. The gate ended at
approximately 06:19:32 UTC; the competing PID 603394 started at 06:19:54 UTC. Its result is therefore
uncontaminated. No profiling or additional timing will run while that process remains on GPU 1.

Decision:

- Reject the unguarded full-width specialization despite its smaller SASS.
- Preserve this checkpoint with the unguarded dispatch so the failure is independently reproducible, then
  restore guarded dispatch in the next commit and rerun a clean five-round gate.
- Keep the N32/tail operator forced and unrouted. Correct N48/N72 execution is a success, but only measured wins
  may influence a later selector.
- Profile guarded and N64 schedules after GPU 1 clears to determine whether instruction ordering, block
  residency, or another compiler effect produced the large guarded-pilot advantage.

Result: unguarded-specialization failure. Deleting predicated epilogue code preserved all visible resource
counts and numerical behavior but destroyed the expected N32 performance gains.

### 2026-07-24 — Success: migrate exclusive Amplin testing to PCI-order GPU 2

The user superseded the initial GPU 1 assignment and directed all Amplin testing to use only physical GPU 2 in PCI
bus order. The live inventory was re-probed rather than treating a CUDA ordinal as a hardware capability. PCI-order
entry 2 resolved to:

| Property | Value |
| --- | --- |
| Inventory index at capture | 2 |
| UUID | `GPU-8be4c651-4058-83df-154b-291f1b86add8` |
| PCI bus at capture | `00000000:64:00.0` |
| Device | NVIDIA PG506-230 |
| Compute capability | 8.0 |
| SMs | 124 |
| NVIDIA-SMI memory | 98,304 MiB |
| PyTorch memory | 97,457 MiB |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |

At validation time, NVIDIA-SMI reported 0 MiB allocated and 0% GPU utilization. A restricted PyTorch probe saw
exactly one device, exposed as logical `cuda:0`, with compute capability 8.0, 124 SMs, and 96,974 MiB free.

All subsequent Amplin commands use this isolation contract:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-8be4c651-4058-83df-154b-291f1b86add8 \
TORCH_CUDA_ARCH_LIST=8.0 \
<command using logical cuda:0>
```

The UUID is resolved from PCI-order entry 2 at the start of a testing session and recorded with the results. Kernel
code continues to inspect the tensor's runtime device and must not contain a fixed index, UUID, PCI address, product
name, or observed SM count.

Decision:

- Run no new Amplin correctness, performance, or profiling work on physical GPU 0, GPU 1, or GPUs 3-7.
- Treat all measurements already recorded on GPU 1 as historical evidence, not directly mixed with new GPU 2 timing.
- Re-establish matched baselines and candidate measurements on GPU 2 before making further performance decisions.

Result: target-migration success. The exclusive GPU 2 process view and required `sm_80` properties were validated;
GPU 2 is the sole Amplin test target going forward.

### 2026-07-24 — Failure: small-batch benchmark help invoked without repository path

Revision before experiment: `b5e8c3a6`

While adding common batch sizes 1, 2, 4, and 8, the first read-only benchmark help check was launched as:

```text
python scripts/benchmark_amplin_model_shapes.py --help
```

It failed during Python import with:

```text
ModuleNotFoundError: No module named 'gptqmodel'
```

The command did not load an Amplin extension, execute CUDA, or produce performance data. This reproduces the
repository-path requirement already observed during the initial benchmark work.

Decision:

- Preserve the failure as a pre-launch environment checkpoint.
- Run repository scripts with `PYTHONPATH=.` from the repository root.
- Do not attribute this setup failure to the kernel or GPU 2.

Result: invocation failure. No correctness or performance conclusion was drawn.

### 2026-07-24 — Success: restrict all Amplin GPU work to PCI-order GPUs 3-6

Revision before experiment: `fe980432`

The user superseded the exclusive GPU 2 assignment after the small-batch correctness and smoke commands completed.
All subsequent Amplin GPU tests, benchmarks, builds, and profiler runs are restricted to physical GPUs 3-6 in the
live PCI-bus order. The allowed inventory was re-probed and resolved to:

| PCI-order index | UUID | PCI bus | Device | Compute capability | SMs |
| ---: | --- | --- | --- | ---: | ---: |
| 3 | `GPU-471ecdd7-a171-4d5c-d61f-a1802dc76e4c` | `00000000:69:00.0` | NVIDIA PG506-230 | 8.0 | 124 |
| 4 | `GPU-14ab23f1-a785-e9df-bbb5-215547154e3c` | `00000000:A0:00.0` | NVIDIA PG506-230 | 8.0 | 124 |
| 5 | `GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473` | `00000000:A5:00.0` | NVIDIA PG506-230 | 8.0 | 124 |
| 6 | `GPU-737e2423-874a-23a4-1126-dfbe3e77c294` | `00000000:DE:00.0` | NVIDIA PG506-230 | 8.0 | 124 |

A restricted PyTorch probe exposed exactly these four devices and verified `sm_80` plus 124 SMs on each. Free
memory at capture was approximately 20,620, 14,984, 472, and 15,562 MiB respectively, so the pool was not clean
enough for gate-quality performance timing. Correctness work may use an allowed device with sufficient free memory;
performance work waits for an uncontended allowed device.

Single-GPU commands select one resolved UUID from the allowed pool:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<resolved UUID for physical GPU 3, 4, 5, or 6> \
TORCH_CUDA_ARCH_LIST=8.0 \
<command using logical cuda:0>
```

Decision:

- Run no new Amplin GPU command on physical GPUs 0-2 or GPU 7.
- Resolve the live PCI-order inventory to UUID before every testing session; do not encode the allowed inventory
  indices as kernel capability checks.
- Record the chosen physical index, UUID, occupancy, and logical-device mapping with every result.
- Retain the completed GPU 2 small-batch smoke only as historical pre-directive evidence; repeat validation on an
  allowed GPU before publishing the implementation success.

Result: target-pool migration success. Physical GPUs 3-6 are the exclusive Amplin GPU test pool going forward.

### 2026-07-24 — Success: add common batches 1, 2, 4, and 8

Revision before experiment: `554a2d40`, with the batch-coverage implementation in the working tree.

The real-model correctness suite previously exercised flattened input-row counts 1, 4, and 16. It now covers
`M={1,2,4,8,16}` across all four extracted Qwen3-8B shapes and all ten extracted Laguna S 2.1 shapes, including
N=48 and N=72 tails, in FP16 and BF16. Prefix-shape preservation is independently parameterized at flattened
batches 1, 2, 4, and 8.

The model-shape benchmark default is now `--m-values 1,2,4,8,16`. When `--m32-sweep` receives M below 32, it
retains canonical `amplin_raw` alongside `marlin_raw`; it does not omit Amplin or mislabel the M32 Tensor Core
schedule as legal. M32 and later schedules retain their existing divisibility contracts.

Static validation passed:

```text
ruff check tests/kernels/test_amplin.py scripts/benchmark_amplin_model_shapes.py
git diff --check
python -m py_compile tests/kernels/test_amplin.py scripts/benchmark_amplin_model_shapes.py
```

The focused GPU correctness command used only allowed physical GPU 3, resolved to UUID
`GPU-471ecdd7-a171-4d5c-d61f-a1802dc76e4c` and exposed as logical `cuda:0`:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-471ecdd7-a171-4d5c-d61f-a1802dc76e4c \
TORCH_CUDA_ARCH_LIST=8.0 \
PYTHONPATH=. \
pytest -q tests/kernels/test_amplin.py \
  -k 'real_model_shapes_and_batches_match_fp32_dequant_reference or preserves_batched_input_prefix_shape'
```

Result: 18 passed and 40 deselected in 18.87 seconds. The 14 real-shape cases each exercised
`M={1,2,4,8,16}` in both dtypes; the remaining four cases checked prefix shapes for flattened batches 1, 2, 4,
and 8.

A one-round functional smoke on the same allowed GPU selected exact Qwen K4096xN1024 and Laguna K1024xN3072
shapes, both dtypes, and `M={1,2,4,8}`. It emitted exactly 32 result rows: `amplin_raw` and `marlin_raw` for all
16 shape/dtype/M cases. The device carried a large unrelated allocation, and the run used only two warmups and
three timed iterations, so its latency values are explicitly excluded from performance claims.

Smoke artifact:

```text
/tmp/amplin-small-batch-smoke-gpu3.json
size 73874 bytes
sha256 179c018faa4e53e269f0acd2e836379129f7fedc63bae10a5c04924e4ec82166
```

Decision:

- Make batches 1, 2, 4, and 8 mandatory in common Amplin correctness and default real-shape benchmark coverage.
- Retain batch 16 as the bridge between common decode batches and M32 Tensor Core schedules.
- Use canonical Amplin for M below 32 and preserve explicit path names in result tables.
- Run a five-round small-batch performance gate only when an allowed physical GPU 3-6 is uncontended.

Result: small-batch coverage success. Common decode batches now have real Qwen/Laguna FP16/BF16 correctness and
matched Amplin/Marlin benchmark-row coverage on an allowed GPU.

### 2026-07-24 — Failure: co-resident GPU 6 invalidates the common-batch timing gate

Revision: `7f22b7a6`

No physical GPU in the allowed 3-6 pool was memory-clean. Physical GPU 6 was selected as the least-active usable
device and resolved to UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`. It had approximately 80,469 MiB
allocated and 16,989 MiB free before launch. A ten-second sample was mostly at 0% SM utilization but included an
11% external spike, so the run was allowed to test whether five-round medians and ranges could reject contamination.

The command covered every extracted Qwen3-8B and Laguna S 2.1 shape in both dtypes:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 \
PYTHONPATH=. \
python scripts/benchmark_amplin_model_shapes.py \
  --model all --dtype both --m-values 1,2,4,8,16 \
  --warmup 50 --iters 200 --rounds 5 \
  --json-out /tmp/amplin-common-batch-all-shapes-gpu6-gate.json
```

The artifact is structurally complete:

```text
paired Amplin/Marlin cases:       120
Marlin-illegal N=48/N=72 cases:    20
samples per timed path:          1,000
numerical/finite checks:         passed
```

The performance gate failed stability validation. Of the 240 paired Amplin and Marlin path records, 113 had a
five-round batch-event spread greater than 10% of their median. The worst spread was 310.6%; several Amplin rows
contained one or more external-delay rounds between roughly 2x and 5x their stable samples. The nominal median
comparison produced 31 Amplin wins and 89 Marlin wins, but those counts and all latency values are rejected because
the two paths were not exposed to matched compute availability.

Artifact:

```text
/tmp/amplin-common-batch-all-shapes-gpu6-gate.json
size 644205 bytes
sha256 9f3e4d0c397dd4b77af524e9f4e623a5a702541be9241a1ff8a32c6237225d6a
```

Decision:

- Reject every latency, speedup, and crossover count from this run.
- Retain only the completed numerical validation and artifact as evidence that memory residency without sustained
  high utilization is still insufficient for microsecond-scale kernel comparisons.
- Retry the same gate only when a physical GPU in the allowed 3-6 pool has no competing allocation or activity.

Result: performance-gate failure due to external GPU contention; no Amplin-versus-Marlin claim is published.

### 2026-07-24 — Success: narrow the exclusive Amplin GPU pool to PCI-order GPUs 4-6

Revision before experiment: `7d783342`

The user excluded physical GPU 3 and restricted every subsequent Amplin GPU test, build, benchmark, and profiler
run to physical GPUs 4-6 in live PCI-bus order. The inventory was re-probed and resolved to:

| PCI-order index | UUID | PCI bus | Device | Compute capability | SMs |
| ---: | --- | --- | --- | ---: | ---: |
| 4 | `GPU-14ab23f1-a785-e9df-bbb5-215547154e3c` | `00000000:A0:00.0` | NVIDIA PG506-230 | 8.0 | 124 |
| 5 | `GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473` | `00000000:A5:00.0` | NVIDIA PG506-230 | 8.0 | 124 |
| 6 | `GPU-737e2423-874a-23a4-1126-dfbe3e77c294` | `00000000:DE:00.0` | NVIDIA PG506-230 | 8.0 | 124 |

At capture, all three allowed devices reported 0 MiB allocated, 0% utilization, and a 210 MHz idle SM clock.
A restricted PyTorch probe exposed exactly the three resolved UUIDs and reported compute capability 8.0, 124 SMs,
and 96,974 MiB free on each.

Decision:

- Run no new Amplin GPU command on physical GPUs 0-3 or GPU 7.
- Select one clean device from physical GPUs 4-6 by resolved UUID for single-GPU correctness and performance work.
- Preserve previous GPU 3 and GPU 6 records as historical provenance; only post-directive measurements on the
  narrowed pool may support new claims.

Result: target-pool narrowing success. Physical GPUs 4-6 are the exclusive Amplin GPU pool.

### 2026-07-24 — Success: clean common-batch Amplin versus Marlin gate

Revision: `87dc147e`

Physical GPU 6 was selected from the exclusive 4-6 pool and resolved to UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`. Before launch it reported 0 MiB allocated, 0% utilization, a 210 MHz
idle SM clock, and 97,458 MiB free. Memory returned to 0 MiB after the run.

The gate compared canonical row-independent `amplin_raw` with `marlin_raw`. M below 32 cannot execute the Amplin
M32 Tensor Core schedule, so this result measures the legal small-M Amplin kernel rather than padding or
mislabeling M32. The command was:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 \
PYTHONPATH=. \
python scripts/benchmark_amplin_model_shapes.py \
  --model all --dtype both --m-values 1,2,4,8,16 \
  --warmup 50 --iters 200 --rounds 5 \
  --json-out /tmp/amplin-common-batch-all-shapes-gpu6-clean-gate.json
```

The primary statistic is the median of five batched CUDA-event rounds, each containing 200 timed launches after
50 warmups. Every path has 1,000 timed samples. All finite/output/error checks against independent FP32 dequant
passed. The 12 Marlin-legal shapes produce 120 paired shape/dtype/M cases; Laguna N=48 and N=72 add 20
Amplin-only cases and are excluded from the comparison.

The table reports `Marlin time / Amplin time` as FP16/BF16. Values above 1 favor Amplin. A `~` suffix marks one
of seven cases whose complete five-round ranges overlap; no robust crossover claim is made for those cells.

```text
model/role                   KxN          B1 FP16/BF16   B2 FP16/BF16   B4 FP16/BF16   B8 FP16/BF16   B16 FP16/BF16
Qwen KV                      4096x1024    2.773/3.019    1.802/1.919    1.496/1.440    1.029/0.994~   0.651/0.607
Qwen Q/O                     4096x4096    1.599/1.554    1.011/1.078~   0.651/0.614    0.365/0.351    0.193/0.190
Qwen MLP-up                  4096x12288   0.841/0.835    0.451/0.437    0.249/0.239    0.128/0.124    0.066/0.068
Qwen MLP-down               12288x4096    0.553/0.548    0.362/0.388    0.220/0.227    0.125/0.124    0.070/0.074
Laguna expert-down           1024x3072    3.337/3.208    2.995/2.949    2.129/2.111    1.455/1.393    0.894/0.857
Laguna KV/expert-up          3072x1024    2.319/2.330    2.130/2.122    1.994/1.836    1.193/1.250    0.784/0.762
Laguna Q-proj-6144           3072x6144    1.363/1.498    0.915~/0.971~  0.585/0.568    0.321/0.308    0.168/0.154
Laguna Q-proj-9216           3072x9216    1.193/1.138    0.836/0.677    0.423/0.395    0.229/0.209    0.114/0.109
Laguna dense-up              3072x12288   0.952~/0.913   0.580/0.557    0.328/0.316    0.198/0.159    0.100/0.082
Laguna O-proj-6144           6144x3072    1.430/1.379    0.838/0.811    0.537/0.515    0.357/0.296    0.191/0.164
Laguna O-proj-9216           9216x3072    1.013~/0.953~  0.587/0.539    0.359/0.342    0.215/0.200    0.116/0.107
Laguna dense-down           12288x3072    0.893/0.790    0.447/0.459    0.323/0.285    0.161/0.157    0.092/0.093
```

Aggregate outcomes across both dtypes:

```text
batch   nominal Amplin wins   nominal Marlin wins   robust A/M/noise   median ratio   full ratio range
1                    15/24                    9/24             14/7/3          1.278x        0.548-3.337x
2                     8/24                   16/24             7/14/3          0.823x        0.362-2.995x
4                     6/24                   18/24             6/18/0          0.469x        0.220-2.129x
8                     5/24                   19/24             5/18/1          0.262x        0.124-1.455x
16                    0/24                   24/24             0/24/0          0.135x        0.066-0.894x
all                  34/120                  86/120            32/81/7          0.562x        0.066-3.337x
```

The clean result establishes a strong shape-dependent crossover:

- Batch 1 is the only regime where Amplin wins a majority: 14 robust wins, seven robust losses, and three noisy
  cases. It reaches 3.337x on Laguna K1024xN3072 and 3.019x on Qwen K4096xN1024 BF16.
- At batch 2 the median moves below parity. Amplin retains seven robust wins, concentrated in low/moderate-N
  shapes, while Marlin has 14 robust wins.
- Amplin retains six robust wins at batch 4 and five at batch 8, but Marlin wins the majority and the aggregate
  median falls to 0.469x and 0.262x.
- Marlin wins all 24 batch-16 cases with disjoint ranges. Canonical Amplin is only 0.066-0.894x Marlin there.
- A selector must use M, K, N, dtype, and measured range separation. Neither kernel is globally optimal at common
  decode batches.

Artifact:

```text
/tmp/amplin-common-batch-all-shapes-gpu6-clean-gate.json
size 644291 bytes
sha256 69466047f23c99fcf943fbf48977c81dca96e041c40f30f110f8bece5bc8afa1
```

Decision:

- Retain canonical Amplin as a competitive batch-1 path for the measured shape classes.
- Do not route batch 2 solely by M; the crossover is shape-specific.
- Prefer Marlin for every measured batch-16 class and for most batch-4/batch-8 classes.
- Keep the seven range-overlap outcomes out of selector training until independently repeated.
- Use these clean GPU-6 results, not the rejected co-resident artifact, as the common-batch baseline.

Result: common-batch gate success. Canonical Amplin wins the majority at batch 1, has targeted crossovers through
batch 8, and is categorically behind Marlin at batch 16.

### 2026-07-24 — Success: profile the large-projection batch-1 deficit

Revision: `ad88f452`

The optimization target moved from Amplin's small-shape batch-1 wins to the large projections where canonical
Amplin trails Marlin. The primary shapes are Qwen3-8B MLP-up K4096xN12288, Qwen3-8B MLP-down
K12288xN4096, Laguna S 2.1 dense-up K3072xN12288, and Laguna S 2.1 dense-down K12288xN3072. The
small-shape K4096xN1024 and K1024xN3072 paths remain unchanged controls.

Physical GPU 6 was selected from the exclusive PCI-order GPU 4-6 pool and resolved to UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, PCI bus `00000000:DE:00.0`. Before capture it reported 0 MiB
allocated, 0% utilization, and a 210 MHz idle SM clock. PyTorch reported NVIDIA PG506-230, compute capability
8.0, 124 SMs, 102,191,202,304 bytes total memory, Torch 2.13.0+cu130, and CUDA runtime 13.0.

Nsight Systems 2024.6.2 captured 50 warmed raw launches per path after 50 warmups:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=8.0 \
nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/amplin-large-k-m1-gpu6 -- \
  python scripts/profile_amplin_vs_marlin.py \
    --device cuda:0 --path both --dtype fp16 \
    --m 1 --k 12288 --n 4096 --warmup 50 --launches 50 \
    --cuda-profiler-api
```

The same command with `--k 4096 --n 12288` produced the N-heavy capture. `nsys stats --report
cuda_gpu_kern_sum,nvtx_gpu_proj_sum,cuda_kern_exec_sum` reported:

```text
shape          path     launches   median kernel   min-max kernel
M1 K12288 N4096
               Amplin         50        47.280 us   46.560-66.113 us
               Marlin         50        24.352 us   23.968-27.392 us
M1 K4096 N12288
               Amplin         50        31.456 us   31.360-34.688 us
               Marlin         50        22.112 us   21.984-25.984 us
```

Each raw path launches exactly one kernel per linear. The deficit is therefore inside the device schedule rather
than recoverable launch overhead. Both Amplin and Marlin passed the independent FP32-dequant reference check.

Artifacts:

```text
/tmp/amplin-large-k-m1-gpu6.nsys-rep
size 380395 bytes
sha256 bc6b0b3458be714c871057d79584d21bbdf269c7dd30416067f1166cd6e6f3b9

/tmp/amplin-large-n-m1-gpu6.nsys-rep
size 385149 bytes
sha256 26a3cbde6b4b36d0fa2bc0c19ce98ba7f7592f33043056feb723d9edfda678f7
```

Nsight Compute 2025.3.1 then profiled the canonical Amplin kernels with `--profile-from-start off`, one launch,
and targeted `SpeedOfLight`, `LaunchStats`, `Occupancy`, `WarpStateStats`, and `SchedulerStats` sections. The
K-heavy command was:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=8.0 \
ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section WarpStateStats --section SchedulerStats \
  --kernel-name regex:amplin_gptq_w4_group128 --launch-count 1 -- \
  python scripts/profile_amplin_vs_marlin.py \
    --device cuda:0 --path amplin --dtype fp16 \
    --m 1 --k 12288 --n 4096 --warmup 50 --launches 1 \
    --cuda-profiler-api
```

The profiler-replay duration is not the gate latency and is reported only with the counters collected in that
same replay:

```text
metric                                      K12288xN4096    K4096xN12288
Compute (SM) throughput                           31.99%          59.62%
Memory throughput                                 36.33%          68.61%
DRAM throughput                                   16.73%          30.83%
L1/TEX throughput                                 38.31%          80.17%
```

For K12288xN4096, the 256-block x 256-thread grid uses 32 registers per thread and 576 bytes static shared
memory. Theoretical occupancy is 100%, but the grid supplies only 0.26 full waves and achieves 25.09% occupancy
or 16.06 active warps per SM. Schedulers issue only 0.31 warps/cycle; 64.5% of the 13.2 cycles between issued
instructions are long-scoreboard stalls waiting on L1TEX dependencies. The existing kernel assigns one N16 tile
to one CTA, so only 256 CTAs cover the 124-SM device and each warp serially processes six group pairs.

The N-heavy K4096xN12288 V0 path is already a balanced, much larger 768-block grid. It reaches 68.61% memory
throughput and 59.62% compute throughput. It remains a secondary tuning target; the first prototype targets the
clearer K-heavy underfill and dependency bottleneck.

The first large-shape mega-kernel prototype will be a one-launch cooperative split-K schedule for batch 1 and
K12288:

```text
phase   ownership                         output/lifetime                 synchronization
A       split-K CTA x N16 tile            FP32 partial[split][N]          CTA-local reduction, then grid sync
B       split 0 CTA x N16 tile            FP16/BF16 final output[N]       read all split partials after grid sync
```

A two-way split produces 512 CTAs, versus 256 today, while halving group-pair work per CTA. The current launch
profile reports capacity for eight 256-thread blocks per SM, or 992 resident CTAs across 124 SMs, so 512 CTAs fit
the cooperative residency bound. The scratch requirement is `2*N*sizeof(float)`, only 32 KiB at N4096 and 24 KiB
at N3072. Runtime code must query cooperative-launch support and active-block capacity on the selected device,
reject CUDA graph capture for the cooperative launch, and preserve the canonical non-cooperative fallback.

Decision:

- Build the first prototype only for `M=1`, `K=12288`, and N divisible by 16, covering the Qwen and Laguna
  dense-down projections.
- Keep the existing V0 K4096 path and every small-shape path unchanged.
- Treat cooperative split-K as a hypothesis, not a selector win; retain it only if correctness passes both dtypes
  and warmed timing improves the targeted real shapes without regressing the controls.
- Follow with multi-row weight reuse only after the batch-1 K-heavy schedule is measured.

Result: large-projection profiling success. The first mega-kernel experiment is grounded in measured grid
underfill and long-scoreboard latency rather than an assumed launch-fusion benefit.

### 2026-07-24 — Failure: cooperative two-way split-K does not improve large down projections

Revision before experiment: `36f272c4`, with the experimental source in the working tree.

The first large-shape mega-kernel prototype split the 48 K-group pairs of K12288 into two cooperative CTA
partitions. Phase A launched one 256-thread CTA per split and N16 tile, wrote FP32 partials, and executed an
in-grid barrier. Phase B used the split-0 CTAs to add the two partials and write FP16/BF16 output. This changed the
Qwen K12288xN4096 grid from 256 to 512 CTAs and the Laguna K12288xN3072 grid from 192 to 384 CTAs without a
second kernel launch.

The first compile attempt failed before launching a GPU kernel because the `cudaLaunchCooperativeKernel` argument
array received a pointer to a `const int` local for `blocks_n`. Making the by-value launch argument mutable fixed
the host-side type error. The rebuilt sm_80 extension then passed all four focused correctness cases on physical
GPU 6: N3072/N4096 x FP16/BF16, versus independent FP32 dequant, including repeated-output determinism.

The clean correctness command was:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
pytest -q tests/kernels/test_amplin.py -k splitk_large_down_projection
```

Result: 4 passed and 58 deselected in 41.46 seconds, including a 34-second JIT rebuild.

The five-round CUDA-event gate used 50 warmups and 200 timed launches per round:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
python scripts/benchmark_amplin_model_shapes.py \
  --model all --shape 12288x4096 --shape 12288x3072 \
  --dtype both --m-values 1 --split-k \
  --warmup 50 --iters 200 --rounds 5 \
  --json-out /tmp/amplin-splitk-m1-large-down-gpu6-gate.json
```

The table compares the median batched CUDA-event time. `canonical/candidate` below 1 means split-K is slower:

```text
model/shape                 dtype   canonical   split-K   canonical/candidate   complete round ranges
Qwen MLP-down 12288x4096    FP16      47.903 us  49.485 us        0.968x          overlap
Qwen MLP-down 12288x4096    BF16      48.051 us  51.338 us        0.936x          overlap
Laguna dense-down 12288x3072 FP16     28.396 us  36.628 us        0.775x          disjoint
Laguna dense-down 12288x3072 BF16     28.723 us  36.716 us        0.782x          disjoint
```

The Laguna controls are extremely stable: canonical ranges are 28.380-28.416 us FP16 and 28.723-28.739 us BF16,
while split-K ranges are 36.603-36.746 us and 36.521-37.627 us. The candidate is therefore 29.0% and 27.8% slower
with disjoint full ranges. The Qwen ranges overlap, but neither dtype shows a nominal win. Marlin round ranges in
this run are much wider and slower than the prior clean common-batch gate, so Marlin values from this artifact are
excluded; the paired canonical Amplin controls are sufficient to reject the candidate.

Artifact:

```text
/tmp/amplin-splitk-m1-large-down-gpu6-gate.json
size 31941 bytes
sha256 e6f3611fb209d3a39715b3a04a5b03e2fbf673df5fc471b28f568368f1c48e5f
```

A targeted 13-pass Nsight Compute capture on Laguna FP16 explains why extra CTAs were insufficient:

```text
grid / block                         384 CTAs / 256 threads
registers / static shared            39 per thread / 576 bytes
theoretical / achieved occupancy     75.00% / 37.94%
active warps per SM                  24.28
issued warps per scheduler/cycle     0.38
compute / memory throughput          35.55% / 40.78%
long-scoreboard stall                9.3 of 16.43 cycles, 56.4%
```

The larger grid raised active-warps coverage, but the cooperative kernel added seven registers per thread, reduced
theoretical occupancy from 100% to 75%, retained dominant L1TEX dependency stalls, and paid an FP32 partial
write/read plus a grid barrier. The added residency did not offset those costs.

Decision:

- Reject cooperative two-way split-K for M1 K12288.
- Remove the experimental operator, test, and benchmark switch; preserve only this failure record and artifact.
- Keep the canonical Amplin and all small-shape paths unchanged.
- Test a barrier-free 16-warp K12288 CTA next. It can expose twice as many resident warps per output tile, reusing
  the proven 512-thread K4096 V0 structure without partial scratch or grid synchronization.

Result: split-K mega-kernel failure. Correctness passed, but the schedule provides no Qwen win and regresses both
stable Laguna dtype cases by approximately 28-29%; no experimental source is retained.

### 2026-07-24 — Success: barrier-free 16-warp K12288 schedule

Revision before experiment: `e72ebdba`, with the candidate implementation in the working tree.

The next large-down candidate removed cooperative launch, FP32 scratch, and grid synchronization. One 512-thread
CTA owns each N16 tile. Its 16 warps divide the 48 K-group pairs evenly, so every warp evaluates three group pairs
before the existing shared-memory cross-warp reduction. The canonical K12288 path uses eight warps and evaluates
six group pairs per warp. Both paths read the same canonical GPTQ int32 qweight and group-128 scale layout.

The research operator is explicitly named `gemv_k12288_wide`; it does not yet change automatic `gemv` routing.
Its contract is M1, K12288, N divisible by 16, FP16/BF16, symmetric W4 group-128, and exact sm_80.

Physical GPU 6 was selected by UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`. At the validation session GPUs
4-6 each carried the same 483 MiB driver/system footprint. A five-second `nvidia-smi dmon` sample reported 0% SM
and 0% memory utilization on all three devices with no compute spikes. Only physical GPU 6 was exposed to the
test process.

Static validation passed:

```text
ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py \
  scripts/benchmark_amplin_model_shapes.py
python -m py_compile gptqmodel/utils/amplin.py tests/kernels/test_amplin.py \
  scripts/benchmark_amplin_model_shapes.py
git diff --check
```

The focused CUDA correctness command was:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
pytest -q tests/kernels/test_amplin.py \
  -k k12288_wide_large_down_projection
```

Result: 4 passed and 58 deselected in 37.19 seconds, including a 31-second sm_80 JIT rebuild. Qwen
K12288xN4096 and Laguna K12288xN3072 passed FP16/BF16 independent FP32-dequant comparison, finite checks, shape
and dtype checks, and exact repeated-output determinism.

The five-round gate used 50 warmups and 200 timed launches per round:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
python scripts/benchmark_amplin_model_shapes.py \
  --model all --shape 12288x4096 --shape 12288x3072 \
  --dtype both --m-values 1 --k12288-wide \
  --warmup 50 --iters 200 --rounds 5 \
  --json-out /tmp/amplin-k12288-wide-m1-large-down-gpu6-gate.json
```

The primary statistic is the median batched CUDA-event time:

```text
model/shape                  dtype   canonical   16-warp   canonical/16-warp   full ranges disjoint
Qwen MLP-down 12288x4096     FP16     47.278 us  40.714 us        1.161x                yes
Qwen MLP-down 12288x4096     BF16     45.711 us  41.544 us        1.100x                yes
Laguna dense-down 12288x3072 FP16     28.155 us  25.498 us        1.104x                yes
Laguna dense-down 12288x3072 BF16     28.575 us  25.933 us        1.102x                yes
```

All four complete five-round ranges are disjoint. The narrowest separation is Qwen FP16: canonical
47.140-53.453 us versus 16-warp 40.499-46.966 us. The candidate retains exactly the canonical maximum error in
all four cases.

The candidate does not yet eliminate the Marlin deficit. Its nominal Marlin speed is 0.572x/0.557x for Qwen
FP16/BF16 and 0.908x/1.002x for Laguna. The Laguna Marlin ranges overlap the candidate and the Qwen candidate is
clearly slower, so no Amplin-over-Marlin win is claimed from this gate.

Artifact:

```text
/tmp/amplin-k12288-wide-m1-large-down-gpu6-gate.json
size 32019 bytes
sha256 8a1b565e3a9c65dc9310d4eeaf3c1cfc1cebb259baca68443c67dd37b920f998
```

A targeted 13-pass Nsight Compute capture used the exact Qwen FP16 K12288xN4096 shape. The table compares the
earlier canonical report with the new wide-CTA report:

```text
metric                                  canonical 8-warp    16-warp
grid x block                              256 x 256         256 x 512
registers per thread                              32                 32
static shared memory                         576 bytes         1.15 KiB
achieved occupancy                              25.09%            48.14%
active warps per SM                              16.06             30.81
issued warps per scheduler/cycle                  0.31              0.55
eligible warps per scheduler                      0.40              1.10
compute throughput                               31.99%            46.61%
memory throughput                                36.33%            53.49%
long-scoreboard share of issue interval          64.5%             53.2%
profiler-replay duration                         66.88 us          47.20 us
```

The wider CTA nearly doubles achieved occupancy without increasing register use, raises scheduler issue rate
77%, and reduces the relative long-scoreboard stall burden. The profiler-replay duration is not substituted for
the five-round CUDA-event gate, but its counters validate the intended mechanism.

Decision:

- Retain `gemv_k12288_wide` as an explicit measured control.
- Do not route it automatically until the full existing K12288 M1 correctness cases and current-stream behavior
  pass through the main operator contract.
- After routing, rerun the four large Qwen/Laguna MLP shapes at M=1/2/4/8/16 in both dtypes. Only M1 K12288
  should change; every other shape and batch is a fallback regression control.
- Continue optimizing because Qwen remains substantially behind Marlin even after this 10-16% Amplin gain.

Result: barrier-free K12288 schedule success. It robustly improves canonical Amplin on all four targeted
shape/dtype cases while preserving numerical behavior and avoiding the failed cooperative split-K costs.

### 2026-07-24 — Failure: external activity invalidates the routed large-MLP batch gate

Revision before experiment: `9db513a3`, with the narrow M1 K12288 main-path routing in the working tree.

The routed implementation passed 18 focused CUDA tests on physical GPU 6: all 14 extracted Qwen/Laguna shapes at
M=1/2/4/8/16 in both dtypes, plus four explicit-wide versus selected-main-path comparisons for K12288xN3072/N4096.
The selected M1 K12288 results are bit-exact with `gemv_k12288_wide`, including execution on a non-default CUDA
stream. Result: 18 passed and 44 deselected in 37.65 seconds, including a 30-second JIT rebuild.

The requested large-shape performance matrix then covered Qwen MLP-up/down and Laguna dense-up/down in both
dtypes and M=1/2/4/8/16:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 --shape 12288x4096 \
  --shape 3072x12288 --shape 12288x3072 \
  --dtype both --m-values 1,2,4,8,16 \
  --warmup 50 --iters 200 --rounds 5 \
  --json-out /tmp/amplin-large-mlp-routed-gpu6-gate.json
```

The artifact is structurally and numerically complete: 40 paired cases, 80 path records, and every finite/error
check against independent FP32 dequant passed. It is rejected for performance. During and after the run all
allowed GPUs 4-6 acquired approximately 3.8-4.0 GiB of memory and sustained 3-4% SM activity from a process not
visible in this container's `nvidia-smi pmon` or compute-app query. A five-second post-run sample confirmed the
activity on all three allowed devices.

Of 80 path records, 48 have a complete five-round spread greater than 10% of their median and 34 exceed 20%. The
worst spread is 82.3%. The instability affects both Amplin and Marlin and includes M1 routed rows, so no latency,
speedup, crossover, or fallback-regression value from this artifact is publishable.

Artifact:

```text
/tmp/amplin-large-mlp-routed-gpu6-gate.json
size 200456 bytes
sha256 ef79f9a6df0511cec152ba29ad64d198b10e35ac9bbde99136e2e4a1e1e90f70
```

Decision:

- Reject all timing values from this run.
- Retain the 18-test routed correctness result and the 40 paired numerical checks.
- Keep the earlier uncontended explicit-path gate as the only performance evidence for the 16-warp schedule.
- Retry the same full large-MLP matrix only when at least one physical GPU in the exclusive 4-6 pool returns to
  zero external utilization with stable allocation.

Result: routed performance-gate failure due to external GPU contention; no batch 1/2/4/8/16 performance claim is
added from this artifact.

### 2026-07-24 — Success: narrowly route M1 K12288 to the retained wide CTA

Revision before routing: `9db513a3`; contention record: `f9cf8138`.

The main canonical `gemv` launcher now selects the retained 16-warp kernel only when flattened M is 1, K is
12288, and N is divisible by 16. This covers exact Qwen K12288xN4096 and Laguna K12288xN3072 dense-down
projections. M=2/4/8/16, every other K, and N tails retain the prior general or K4096 V0 paths. The capability
guard remains the runtime-selected device's exact compute capability 8.0; no CUDA index implies support.

Validation on allowed physical GPU 6:

```text
CUDA_VISIBLE_DEVICES=GPU-737e2423-874a-23a4-1126-dfbe3e77c294 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
pytest -q tests/kernels/test_amplin.py \
  -k 'k12288_wide_large_down_projection or real_model_shapes_and_batches_match_fp32_dequant_reference'
```

Result: 18 passed and 44 deselected. The 14 real-shape tests cover M=1/2/4/8/16 and both dtypes. The four
wide-path tests prove that selected `gemv` is bit-exact with the retained explicit control for Qwen/Laguna,
FP16/BF16, including a non-default current CUDA stream.

Performance authority remains the uncontended explicit-path gate recorded above: four of four targeted cases beat
canonical Amplin by 1.100-1.161x with disjoint ranges. The later routed full-batch artifact is numerical evidence
only because external activity invalidated its timing.

Decision:

- Retain the narrow automatic route.
- Preserve `gemv_k12288_wide` as a forced research/control operator.
- Do not broaden the route to M>1; the existing row-independent schedule still reloads weights per activation row
  and needs a separate multi-row design.
- Retry the complete batch matrix on the exclusive GPU 4-6 pool when a device is uncontended.

Result: K12288 wide-CTA routing success. Large-down batch 1 receives the measured schedule while all unrelated
shape, batch, dtype, device, and CPU/non-target fallbacks remain unchanged.

### 2026-07-24 — Success: add a correctness-clean multi-row weight-reuse control

Revision before experiment: `26144673`, with the multi-row implementation in the working tree.

The next large-shape research operator, `gemv_multirow`, changes CTA ownership instead of the GPTQ storage
format. Each 512-thread CTA owns one N16 tile and either two or four activation rows. A thread loads each int32
W4 word and group scale once, then applies the dequantized codes to every owned row. M2 uses a two-row tile;
M4/M8/M16 use four-row tiles. The operator supports the large-MLP K classes 3072, 4096, and 12288 with N divisible
by 16, in FP16/BF16 on exact sm_80.

The reference row-independent path requests the full qweight and scale arrays M times. The multi-row schedule
requests them `M/2` times at M2 and `M/4` times at M4/M8/M16. It stays a one-launch fused dequant/dot/reduction
operator and requires no scratch, atomics, cooperative launch, or grid barrier.

Static validation passed:

```text
ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py \
  scripts/benchmark_amplin_model_shapes.py
python -m py_compile gptqmodel/utils/amplin.py tests/kernels/test_amplin.py \
  scripts/benchmark_amplin_model_shapes.py
git diff --check
```

Correctness used only allowed physical GPU 5, UUID
`GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473`:

```text
CUDA_VISIBLE_DEVICES=GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
pytest -q tests/kernels/test_amplin.py \
  -k multirow_large_mlp_projection
```

Result: 8 passed and 62 deselected in 6.63 seconds after the extension was cached. The eight test items cover four
exact Qwen/Laguna MLP shapes x two dtypes; each item exercises M=2/4/8/16, for 32 independent FP32-dequant
comparisons. All outputs are finite and within the established FP16/BF16 tolerances. Every call runs on a
non-default current CUDA stream.

The retained cubin has no stack or local-memory spill:

```text
specialization       registers/thread   static shared   local   stack
FP16/BF16 rows=2             32             2304 B        0       0
FP16/BF16 rows=4             40             4608 B        0       0
```

A five-round performance experiment was attempted on physical GPU 5 while the hidden external workload remained
active:

```text
CUDA_VISIBLE_DEVICES=GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473 \
TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH=. \
python scripts/benchmark_amplin_model_shapes.py \
  --model all \
  --shape 4096x12288 --shape 12288x4096 \
  --shape 3072x12288 --shape 12288x3072 \
  --dtype both --m-values 2,4,8,16 --multirow \
  --warmup 50 --iters 200 --rounds 5 \
  --json-out /tmp/amplin-multirow-large-mlp-gpu5-contended-gate.json
```

The artifact contains 32 cases x canonical Amplin, multi-row Amplin, and Marlin, for 96 path records. Numerical
checks pass. Timing is rejected because physical GPUs 4-6 still held approximately 4.5 GiB and 2-4% hidden
external activity. Thirteen path records exceed 10% five-round spread, five exceed 20%, and the maximum is 566%.

The contaminated nominal comparison is useful only for forming the next clean gate: multi-row has 18
range-separated wins, two range-separated losses, and 12 overlaps versus canonical Amplin. Wins concentrate at
M8/M16; small-M K12288 can lose because row fusion reduces CTA coverage before enough row tiles exist. These are
hypotheses, not publishable performance claims.

Artifact:

```text
/tmp/amplin-multirow-large-mlp-gpu5-contended-gate.json
size 241019 bytes
sha256 20781c912e235c400bc7cf88426316a044370e36e904f8e53e113731fe51fc9b
```

Decision:

- Retain `gemv_multirow` as an explicit research/control operator.
- Do not route it automatically from contaminated timing.
- Prioritize a clean M8/M16 gate, where CTA coverage and weight reuse are both favorable.
- Keep M2/M4 shape-specific and test a two-row tile for underfilled K12288 M8 if clean results confirm the
  current four-row occupancy tradeoff.

Result: multi-row prototype correctness success. The implementation, tests, resource bounds, and benchmark
switch are retained; performance selection waits for an uncontended allowed GPU.

### 2026-07-24 — Failure: scalar multi-row reuse does not close the large-shape Marlin deficit

Revision under test: `bdf7d60a`.

The allowed device pool changed to physical PCI-order GPUs 1 and 2. Live discovery resolved them without
assuming CUDA ordinals:

```text
physical  PCI bus       UUID                                          device          memory   CC   SMs
1         0000:2B:00.0  GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855   PG506-232       96 GiB   8.0  124
2         0000:64:00.0  GPU-8be4c651-4058-83df-154b-291f1b86add8   PG506-230       96 GiB   8.0  124
```

Both devices reported zero utilization and no material allocation before testing. PyTorch was
`2.13.0+cu130`, the CUDA runtime was 13.0, and the driver was 610.43.02. The exact multi-row correctness matrix
passed independently on both devices:

```text
CUDA_VISIBLE_DEVICES=<physical-GPU-UUID> TORCH_CUDA_ARCH_LIST=8.0 \
pytest -q tests/kernels/test_amplin.py -k multirow_large_mlp_projection
```

Each device passed eight parametrized items covering the four Qwen/Laguna large MLP shapes and two dtypes. Each
item loops over M=2/4/8/16 on a non-default stream, so each GPU completed 32 independent FP32-dequant comparisons.

The first timing launch used `python scripts/benchmark_amplin_model_shapes.py` and failed before CUDA
initialization with `ModuleNotFoundError: No module named 'gptqmodel'`. No measurement was produced. The corrected
repository module invocation was then used. GPU 1 and GPU 2 were first run concurrently and then repeated
sequentially to remove concurrent host/JIT work:

```text
CUDA_VISIBLE_DEVICES=<physical-GPU-UUID> TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 4096x12288 --shape 12288x4096 \
  --shape 3072x12288 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both --multirow \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-multirow-large-mlp-gpu<1-or-2>-sequential-r7.json
```

Both sequential artifacts contain 32 cases and 96 path records. All numerical comparisons pass. Against
canonical Amplin, each device independently classifies the scalar multi-row control as 24 range-separated wins,
four losses, and four overlaps. The median improvement reaches 1.172x on GPU 1 and 1.181x on GPU 2, but this only
reduces the cost of a scalar-FMA design whose latency continues to grow almost linearly with M.

No multi-row case beats Marlin with range separation on both devices. Thirty cases lose to Marlin with disjoint
ranges on both devices; two are mixed or overlap because Marlin timing was unstable. Marlin itself has more than
10% seven-round spread in 9/32 GPU-1 rows and 17/32 GPU-2 rows, with occasional cross-device outliers, so isolated
Marlin medians from these artifacts are not used as exact performance claims. The decisive result is structural
and repeatable in the Amplin measurements: row fusion reduces weight requests but still performs scalar FP32
FMAs for every activation row, so M=4/8/16 latency scales with row count while the production Tensor-Core path
does not.

Sequential artifacts:

```text
/tmp/amplin-multirow-large-mlp-gpu1-sequential-r7.json
size 251470 bytes
sha256 14cff70f506720e0747be9cbd1dafd19a32c865e2b18631e9cb62fa7d7bc4119

/tmp/amplin-multirow-large-mlp-gpu2-sequential-r7.json
size 251459 bytes
sha256 9c8ed9d2b5d614f4c746f8d5b80ba55f69e7159680a8a8a90a10dcdb4037af09
```

Decision:

- Keep `gemv_multirow` as an explicit scalar control and do not route it automatically.
- Stop tuning row count or CTA coverage on this scalar design; the remaining Marlin gap is not a small launch
  geometry problem.
- Prototype a packed Ampere Tensor-Core path for M=2/4/8/16. Pad only the A fragment to M16 in registers, retain
  real-row guarded stores, and use N16 warp ownership so the four exact large shapes expose enough independent
  work without a barrier, reduction scratch, or cooperative launch.
- Treat the packed MMA-lane weight layout as the candidate inference format rather than forcing the legacy
  canonical int32 layout into the hot kernel.

Result: clean cross-device correctness success and scalar multi-row performance failure. The next large-shape
prototype moves weight reuse into native Ampere MMA fragments.

### 2026-07-24 — Success: packed padded-M16 Tensor-Core path wins Laguna M2/M4

Revision before experiment: `de4ed213`, with the padded-M16 implementation in the working tree.

The new explicit `mma_lane_m16_n16_padded` operator changes both execution ownership and the hot weight layout:

- weights use the retained native Ampere lane format `[N64, K128, K16, N16, lane]`;
- one 32-thread CTA owns one N16 output tile;
- M=2/4/8 inputs are zero-padded only while forming the M16 A fragment in registers;
- real rows are guarded at the output store;
- every K16 step issues two `mma.sync.m16n8k16` instructions;
- no shared memory, CTA barrier, scratch, atomics, cooperative launch, or cross-CTA reduction is used.

The packed lane layout is the candidate inference format. It remains an int32 storage tensor, but its words are
ordered for direct Ampere B-fragment register ownership rather than canonical GPTQ serialization order.
Canonical save/load remains unchanged and the explicit packer performs the conversion outside the timed kernel.

The first focused run compiled JIT fingerprint `b4fba1060e23393b` for `sm_80`. Six of nine items passed, while
three long-K items produced large errors. Isolating one failed item passed, revealing a test-order race: packed
tensors produced on the default stream were consumed on a fresh non-default stream without an explicit dependency.
Adding `stream.wait_stream(torch.cuda.current_stream(device))` before the research launch fixed the test contract;
the kernel arithmetic did not change. The same dependency was added to the preceding multi-row stream test.

Corrected validation on physical GPUs 1 and 2:

```text
CUDA_VISIBLE_DEVICES=<physical-GPU-UUID> TORCH_CUDA_ARCH_LIST=8.0 \
pytest -q tests/kernels/test_amplin.py \
  -k 'padded_m16_large_mlp_projection or hmma_gemm_rejects_inputs_outside_contract'
```

Each GPU reports 9 passed and 69 deselected. The eight numerical items cover four exact Qwen/Laguna shapes x two
dtypes, and each loops over M=2/4/8/16 for 32 FP32-dequant comparisons. The ninth item validates the M and N
contract rejections. Both GPUs pass the non-default-stream matrix after the explicit producer-consumer dependency.

Retained cubin resources:

```text
dtype  threads/CTA  registers/thread  shared  local  stack
FP16       32              70            0      0      0
BF16       32              72            0      0      0
```

The primary performance run used fully empty physical GPU 2. Physical GPU 1 provided the cross-device repeat; it
had 0% utilization and no visible compute process, although `nvidia-smi` reported an unexplained 959 MiB
allocation. Both runs used 50 warmups, 200 launches per round, seven rounds, both dtypes, and all four exact large
MLP shapes at M=2/4/8/16:

```text
CUDA_VISIBLE_DEVICES=<physical-GPU-UUID> TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 4096x12288 --shape 12288x4096 \
  --shape 3072x12288 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both --multirow --padded-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-padded-m16-large-mlp-gpu<1-or-2>-r7.json
```

All 64 candidate numerical comparisons pass. Relative to both canonical Amplin and the scalar multi-row control,
the padded path has 30 range-separated wins and two losses on both devices. The four cross-device Marlin wins are:

```text
shape                  dtype  M  GPU1 padded range us  GPU1 Marlin range us  speedup  GPU2 padded range us  GPU2 Marlin range us  speedup
Laguna 3072x12288      FP16   2  18.652-18.739         22.569-26.455          1.224x   18.668-18.770         21.765-28.877          1.179x
Laguna 3072x12288      FP16   4  19.287-19.338         22.190-27.274          1.174x   19.011-19.052         21.591-24.279          1.147x
Laguna 3072x12288      BF16   2  20.362-22.216         22.390-26.711          1.106x   20.337-20.352         22.579-25.272          1.204x
Laguna 3072x12288      BF16   4  20.731-20.859         21.985-35.154          1.135x   20.603-20.649         22.420-23.327          1.105x
```

Every listed candidate range is entirely below its same-device Marlin range. The other 28 cells lose to Marlin
with disjoint ranges on both devices. In particular, K12288 candidate latency is approximately 55-60 us at M2
and 77-85 us at M16, versus stable Marlin medians near 22-27 us. The one-warp CTA exposes only 192 or 256 warps
across the K12288 down-projection grids and leaves each warp with a 96-group serial dependency chain.

Artifacts:

```text
/tmp/amplin-padded-m16-large-mlp-gpu1-r7.json
size 342974 bytes
sha256 ac2a73674e5f86c556268bd2c4b2fcb2ae588c13e842fdcb71af89badb811dcf

/tmp/amplin-padded-m16-large-mlp-gpu2-r7.json
size 342994 bytes
sha256 2e10ac187280745d9c11881146b1c535618e5e84d7dee55bd8f2044458436f49
```

Decision:

- Retain `mma_lane_m16_n16_padded` as an explicit packed-layout research operator.
- Do not route canonical `gemv`: production selection needs a backend-owned packed weight lifecycle, and only the
  exact Laguna K3072xN12288 M2/M4 cells currently beat Marlin.
- Preserve the scalar multi-row operator only as a control.
- Next, split K12288 across four warps inside one ordinary CTA. Each warp will own disjoint K groups and the CTA
  will reduce distributed FP32 MMA fragments through a small shared buffer. This increases resident/eligible
  warps without rereading weights or requiring a grid barrier.

Result: the first large-MLP Amplin Tensor-Core schedule beats Marlin on real Qwen/Laguna-era shape classes. The
win is narrow but cross-device and range-separated, and it validates redesigning the packed int32 layout around
Ampere fragment ownership.

### 2026-07-24 — Partial success: K12288 split-K4 halves Amplin latency but trails Marlin

Revision before experiment: `51c28108`, with the split-K4 implementation in the working tree.

The explicit `mma_lane_m16_n16_splitk4` operator keeps the packed padded-M16 contract but assigns four warps to
each N16 output tile. Warp `w` accumulates K groups `w, w+4, ...`; all four warps write their two distributed
FP32 accumulator fragments to a 4 KiB shared buffer, cross one CTA barrier, and warp 0 performs the final fragment
sum and guarded real-row store. Total weight and activation traffic is unchanged from the one-warp path. The
candidate uses one ordinary launch with no global scratch, atomics, cooperative launch, or grid barrier.

The `sm_80` JIT fingerprint is `aaad7b564c34be90`. Static resources are identical for FP16 and BF16:

```text
threads/CTA  registers/thread  static shared  local  stack
128                 48             4096 B       0      0
```

The focused padded-M16 matrix passes on physical GPU 2 after the rebuild: 9 passed and 69 deselected. The two
K12288 shapes now execute both one-warp and split-K4 controls for FP16/BF16 at M=2/4/8/16, adding 16 split-K4
FP32-dequant comparisons. The boundary test also proves that split-K4 rejects K values other than 12288.

The seven-round gate then ran the two exact K12288 down projections on physical GPUs 1 and 2:

```text
CUDA_VISIBLE_DEVICES=<physical-GPU-UUID> TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both \
  --padded-m16 --splitk4-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-splitk4-m16-k12288-gpu<1-or-2>-r7.json
```

Split-K4 is stable and materially improves the one-warp padded path on both devices:

```text
shape/dtype/M                 GPU1 one-warp/split  GPU2 one-warp/split
Qwen FP16 M2                        2.094x                 2.082x
Qwen BF16 M2                        2.076x                 2.080x
Laguna FP16 M2                      2.105x                 2.116x
Laguna BF16 M2                      2.094x                 2.096x
Qwen FP16 M4                        2.056x                 2.067x
Qwen BF16 M4                        2.076x                 2.080x
Laguna FP16 M4                      2.072x                 2.084x
Laguna BF16 M4                      2.083x                 2.089x
```

The candidate is approximately 26-29 us at M2/M4, versus 55-61 us for one-warp padded M16. It does not establish
a Marlin win: eight of 16 cells lose with disjoint full ranges on both devices, and the remaining eight are mixed
or overlap because GPU-2 Marlin timing is noisy. Stable GPU-1 M2/M4 comparisons place split-K4 approximately
14-19% behind Marlin. No automatic route is justified.

Benchmark artifacts:

```text
/tmp/amplin-splitk4-m16-k12288-gpu1-r7.json
size 173046 bytes
sha256 53c66e0de06d59b158b91bae45bf4fe55e710d72129d969ad7c244688b6a7156

/tmp/amplin-splitk4-m16-k12288-gpu2-r7.json
size 173060 bytes
sha256 5f6c88e86bfef87393afe9514376c3019fa8ab1872e0966dda7a4808b442089c
```

A targeted 13-pass Nsight Compute capture profiled Qwen FP16 M2 K12288xN4096 on physical GPU 2:

```text
CUDA_VISIBLE_DEVICES=GPU-8be4c651-4058-83df-154b-291f1b86add8 \
TORCH_CUDA_ARCH_LIST=8.0 \
ncu --target-processes all --kernel-name regex:splitk4 --launch-count 1 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  -o /tmp/amplin-splitk4-k12288-m2-gpu2 \
  python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --m-values 2 --dtype fp16 --splitk4-m16 \
  --warmup 1 --iters 2 --rounds 1
```

Raw report metrics:

```text
grid / block                         256 CTAs / 128 threads
registers / static shared            48 / 4.10 KiB
NCU replay duration                  58.30 us
compute / memory throughput          14.98% / 18.62%
theoretical / achieved occupancy     62.50% / 12.61%
achieved active warps per SM         8.07
active / eligible warps per sched    2.07 / 0.19
issued warps per scheduler           0.16
no eligible scheduler cycles         83.52%
long-scoreboard stall share          69.11%
```

The report classifies the kernel as latency-bound. Register/shared limits permit much higher residency, but the
256-CTA grid supplies only about two CTAs, or eight warps, per SM. Most issue slots are empty while those warps
wait on L1TEX dependencies; neither compute nor memory throughput is close to saturation. NCU replay duration is
diagnostic only and is not used as final timing evidence.

Profiler artifact:

```text
/tmp/amplin-splitk4-k12288-m2-gpu2.ncu-rep
size 2207966 bytes
sha256 cf59002885a7953c64e20bbbe5125c5ef3cd8305846505ea00cffc8f90e810eb
```

Decision:

- Retain split-K4 as an explicit K12288 research control because it robustly halves the best prior Amplin
  Tensor-Core latency.
- Do not route it automatically because it does not beat Marlin.
- Test eight intra-CTA K partitions next. A 256-thread CTA preserves the 256 output CTAs but doubles the supplied
  warps to about 16 per SM and shortens each warp's serial group chain from 24 to 12. The expected costs are an
  8 KiB shared fragment buffer and a larger warp-0 reduction.

Result: intra-CTA split-K is the correct mechanism for the K12288 latency deficit, but four warps do not expose
enough eligible work on a 124-SM Ampere device.

### 2026-07-24 — Success: K12288 split-K8 reaches four cross-device Marlin wins

Revision before experiment: `05d76a6d`, with the split-K8 implementation in the working tree.

The explicit `mma_lane_m16_n16_splitk8` operator preserves the same
`[N64,K128,K16,N16,lane]` packed-int32 layout and M16 register-padding contract as split-K4. It assigns eight
warps to each N16 output tile, so warp `w` owns K128 groups `w, w+8, ...`. Each warp therefore traverses 12
groups rather than split-K4's 24. The CTA writes distributed FP32 accumulator fragments to an 8 KiB shared
buffer, crosses one CTA barrier, and lets warp 0 reduce and store the real M rows. Weight/scale/activation
traffic is unchanged. There is still one ordinary launch with no global scratch, atomics, cooperative launch,
or grid barrier.

The `sm_80` JIT fingerprint is `1cb35b3168fe2829`. `cuobjdump --dump-resource-usage` reports the same static
resources for FP16 and BF16:

```text
schedule   threads/CTA  registers/thread  static shared  local  stack
split-K4       128              48            4096 B       0      0
split-K8       256              48            8192 B       0      0
```

The focused matrix passed on both allowed physical devices:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
pytest -q tests/kernels/test_amplin.py \
  -k 'padded_m16_large_mlp_projection or hmma_gemm_rejects_inputs_outside_contract'
```

Each device reports 9 passed and 69 deselected. The matrix performs 64 numerical comparisons per device:
32 one-warp padded-M16, 16 split-K4, and 16 split-K8 outputs against independent FP32 dequantized weights.
It covers exact Qwen3-8B and Laguna S 2.1 large MLP shapes, FP16/BF16, and M=2/4/8/16 on a non-default CUDA
stream. The boundary test separately proves that split-K8 rejects K values other than 12288.

The seven-round timing gate used both exact K12288 down projections:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both \
  --padded-m16 --splitk4-m16 --splitk8-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-splitk8-m16-k12288-gpu<1-or-2>-r7.json
```

Physical GPU 2 was empty and idle. Physical GPU 1 still had 0% utilization and no visible compute process, but
`nvidia-smi` reported the same unexplained 959 MiB allocation and 1215 MHz SM clock seen in the prior gate.
Candidate timings nevertheless reproduce closely across devices. Split-K8 has 11/16 cross-device,
range-separated wins over split-K4, no cross-device loss, and five overlap/mixed cells. Its per-device median
speedup over split-K4 ranges from approximately 0.994-1.430x on GPU 1 and 1.001-1.482x on GPU 2.

Four cells beat Marlin with disjoint full seven-round ranges on both devices:

```text
shape/dtype/M          GPU1 split-K8 range  GPU1 Marlin range  speedup  GPU2 split-K8 range  GPU2 Marlin range  speedup
Laguna BF16 M2          20.111-20.132 us     23.265-28.749 us   1.216x   19.389-19.420 us     22.856-26.081 us   1.196x
Laguna BF16 M4          23.096-23.117 us     23.199-26.726 us   1.029x   22.364-22.426 us     22.789-28.370 us   1.030x
Laguna FP16 M2          19.308-19.343 us     22.605-29.020 us   1.212x   18.483-22.211 us     22.917-26.916 us   1.253x
Qwen BF16 M2            22.994-23.030 us     24.356-28.109 us   1.069x   22.543-22.758 us     24.422-71.158 us   1.641x
```

The GPU-2 Qwen BF16 Marlin median is visibly contaminated, so its 1.641x value is not treated as a stable
speedup magnitude; the range ordering still independently confirms the candidate win. Overall cross-device
outcomes versus Marlin are four wins, seven losses, and five mixed/overlap cells. The robust wins are narrow
and concentrated at M=2 plus Laguna BF16 M=4; M=8/16 remain Marlin territory.

Artifacts:

```text
/tmp/amplin-splitk8-m16-k12288-gpu1-r7.json
size 216180 bytes
sha256 55e552a38e06b1b825aa45a9fbade2d66663918ac64fb37565d6cf4faddc6f91

/tmp/amplin-splitk8-m16-k12288-gpu2-r7.json
size 216114 bytes
sha256 27c4586ec42544714a5cadc64c7903e80c4cbe03b01c4c6b64bf438d078a6d84
```

A matched 13-pass Nsight Compute capture profiled Qwen FP16 M2 K12288xN4096 on physical GPU 2:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-8be4c651-4058-83df-154b-291f1b86add8 \
TORCH_CUDA_ARCH_LIST=8.0 \
ncu --target-processes all --kernel-name regex:splitk8 --launch-count 1 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  -o /tmp/amplin-splitk8-k12288-m2-gpu2 \
  python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --m-values 2 --dtype fp16 --splitk8-m16 \
  --warmup 1 --iters 2 --rounds 1
```

The profile verifies the intended mechanism:

```text
metric                              split-K4     split-K8
NCU replay duration                 58.30 us      35.14 us
compute / memory throughput       14.98/18.62%  25.88/30.78%
achieved occupancy                   12.61%        25.03%
achieved active warps per SM           8.07         16.02
active warps per scheduler              2.07          4.06
eligible warps per scheduler            0.19          0.41
issued warps per scheduler              0.16          0.29
no-eligible scheduler cycles           83.52%        71.14%
long-scoreboard stall share             69.11%        65.04%
```

NCU replay duration remains diagnostic and is not final timing evidence. Split-K8 doubles achieved active
warps, more than doubles eligible warps, and nearly doubles issue rate without changing register use. The
kernel remains latency-bound: its 256-CTA grid is only 0.41 waves, 71.14% of scheduler cycles have no eligible
warp, and L1TEX long-scoreboard dependencies still account for about 65% of issue interval.

Profiler artifact:

```text
/tmp/amplin-splitk8-k12288-m2-gpu2.ncu-rep
size 2286978 bytes
sha256 3a66ae6a02d55cef7a6fcee1c68a64e292c2d14416fc7ff66c7b055d367f5ae4
```

Decision:

- Retain split-K8 as the new best explicit K12288 packed-layout control.
- Do not route canonical `gemv` yet. The winning operator requires the alternate packed weight lifecycle, and
  only four of 16 cells beat Marlin robustly.
- Preserve split-K4 as the direct scheduling control.
- Test 16 intra-CTA K partitions next. A 512-thread CTA gives each warp six K128 groups and can supply about
  32 active warps per SM for these grids. The expected costs are 16 KiB shared memory, a 16-way warp-0
  fragment reduction, and lower block residency.

Result: increasing warp-level K parallelism directly fixes the Ampere scheduler starvation diagnosed in
split-K4, and the redesigned packed-int32 path now beats Marlin on four real large-model low-batch cells.

### 2026-07-24 — Mixed result: split-K16 accelerates N3072 but hits the N4096 CTA-wave cliff

Revision before experiment: `8dae6327`, with the split-K16 implementation in the working tree.

The explicit `mma_lane_m16_n16_splitk16` control extends the same packed layout and intra-CTA reduction to 16
warps. Each warp owns six of the 96 K128 groups. The CTA uses 512 threads and a 16 KiB shared FP32 fragment
buffer. The JIT fingerprint is `e845bbd0df08d178`. Unlike split-K4/K8, code generation requires 54
registers/thread for both FP16 and BF16:

```text
schedule    threads/CTA  registers/thread  static shared  local  stack
split-K8        256              48            8192 B       0      0
split-K16       512              54           16384 B       0      0
```

The same focused command passes on physical GPUs 1 and 2: 9 passed and 69 deselected per device. With
split-K16 added, each run now performs 80 independent FP32-dequant comparisons across the one-warp, split-K4,
split-K8, and split-K16 schedules. The contract test proves rejection outside K=12288.

The seven-round gate used:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both \
  --splitk8-m16 --splitk16-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-splitk16-m16-k12288-gpu<1-or-2>-r7.json
```

Split-K16 versus split-K8 has seven cross-device range-separated wins, four losses, and five mixed/overlap
cells. All seven robust wins are Laguna N=3072 cells; the only Laguna cell without a robust result is noisy
FP16 M4. Laguna median speedup is 1.073-1.250x on GPU 1 and 1.049-1.196x on GPU 2. Qwen N=4096 M4/M8 instead
regresses to 0.762-0.829x split-K8 on both devices.

The shape split is caused by launch topology, not numerical behavior:

```text
shape                    N16 CTAs  resident CTA capacity  launch consequence
Laguna K12288xN3072          192          248             all CTAs fit one resident wave
Qwen K12288xN4096            256          248             eight CTAs form a second tail wave
```

At 54 registers/thread, only two 512-thread CTAs fit each of the 124 SMs. The Qwen grid therefore crosses the
248-CTA capacity cliff, while Laguna does not. The extra Qwen tail grows with the amount of real-row activation
work. Split-K16 has only three robust Marlin wins, nine losses, and four mixed/overlap cells across devices:
Laguna FP16 M2 and Laguna BF16 M2/M4 win. It improves the magnitude of those existing low-M wins but does not
add a new robust winning cell beyond the shape-aware split-K8 set.

Artifacts:

```text
/tmp/amplin-splitk16-m16-k12288-gpu1-r7.json
size 173486 bytes
sha256 41db27d0ceb16196a7b70ffa4c212b202f3a1d4c7adb4770dd8cb3dd4633f6a4

/tmp/amplin-splitk16-m16-k12288-gpu2-r7.json
size 173577 bytes
sha256 b61c0a6682df7a8374da2b1945efc8a22ff9f1a491edbddda188cdbdabc5042e
```

The matched Qwen FP16 M2 GPU-2 Nsight Compute capture used the previous section's command with
`regex:splitk16` and `--splitk16-m16`. It directly diagnoses the cliff:

```text
metric                              split-K8     split-K16
grid / block                        256 / 256     256 / 512
registers / static shared           48 / 8 KiB    54 / 16 KiB
waves per SM                           0.41          1.03
register block limit                      5             2
theoretical / achieved occupancy   62.50/25.03%  50.00/47.84%
achieved active warps per SM           16.02         30.62
eligible warps per scheduler            0.41          0.82
issued warps per scheduler              0.29          0.37
no-eligible scheduler cycles           71.14%        62.69%
NCU replay duration                 35.14 us      43.62 us
```

The profiler explicitly reports one full wave plus a partial wave of eight CTAs that may account for up to 50%
of runtime. Split-K16 improves occupancy and scheduler eligibility as intended, but the Qwen tail outweighs
that gain. NCU duration remains diagnostic only.

```text
/tmp/amplin-splitk16-k12288-m2-gpu2.ncu-rep
size 2378770 bytes
sha256 f0a95809847b574ed1294f843a71b221822037591e50a88954503a496ed00548
```

Decision:

- Retain split-K16 as an explicit N3072 scheduling control; never select it for N4096.
- Keep split-K8 as the better N4096 control.
- Do not integrate either into canonical packed-weight routing yet.
- Test split-K12 next. K12288 has 96 groups divisible by 12, giving eight groups/warp. A 384-thread CTA can fit
  three blocks/SM even near the current register count, so both the 192- and 256-CTA grids should complete
  without a partial tail wave.

Result: more Ampere warp parallelism is beneficial only while launch geometry remains below a resident-CTA
cliff. CTA-wave topology must be a first-class dimension of Amplin schedule selection.

### 2026-07-24 — Success: split-K12 removes the N4096 tail and improves Qwen BF16

Revision before experiment: `2f2f87a3`, with the split-K12 implementation in the working tree.

The explicit `mma_lane_m16_n16_splitk12` operator uses 12 warps, eight K128 groups/warp, and a 12 KiB shared
fragment buffer. The JIT fingerprint is `eefb9d5f43977aa0`. Static resources differ by dtype:

```text
dtype  threads/CTA  registers/thread  static shared  local  stack  register block limit
FP16       384              40           12288 B       0      0             4
BF16       384              56           12288 B       0      0             3
```

Even BF16 can host three CTAs/SM, for 372 resident CTAs across 124 SMs. Both the Qwen 256-CTA and Laguna
192-CTA grids therefore fit without the split-K16 partial-wave tail.

Focused correctness passes on physical GPUs 1 and 2: 9 passed and 69 deselected per device. Each matrix now
executes 96 FP32-dequant comparisons: 32 one-warp cases plus 16 each for split-K4/K8/K12/K16. The split-K12
contract rejects K values other than 12288.

The cross-device timing gate used all three serious split-K controls:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both \
  --splitk8-m16 --splitk12-m16 --splitk16-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-splitk12-m16-k12288-gpu<1-or-2>-r7.json
```

Split-K12 versus split-K8 has nine cross-device range-separated wins, two losses, and five mixed/overlap
cells. It wins Qwen BF16 M2/M4/M8 on both devices by median factors of approximately 1.015-1.047x. It also
wins six Laguna cells, although split-K16 remains the stronger Laguna schedule. The two robust split-K8 wins
are Laguna FP16 M2 and Qwen FP16 M4.

Split-K12 versus split-K16 is intentionally balanced: five cross-device wins, five losses, and six
mixed/overlap cells. All five robust wins are Qwen rows (BF16 M2/M4/M8 and FP16 M4/M8). All five robust losses
are Laguna rows. This confirms that the CTA-capacity model predicts which schedule to use:

```text
shape/dtype tendency                     retained best control
Qwen N4096 BF16 M2/M4/M8                split-K12
Qwen N4096 FP16 M2/M4                   split-K8 or split-K16 by exact M
Laguna N3072 low-M and most other cells  split-K16
```

Against Marlin, split-K12 has four robust cross-device wins, eight losses, and four mixed/overlap cells. The
wins remain Laguna FP16 M2, Laguna BF16 M2/M4, and Qwen BF16 M2. The schedule improves Qwen BF16 M2 to stable
22.016-22.047 us on GPU 2 and 22.328-22.369 us on GPU 1, versus Marlin ranges of 24.243-53.212 us and
24.274-27.674 us respectively. M8/M16 still lose to Marlin and are now treated as an activation-reuse target
rather than a pure occupancy target.

Artifacts:

```text
/tmp/amplin-splitk12-m16-k12288-gpu1-r7.json
size 216676 bytes
sha256 f526f77ee7475d5e1f02bcb91d3428a298a81c4d8dfcaeaec052020be5e2e6b6

/tmp/amplin-splitk12-m16-k12288-gpu2-r7.json
size 216580 bytes
sha256 9f4024ddca522571df0ca1e74a1f4505d717a117a6cd9574fdd5d928204d698e
```

A 13-pass Nsight Compute capture profiled the Qwen BF16 M2 split-K12 winner on physical GPU 2. The command
matches the earlier profile command with `regex:splitk12`, `--dtype bf16`, and `--splitk12-m16`.

```text
grid / block                         256 CTAs / 384 threads
registers / static shared            56 / 12.29 KiB
waves per SM                         0.69
theoretical / achieved occupancy     56.25% / 36.78%
achieved active warps per SM         23.54
active / eligible warps per sched    6.16 / 0.95
issued warps per scheduler           0.43
no-eligible scheduler cycles         57.01%
compute / memory throughput          33.58% / 37.11%
long-scoreboard stall share          52.11%
NCU replay duration                  29.15 us
```

There is no partial wave. Relative to the earlier split-K8 FP16 profile, split-K12 substantially increases
active/eligible work and issue rate while reducing no-eligible cycles. The kernel is still latency-bound and
repeats the same activation fragment for every N16 CTA. NCU duration is diagnostic only.

```text
/tmp/amplin-splitk12-k12288-bf16-m2-gpu2.ncu-rep
size 2464048 bytes
sha256 2c04e0c75fdb69f40506cc9db8207ad1c7c40c9946c22965b72e58261dce48c0
```

Decision:

- Retain split-K12 as the best Qwen BF16 schedule and an explicit scheduling control.
- Retain split-K16 for Laguna and split-K8 for the remaining Qwen low-M cases.
- Do not route canonical weights until the alternate packed layout has an owned lifecycle.
- The next M8/M16 experiment will make one split-K12 warp accumulate N32 instead of N16. It reuses each loaded
  A fragment across two adjacent N16 weight tiles, halves activation loads and CTA count, and measures whether
  added instruction-level parallelism can offset the lower grid-level warp supply.

Result: Amplin requires an Ampere-aware schedule family selected by dtype, output width, and resident-wave
capacity. A single split-K factor is demonstrably not optimal even for one K value.

### 2026-07-24 — Success: N32 activation reuse wins every M8/M16 control comparison

Revision before experiment: `34fc7cbf`, with the N32 implementation in the working tree.

The explicit `mma_lane_m16_n32_splitk12` operator makes each split-K12 warp accumulate four M16xN8 FP32
fragments instead of two. One loaded A fragment is reused across two adjacent N16 packed-weight tiles. This
halves activation-fragment loads and grid size while preserving the exact packed-int32 layout, total weight
traffic, and one-launch/no-global-scratch contract. The CTA uses 12 warps and a 24 KiB shared partial buffer.

The `sm_80` JIT fingerprint is `72a390b2cc3d48db`. Static resources:

```text
dtype  output tile  grid Qwen/Laguna  threads  registers/thread  shared
FP16      N16          256 / 192        384           40          12 KiB
BF16      N16          256 / 192        384           56          12 KiB
FP16      N32          128 / 96         384           54          24 KiB
BF16      N32          128 / 96         384           52          24 KiB
```

Focused correctness passes on both physical GPUs 1 and 2: 9 passed and 69 deselected per device. Each run now
performs 112 FP32-dequant comparisons, including 16 N32 cases over both exact K12288 projections, FP16/BF16,
and M=2/4/8/16 on a non-default stream. The explicit K and N32-width contracts are also checked.

The seven-round gate compared N32 with N16 split-K12, N16 split-K16, and Marlin:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both \
  --splitk12-m16 --splitk12-n32 --splitk16-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-splitk12-n32-k12288-gpu<1-or-2>-r7.json
```

The crossover is exact and reproduces on both GPUs:

```text
M regime   N32 vs N16 split-K12   N32 vs N16 split-K16
2, 4       8 robust losses         8 robust losses
8, 16      8 robust wins           8 robust wins
```

At M8, N32's cross-device median speedup over N16 split-K12 is approximately 1.026-1.092x. At M16 it rises to
1.300-1.419x. Representative M16 results:

```text
shape/dtype     GPU1 N32 / N16 us  speedup   GPU2 N32 / N16 us  speedup
Laguna BF16       31.478 / 42.926    1.364x    31.524 / 42.276    1.341x
Qwen BF16         47.549 / 63.539    1.336x    46.787 / 63.145    1.350x
Laguna FP16       31.534 / 44.037    1.396x    31.176 / 44.237    1.419x
Qwen FP16         47.795 / 62.126    1.300x    46.961 / 62.889    1.339x
```

N32 still records no robust cross-device Marlin win. Its M16 median is approximately 31.2 us for Laguna and
47 us for Qwen, versus Marlin near 24-25 us. The experiment therefore validates activation reuse but also
exposes insufficient grid-level warp supply.

Artifacts:

```text
/tmp/amplin-splitk12-n32-k12288-gpu1-r7.json
size 221662 bytes
sha256 56a26341b3523aa4f7b398d53614c8473eee1d445357c8357e234ad502448ac5

/tmp/amplin-splitk12-n32-k12288-gpu2-r7.json
size 221631 bytes
sha256 6a59aca5dae74c17f74c70bd693f6c5c0b2f59b8491546511711cd6603b381d2
```

Matched Laguna BF16 M16 N16/N32 Nsight Compute captures on physical GPU 2 show the tradeoff:

```text
metric                              N16 split-K12   N32 split-K12
grid / block                         192 / 384       96 / 384
registers / shared                    56 / 12 KiB     52 / 24 KiB
waves per SM                              0.52            0.26
achieved occupancy                       30.62%          18.48%
active warps per SM                      19.60           11.83
eligible warps per scheduler              0.26            0.29
issued warps per scheduler                0.19            0.25
L1/TEX throughput                        87.39%          47.92%
memory throughput                        67.78%          34.83%
long-scoreboard cycles/share          16.8 / 65.5%    8.9 / 73.3%
NCU replay duration                      49.73 us         51.14 us
```

N32 halves L1/TEX pressure and raises issue rate despite having about 40% fewer active warps. The NCU replay
duration inverts the warmed seven-round result and is excluded from timing claims; replay is diagnostic only.
The low 96-CTA Laguna grid leaves 28 SMs empty, and NCU explicitly flags this underfill.

```text
/tmp/amplin-n16-splitk12-k12288-bf16-m16-gpu2.ncu-rep
size 2625331 bytes
sha256 0df038155e4db81022b7d04a021233a4788e93bc6ba8336c0549dc717857c4dd

/tmp/amplin-n32-splitk12-k12288-bf16-m16-gpu2.ncu-rep
size 2625408 bytes
sha256 2ba5ff27bae5c3d1175a672acc80d72057658909dfffaa95defb7e1fb9eab981
```

Decision:

- Retain N32 split-K12 as the best M8/M16 activation-reuse control.
- Never select N32 at M2/M4.
- Do not route it automatically because Marlin still wins every robust cross-device comparison.
- Next, keep N32 reuse but increase the split to 16 warps. N32 split-K16 supplies 16.5 average warps/SM for
  Qwen and 12.4 for Laguna instead of 12.4/9.3, while each warp traverses six rather than eight K128 groups.

Result: for larger batches, repeated A-fragment traffic—not just K dependency depth—is a first-order Ampere
bottleneck. N32 reuse is the correct M8/M16 direction, but it needs more warps without reducing the grid again.

### 2026-07-24 — Mixed result: N32 split-K16 wins M8 but loses M16

Revision before experiment: `5c6f5b12`, with N32 split-K16 in the working tree.

The explicit `mma_lane_m16_n32_splitk16` control keeps the exact N32 activation-reuse math and increases only
the intra-CTA K partitions from 12 to 16. Each warp traverses six K128 groups rather than eight. The grid
remains 128 CTAs for Qwen and 96 for Laguna. JIT fingerprint `b00296b4e86859c7` reports:

```text
dtype  schedule       threads/CTA  registers/thread  static shared
FP16   N32 split-K12      384              54           24 KiB
BF16   N32 split-K12      384              52           24 KiB
FP16   N32 split-K16      512              48           32 KiB
BF16   N32 split-K16      512              50           32 KiB
```

The 512-thread candidate supplies 16.5 average warps/SM for Qwen and 12.4 for Laguna, versus split-K12's
12.4/9.3. Its resource limits permit two resident CTAs/SM, so neither grid creates a split-K16 N16-style tail.

Focused correctness passes on both physical GPUs 1 and 2: 9 passed and 69 deselected per device. The matrix now
performs 128 FP32-dequant comparisons per device and covers both N32 split factors at every tested dtype/shape/M.

The seven-round gate used:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 2,4,8,16 --dtype both \
  --splitk12-n32 --splitk16-n32 --splitk16-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-splitk16-n32-k12288-gpu<1-or-2>-r7.json
```

N32 split-K16 versus N32 split-K12 records 11 cross-device range-separated wins, four losses, and one mixed
cell. The regime boundary is informative:

```text
M       outcome versus N32 split-K12
2, 4    wins broadly, but N16 controls remain substantially faster
8       3 robust wins and 1 mixed cell; median improvement 1.010-1.083x
16      4 robust losses; split-K16 is 3.0-10.2% slower
```

At M8, N32 split-K16 also beats N16 split-K16 in all four shape/dtype cells on both devices. At M16 it still
beats N16 split-K16, but N32 split-K12 remains the best Amplin control. The final large-batch schedule map is:

```text
M=8   -> N32 split-K16
M=16  -> N32 split-K12
```

The candidate establishes only one robust cross-device Marlin win, Laguna BF16 M2, where the already-retained
N16 schedules are much faster. It therefore adds no useful Marlin crossover. M8/M16 remain behind Marlin.

Artifacts:

```text
/tmp/amplin-splitk16-n32-k12288-gpu1-r7.json
size 221791 bytes
sha256 27664002e0512e588145cc4a0633414877a8e21e42f2c96fdbe40052828e2bb6

/tmp/amplin-splitk16-n32-k12288-gpu2-r7.json
size 221746 bytes
sha256 6cfc8c2b5487825e79e5d4e48ab5ed87a8fcdf829a78dcae03d61daf948d4bda
```

Decision:

- Retain N32 split-K16 only as the best M8 control.
- Retain N32 split-K12 as the best M16 control.
- Never select either N32 path for M2/M4.
- Do not route any explicit packed operator automatically until its alternate weight lifecycle is integrated.
- Stop increasing the split factor. The matched N32 profile already shows L1TEX long-scoreboard stalls dominate
  while the 96/128-CTA grids are underfilled. The next bounded M8/M16 optimization should preserve the selected
  N32 split factor and software-pipeline global A/packed-word loads to create more per-warp eligible work.

Result: the best Ampere schedule is piecewise even within M8/M16. More warps improve M8, while M16 benefits from
fewer partitions and a smaller final reduction. Amplin now has evidence-backed controls for both regimes.

### 2026-07-24 — Failure: concurrent five-batch gate perturbs launch timing

Revision: `d44c9ffd`.

The complete large-projection comparison was extended to the requested common batches `M={1,2,4,8,16}`. It
included the batch-1 K12288 wide path, every retained N16/N32 split-K control for M2-M16, canonical Amplin, and
Marlin:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 1,2,4,8,16 --dtype both \
  --k12288-wide \
  --splitk8-m16 --splitk12-m16 --splitk12-n32 \
  --splitk16-n32 --splitk16-m16 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-five-batch-k12288-gpu<1-or-2>-r7.json
```

The two UUID-pinned benchmark processes ran concurrently. Both completed and each artifact contains 124 timing
rows. All 124 independent FP32-dequant comparisons per device are finite and within the existing dtype
tolerances; maximum absolute error is 0.0085373.

The timing stability gate failed:

```text
device  rows  rows with >10% round spread  Marlin  Amplin  worst spread
GPU 1   124               30                 17      13       135.9%
GPU 2   124               46                 19      27       651.3%
```

CUDA events around a batch of launches include gaps when the host fails to enqueue the next launch before the
GPU drains its queue. Running both Python harnesses concurrently therefore perturbed even separate physical
devices. Several Marlin medians also moved far outside the prior clean gates. The artifacts prove all five
batches execute correctly, but they are rejected for performance claims.

Artifacts:

```text
/tmp/amplin-five-batch-k12288-gpu1-r7.json
sha256 15a24afe41e553ff058c6163ed45cb9394acbacaefb9645d1329c40c7ea18927

/tmp/amplin-five-batch-k12288-gpu2-r7.json
sha256 93823d1c3c7f931d689bd55d3b5e480c0feb83fbf0490630ca9d4fa6cab14824
```

Decision:

- Keep the five-batch correctness coverage.
- Reject all performance numbers from the concurrent run.
- Repeat the identical gate sequentially on physical GPUs 1 and 2 before publishing Amplin versus Marlin.

Result: performance-gate failure caused by concurrent host launch contention; no kernel or schedule change.

### 2026-07-24 — Success: sequential GPU-1/GPU-2 five-batch gate

Revision: `c609faaf`.

The rejected concurrent command was repeated unchanged, one process at a time, on physical PCI-order GPU 2 and
then GPU 1. Both devices were selected by UUID, not inferred from a CUDA index:

```text
device  UUID                                          board       CC   SMs  memory
GPU 1   GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855    PG506-232   8.0  124  96 GiB
GPU 2   GPU-8be4c651-4058-83df-154b-291f1b86add8    PG506-230   8.0  124  96 GiB
```

The software stack remained PyTorch 2.13.0+cu130 and CUDA runtime 13.0. Each device produced 124 timing rows and
124 independent FP32-dequant comparisons over exact Qwen3-8B K12288xN4096 and Laguna S 2.1 K12288xN3072
projections, FP16/BF16, and `M={1,2,4,8,16}`. Every correctness comparison passed; maximum absolute error was
0.0085373.

Sequential execution reduced rows with greater than 10% seven-round spread from 30/46 to 20/12 on GPU 1/GPU 2,
and reduced worst spread from 135.9%/651.3% to 31.0%/19.9%. Remaining spread is concentrated in exploratory
controls and Marlin. The primary statistic remains the median of seven batched CUDA-event rounds, each containing
200 launches after 50 warmups. Range-overlap cells are explicitly not promoted to wins.

The table uses the previously selected piecewise Amplin schedule. `speedup = Marlin / Amplin`; values above one
favor Amplin. `A`, `M`, and `~` mean the seven-round ranges are respectively disjoint in Amplin's favor, disjoint
in Marlin's favor, or overlap on that GPU.

```text
model   dtype  M   Amplin schedule  GPU1 A/M us    speedup/outcome  GPU2 A/M us    speedup/outcome
Qwen    FP16   1   wide             38.236/25.702      0.672x M     42.639/24.468      0.574x M
Qwen    FP16   2   N16 split-K16    22.513/23.455      1.042x A     22.415/23.398      1.044x A
Qwen    FP16   4   N16 split-K8     26.071/23.470      0.900x M     25.871/23.419      0.905x ~
Qwen    FP16   8   N32 split-K16    32.799/22.625      0.690x M     33.608/22.523      0.670x M
Qwen    FP16  16   N32 split-K12    47.421/25.426      0.536x M     47.273/25.359      0.536x M
Qwen    BF16   1   wide             38.717/24.873      0.642x M     39.357/23.224      0.590x M
Qwen    BF16   2   N16 split-K12    22.446/26.450      1.178x A     21.960/24.499      1.116x A
Qwen    BF16   4   N16 split-K12    25.201/24.571      0.975x ~     24.837/24.509      0.987x ~
Qwen    BF16   8   N32 split-K16    33.777/25.999      0.770x M     34.376/23.491      0.683x M
Qwen    BF16  16   N32 split-K12    47.386/27.356      0.577x M     46.935/27.213      0.580x M
Laguna  FP16   1   wide             25.595/21.356      0.834x M     25.549/21.489      0.841x M
Laguna  FP16   2   N16 split-K16    15.764/22.518      1.428x A     15.775/22.093      1.401x A
Laguna  FP16   4   N16 split-K16    18.775/22.523      1.200x A     18.790/22.139      1.178x A
Laguna  FP16   8   N32 split-K16    24.003/21.791      0.908x ~     23.941/21.002      0.877x M
Laguna  FP16  16   N32 split-K12    31.329/25.544      0.815x M     31.380/23.572      0.751x M
Laguna  BF16   1   wide             25.989/21.745      0.837x M     25.974/21.458      0.826x M
Laguna  BF16   2   N16 split-K16    16.189/23.977      1.481x A     16.323/22.932      1.405x A
Laguna  BF16   4   N16 split-K16    19.016/24.028      1.264x A     19.005/22.892      1.204x A
Laguna  BF16   8   N32 split-K16    24.509/22.646      0.924x ~     24.509/21.627      0.882x M
Laguna  BF16  16   N32 split-K12    31.524/25.042      0.794x M     31.662/24.914      0.787x M
```

Cross-device range-separated summary:

```text
outcome                             cells
Amplin wins both devices              6
Marlin wins both devices             10
mixed or overlapping ranges           4
```

The robust Amplin wins are all four batch-2 cells plus Laguna FP16/BF16 batch 4. Marlin robustly wins all four
large-projection batch-1 cells and all four batch-16 cells. Batch 8 is also Marlin-dominant. Qwen batch 4 is
either a Marlin win or statistical parity, while Laguna batch 4 remains an Amplin strength.

Artifacts:

```text
/tmp/amplin-five-batch-k12288-gpu1-seq-r7.json
size 342110 bytes
sha256 0415a61db064ef43c5a7d4b737c9c3d5918cbf4c86f7f15bd2b842ee6c5fff83

/tmp/amplin-five-batch-k12288-gpu2-seq-r7.json
size 342150 bytes
sha256 59d346b64257440f5b026bb4758512e221d4ee0f0774b7cca3f87adaa34933cb
```

Decision:

- Keep batch 1, 2, 4, 8, and 16 in every real-shape gate.
- Preserve the piecewise research schedule only as explicit benchmark controls; no automatic packed-weight
  lifecycle or production routing is added.
- Focus the next large-shape kernel work on M8/M16 activation/load pipelining. Do not infer that the batch-2
  split-K win generalizes to batch 8 or 16.

Result: complete cross-device five-batch performance and correctness gate. Amplin owns a clear large-projection
batch-2 niche and Laguna batch 4, while Marlin remains the target at batch 1, batch 8, and batch 16.

### 2026-07-24 — Success: reconcile the batch-1 all-shape and large-shape results

Revision: `a3b31ab0`.

The apparent reversal between the earlier statement that Amplin wins batch 1 and the latest table was checked
directly. The tables covered different shape sets:

- The earlier common-batch gate covered all 12 Marlin-legal Qwen3-8B and Laguna S 2.1 projection classes.
- The latest five-batch table deliberately covered only K12288xN4096 and K12288xN3072, the two large-down
  projections selected as the mega-kernel optimization target.

The historical all-shape gate already reported batch-1 losses for those two large-down projections:

```text
shape                  historical GPU6 Marlin/Amplin FP16/BF16
Qwen 12288x4096                         0.553x / 0.548x
Laguna 12288x3072                       0.893x / 0.790x
```

Therefore the large-shape table did not demonstrate a regression. It isolated the known deficit.

The current source was nevertheless rechecked on only physical PCI-order GPUs 1 and 2. Runs were sequential and
UUID-pinned:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<physical-GPU-1-or-2-UUID> \
TORCH_CUDA_ARCH_LIST=8.0 \
python -m scripts.benchmark_amplin_model_shapes \
  --model all --dtype both --m-values 1 --k12288-wide \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-batch1-all-shapes-gpu<1-or-2>-recheck-r7.json
```

Both devices remained sm_80, 124 SM, 96 GiB boards under PyTorch 2.13.0+cu130 and CUDA runtime 13.0. Each
artifact contains 56 rows and 56 passing independent FP32-dequant comparisons. The table reports median
`Marlin/Amplin` speedup; values above one favor Amplin:

```text
model/role             KxN          GPU1 FP16/BF16   GPU2 FP16/BF16
Qwen KV                4096x1024      2.664/2.664      2.472/2.747
Qwen Q/O               4096x4096      1.523/1.467      1.482/1.537
Qwen MLP-up            4096x12288     0.794/0.760      0.803/0.825
Qwen MLP-down         12288x4096      0.622/0.598      0.585/0.662
Laguna expert-down     1024x3072      2.747/2.914      2.849/3.249
Laguna KV/expert-up    3072x1024      2.200/2.208      2.063/2.332
Laguna Q-proj-6144     3072x6144      1.349/1.298      1.280/1.333
Laguna Q-proj-9216     3072x9216      1.148/1.160      1.136/1.159
Laguna dense-up        3072x12288     0.933/0.887      0.886/1.044
Laguna O-proj-6144     6144x3072      1.426/1.339      1.381/1.480
Laguna O-proj-9216     9216x3072      0.961/0.928      0.943/1.018
Laguna dense-down     12288x3072      0.877/0.848      0.890/0.998
```

Nominally, Amplin wins 14/24 dtype/shape cells on GPU 1 and 16/24 on GPU 2. Requiring the complete seven-round
ranges to be disjoint on both devices gives 13 robust Amplin wins, two robust Marlin wins, and nine
mixed/overlap cells. Twenty-five of 56 GPU-1 rows and 34 of 56 GPU-2 rows exceed 10% full-range spread, mostly
because of isolated Marlin rounds, so only the cross-device range-separated cells are called robust.

For K12288 M1, automatic `amplin_raw` and explicit `amplin_k12288_wide_raw` are equal within 0.06% in every
shape/dtype/device cell. The production research route therefore did select the intended wide path; the
large-shape loss is not a dispatch error.

Artifacts:

```text
/tmp/amplin-batch1-all-shapes-gpu1-recheck-r7.json
size 154155 bytes
sha256 a75e05f3fe93c79e5b6dfff8d4a5a4875ed5d54b6ca8f99442401ab58f723c80

/tmp/amplin-batch1-all-shapes-gpu2-recheck-r7.json
size 154172 bytes
sha256 b60a4093e0728dd6293292e1e8ca55a95ccc33c6aa37a079c875774b10244e7f
```

Decision:

- Keep the statement that Amplin wins most batch-1 small/moderate projection classes.
- Qualify every batch claim by KxN shape; batch size alone is not a valid selector.
- Keep the large K12288/down projections as the batch-1 optimization target because Marlin still leads there.
- Do not interpret the large-shape-only table as the complete model-shape result.

Result: no batch-1 regression. The apparent contradiction was a reporting-scope mismatch, now verified on both
allowed GPUs.

### 2026-07-24 — Mixed success: N32 pipe2 removes scoreboard stalls and reaches Laguna M8 parity

Revision before experiment: `b1d5a487`, with the pipe2 implementation in the working tree.

The retained N32 profiles showed that each K128 group's eight K16 steps serialized four guarded activation
loads and two packed-int32 loads before four dequantize/MMA pairs. The bounded candidate changes only that
instruction schedule:

```text
current:   load A[k], W0[k], W1[k] -> dequantize/MMA four N8 fragments
pipe2:     prefetch A[k+1], W0[k+1], W1[k+1] before consuming step k
```

The candidate keeps the exact packed-int32/scales layout, N32 ownership, K12/K16 split, grid, block size,
one-launch contract, shared partial buffer, barrier, reduction order, output conversion, and sm_80 gate.
It adds no global scratch or persistent state. Current and pipe2 kernels remain separate explicit operators so
they can be compared in one process; no automatic backend route changes.

Implementation and validation support added:

- `mma_lane_m16_n32_splitk12_pipe2` and `mma_lane_m16_n32_splitk16_pipe2` CUDA/Torch operators and Python
  wrappers;
- `--splitk-n32-pipe2` in the real-model benchmark;
- focused N32 split-K controls in `scripts/profile_amplin_vs_marlin.py`;
- FP32-dequant comparisons for both pipe2 paths in the existing real-model, non-default-stream test.

Static checks passed:

```text
ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py \
  scripts/benchmark_amplin_model_shapes.py scripts/profile_amplin_vs_marlin.py
python -m py_compile gptqmodel/utils/amplin.py tests/kernels/test_amplin.py \
  scripts/benchmark_amplin_model_shapes.py scripts/profile_amplin_vs_marlin.py
git diff --check
```

The sm_80 JIT fingerprint is `59da24cd41d2a167`. GPU 2 compiled the extension and passed all 14 real-model
shape tests; GPU 1 passed both targeted K12288 shapes. Each targeted test performs 72 independent FP32-dequant
comparisons across FP16/BF16, M=2/4/8/16, and nine schedules. Pipe2 exactly matches its corresponding current
N32 error in every case.

Generated resources from `cuobjdump --dump-resource-usage`:

```text
dtype  schedule       current regs  pipe2 regs  static shared  local  stack
FP16   N32 split-K12       54           56         24 KiB        0      0
BF16   N32 split-K12       52           56         24 KiB        0      0
FP16   N32 split-K16       48           64         32 KiB        0      0
BF16   N32 split-K16       50           64         32 KiB        0      0
```

The exact pre-edit baseline and same-process candidate gates ran sequentially on UUID-pinned physical GPUs 1
and 2. Both use 50 warmups, 200 launches per batched CUDA-event round, seven rounds, both dtypes, exact
Qwen K12288xN4096 and Laguna K12288xN3072, and M8/M16:

```text
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 8,16 --dtype both \
  --splitk12-n32 --splitk16-n32 --splitk-n32-pipe2 \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-n32-pipe2-gpu<1-or-2>-r7.json
```

The table compares pipe2 with the selected current control: K16 at M8 and K12 at M16. `speedup =
current/pipe2`; `P`, `C`, and `~` mean disjoint pipe2 win, disjoint current-control win, or overlapping
seven-round ranges.

```text
shape   dtype  M   GPU1 pipe/current us  speedup/outcome  GPU2 pipe/current us  speedup/outcome
Qwen    FP16   8      36.782/33.029          0.898x ~        36.685/33.260          0.907x ~
Qwen    FP16  16      46.664/47.176          1.011x ~        47.084/47.089          1.000x ~
Qwen    BF16   8      38.257/33.987          0.888x C        38.205/33.946          0.889x C
Qwen    BF16  16      45.988/46.868          1.019x ~        45.967/46.981          1.022x P
Laguna  FP16   8      21.914/23.967          1.094x P        22.026/24.105          1.094x P
Laguna  FP16  16      30.131/31.462          1.044x P        30.111/31.437          1.044x P
Laguna  BF16   8      22.472/24.397          1.086x P        22.625/24.499          1.083x P
Laguna  BF16  16      29.998/31.601          1.053x P        29.870/31.703          1.061x P
```

All four Laguna cells are robust pipe2 wins on both devices. Qwen M8 loses, while Qwen M16 is neutral to a
small BF16 gain. The Laguna M8 candidate also reaches Marlin parity:

```text
dtype  GPU1 pipe2/Marlin us  Marlin/pipe2  range  GPU2 pipe2/Marlin us  Marlin/pipe2  range
FP16       21.914/22.784        1.040x       A        22.026/22.431        1.018x       ~
BF16       22.472/22.840        1.016x       A        22.625/22.794        1.007x       ~
```

The first Nsight Compute invocation failed before launch because the command placed
`CUDA_DEVICE_ORDER=PCI_BUS_ID` where NCU expected an executable:

```text
==ERROR== 'CUDA_DEVICE_ORDER=PCI_BUS_ID' does not exist or is not an executable.
```

The corrected command used `/usr/bin/env` after NCU's `--` separator:

```text
ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section WarpStateStats --section SchedulerStats \
  -o /tmp/<report> -- \
  /usr/bin/env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-8be4c651-4058-83df-154b-291f1b86add8 \
  TORCH_CUDA_ARCH_LIST=8.0 \
  python -m scripts.profile_amplin_vs_marlin \
    --path <n32_splitk16-or-pipe2> --dtype bf16 \
    --m 8 --k 12288 --n <3072-or-4096> \
    --warmup 50 --launches 1 --cuda-profiler-api
```

Relevant raw NCU detail output:

```text
shape/path       grid  regs  duration  compute  memory  eligible/sched  issued/sched  no eligible  warp cycles  L1TEX wait
Laguna current     96    50   57.57 us   10.25%  18.89%      0.18           0.14         85.58%       27.58       22.5
Laguna pipe2       96    64   31.49 us   32.71%  35.05%      0.83           0.47         52.53%        8.42        3.0
Qwen current      128    50   60.35 us   13.16%  24.26%      0.19           0.15         85.34%       28.10       22.9
Qwen pipe2        128    64   47.49 us   28.60%  30.64%      0.84           0.46         53.92%        9.01        3.5
```

Block size remains 512, static shared memory 32.77 KiB, theoretical occupancy 50%, and achieved occupancy
approximately 24-25% in all four reports. Pipe2 demonstrably removes the intended serial L1TEX dependency.
NCU replay duration improves even for Qwen, contrary to the stable warmed CUDA-event result; replay timing is
therefore diagnostic only and is excluded from schedule selection.

Artifacts:

```text
/tmp/amplin-n32-pipe-baseline-gpu1-r7.json
size 90618 bytes
sha256 612cd994397b660a7eb6e3863b5e0ef8f89aabfd00d6bdc63d775ebf183644df

/tmp/amplin-n32-pipe-baseline-gpu2-r7.json
size 90564 bytes
sha256 b7d46aad1bbf56577b20471598d983e2ef187d18b6ff864c014238dbee9dcc42

/tmp/amplin-n32-pipe2-gpu1-r7.json
size 134808 bytes
sha256 c24999285fe50f77c995745f73088ed96b342e145e54ad0aeeab89b60f71732f

/tmp/amplin-n32-pipe2-gpu2-r7.json
size 134814 bytes
sha256 cd10cb56804030ce77e976a052bcc6f35bd9958d4cf4ef73da732777c82a5ac4

/tmp/amplin-laguna-bf16-m8-n32-k16-base-gpu2.ncu-rep
sha256 a23227434c722fed116742084069522b9689f38e6569edc49f2c317b4c0e66b6

/tmp/amplin-laguna-bf16-m8-n32-k16-pipe2-gpu2.ncu-rep
sha256 6f61bb2f171620474036f054d49f5b38ed055c4546db6168de8240b192802713

/tmp/amplin-qwen-bf16-m8-n32-k16-base-gpu2.ncu-rep
sha256 32159141bfa75c61b38252adb1c70a92dde7584745c376fdfade9d4f15ef17a3

/tmp/amplin-qwen-bf16-m8-n32-k16-pipe2-gpu2.ncu-rep
sha256 9d06f315798ec1e163abdd31459be10891283e586d7a10c90b314c7507347087
```

Decision:

- Retain pipe2 as an explicit research control and as the best measured Laguna N3072 M8/M16 schedule.
- Do not select pipe2 for Qwen N4096 M8 and do not add automatic production routing.
- Treat ordinary CUDA-event timing as authoritative where NCU replay disagrees.
- Next, split the prefetch mechanism into activation-only and packed-word-only controls. The goal is to preserve
  the Laguna scoreboard win with fewer than 64 K16 registers and remove the Qwen M8 regression.

Result: mixed scheduling success. Explicit register prefetching fixes the measured scoreboard bottleneck and
brings Laguna M8 to Marlin parity, but output-width/grid behavior prevents one universal N32 schedule.

### 2026-07-24 — Failed experiment: splitting activation and packed-word prefetch

Revision before experiment: `30d2703d`. This experiment tested the previous checkpoint's proposed
factorization without changing the packed-int32/scales layout, launch geometry, shared partials, reduction, or
output conversion. Two temporary K16 N32 controls were added:

```text
pipeA: prefetch only the next four-register activation fragment
pipeW: prefetch only the next two packed-int32 words
```

The first sm_80 compile failed before a GPU launch because the refactored implementation template gained a
fourth Boolean parameter while the existing K12/K16 pipe2 wrappers still supplied only three:

```text
failed JIT fingerprint: 0ae38a64bce25dd1
static assertion: a prefetch mode was instantiated without the N32 output mode
pytest result: 2 setup errors; no CUDA kernels launched
```

After explicitly passing both prefetch flags in the pipe2 wrappers, JIT fingerprint `418ecf7686b1f715`
compiled. Both K12288 real-shape tests then passed on physical GPU 2, including both dtypes, M=2/4/8/16, and
all FP32-dequant schedule comparisons.

Generated K16 resource usage explains why packed-word prefetch was not the hoped-for low-pressure control:

```text
dtype  current regs  pipe2 regs  pipeA regs  pipeW regs  static shared  local  stack
FP16        48            64          64          74         32 KiB        0      0
BF16        50            64          64          52         32 KiB        0      0
```

The exact Qwen K12288xN4096 and Laguna K12288xN3072 shapes ran sequentially on UUID-pinned physical GPUs 1
and 2. Each row uses M8, 50 warmups, 200 launches per batched CUDA-event round, and seven rounds. `A/current`
and `W/current` are current-control latency divided by candidate latency. `P`, `C`, and `~` mean a disjoint
candidate win, disjoint current-control win, or overlapping complete seven-round ranges:

```text
GPU  shape   dtype  current us  pipe2 us  pipeA us  A/current  range  pipeW us  W/current  range
 1   Qwen    FP16      34.806     37.007    36.992     0.941x     ~      43.786     0.795x     C
 1   Qwen    BF16      35.574     38.385    38.380     0.927x     C      36.833     0.966x     C
 1   Laguna  FP16      24.049     22.001    22.026     1.092x     P      26.025     0.924x     C
 1   Laguna  BF16      24.494     22.615    22.605     1.084x     P      23.178     1.057x     P
 2   Qwen    FP16      41.139     42.911    42.803     0.961x     ~      52.316     0.786x     C
 2   Qwen    BF16      36.500     38.866    38.912     0.938x     C      38.912     0.938x     C
 2   Laguna  FP16      24.228     22.175    21.939     1.104x     P      26.230     0.924x     ~
 2   Laguna  BF16      24.550     22.415    22.415     1.095x     P      23.127     1.062x     P
```

Activation-only prefetch is effectively the existing pipe2 schedule: it preserves the 8-10% Laguna gain,
preserves the 4-7% Qwen loss, uses the same 64 registers, and does not separate consistently from pipe2.
Packed-word-only prefetch saves registers only in BF16, produces a smaller Laguna BF16 gain, still loses Qwen,
and expands to 74 registers in FP16. Neither control removes the output-width dependence or improves the
retained niche.

Artifacts:

```text
/tmp/amplin-n32-pipe-split-m8-gpu1-r7.json
size 80258 bytes
sha256 fb22938262bda43b6c59236eda1c04673926406d2c78ea3719b65f9f4d0498c5

/tmp/amplin-n32-pipe-split-m8-gpu2-r7.json
size 80278 bytes
sha256 24f8c131b5c31c078e9f4617ece3f13ef0354a3defcd76238c133a892512eff6
```

Decision:

- Reject and remove the temporary pipeA and pipeW operators, wrappers, benchmark flag, and correctness rows.
- Keep the already committed pipe2 research control; it captures the useful Laguna behavior with no extra
  production route.
- Do not profile the rejected controls with NCU: they provide no timing or resource advantage over the retained
  pipe2 control.
- Continue large-shape work at a coarser level than K-step prefetch factorization, because N4096 versus N3072
  grid behavior is now the unresolved discriminator.

Result: failed factorization experiment, with the failure and all transient evidence retained here while the
rejected code is removed from the branch.

### 2026-07-24 — Diagnostic: K12 pipe2 N4096 loss is not a resident-wave cliff

Revision: `9a237349`.

The fastest measured M8 Amplin control is N32 split-K12 pipe2, not split-K16 pipe2. In the earlier seven-round
gate it reaches approximately 30.2-30.7 us for Qwen K12288xN4096 and 21.0-21.2 us for Laguna K12288xN3072.
That still loses to Marlin's approximately 22.4-23.5 us on Qwen, while beating Marlin's approximately
22.4-22.8 us on Laguna. Matched 13-pass Nsight Compute captures on physical GPU 2 tested whether the Qwen gap
was caused by crossing the 124-SM device width:

```text
metric                              Qwen N4096      Laguna N3072
grid / block                         128 / 384         96 / 384
registers / static shared             56 / 24 KiB      56 / 24 KiB
theoretical / achieved occupancy    56.25/19.05%     56.25/18.51%
compute / memory throughput         17.43/32.04%     14.69/27.01%
eligible warps per scheduler             0.26              0.25
issued warps per scheduler               0.21              0.21
no-eligible scheduler cycles            78.81%            78.86%
warp cycles per issued instruction      14.61             14.27
L1TEX scoreboard cycles/share         10.4/71.2%        10.1/71.0%
NCU replay duration                    45.25 us          39.84 us
```

K12's resources permit multiple resident CTAs/SM; neither grid approaches its resident-CTA capacity. The
per-warp scheduler and stall metrics are essentially invariant across output width, and the Qwen grid improves
aggregate SM utilization relative to Laguna. The N4096 gap is therefore not the N16 split-K16 capacity cliff
seen earlier. It is the cost of 4/3 as many N32 tiles on a latency-bound kernel whose 12 warps do not hide the
remaining global-load dependency.

As in prior captures, NCU replay duration is diagnostic only and does not replace warmed CUDA-event timing.

Artifacts:

```text
/tmp/amplin-qwen-bf16-m8-n32-k12-pipe2-gpu2.ncu-rep
size 3098296 bytes
sha256 2d799259a9d8bbe62613994bd6b6fa3f598803ed5a0ef80bee5a50e7eeeb4962

/tmp/amplin-laguna-bf16-m8-n32-k12-pipe2-gpu2.ncu-rep
size 3098600 bytes
sha256 e9dcfdeb89ce1b17145d30d77899f3358d8390e2bffcea4d430825047f7a40e5
```

Decision:

- Do not remap 128 N32 tiles onto 124 SMs; K12 has no resident-wave capacity cliff to remove.
- Preserve the N32 tile and K12 partition, because they are already the best M8 large-shape Amplin control.
- Test a K12-only three-stage register pipeline. Prefetching two K16 steps ahead should increase load-to-use
  distance for the measured L1TEX dependency while still fitting below the K16 pipe2 register footprint.

Result: the next optimization target is per-warp load latency, not physical-SM grid numbering.

### 2026-07-24 — Failure: K12 pipe3 does not improve the retained pipeline

Revision before experiment: `8dba016d`, with the pipe3 candidate in the working tree.

The temporary `mma_lane_m16_n32_splitk12_pipe3` control preserved N32 ownership, K12 partitioning, the
packed-int32/scales layout, 384-thread launch, shared partial buffer, reduction order, output conversion, and
one-launch contract. Relative to pipe2, it held two future K16 activation fragments and packed-word pairs so
each load was issued two loop iterations before use. No automatic route changed.

Two pre-launch failures were corrected and retained as process evidence:

```text
pytest filter failure:
  `-k ... and 12288` selected zero tests because parameter IDs use model/role names, not dimensions
  result: exit 5, 78 deselected, no compile or CUDA launch

first JIT fingerprint: 441595fd70c2bfa3
compile failure:
  a context-matched static assertion was inserted in the adjacent N16 helper
  nvcc: identifier "PipelineStages" is undefined
  result: four setup errors, no CUDA kernel launch
```

Moving the assertion into the N32 helper produced JIT fingerprint `fb209fc918cd75ff`. Exact node IDs then
passed all four Qwen/Laguna down-projection dtype cases on both physical GPUs 1 and 2. Each device run covers
both exact K12288 projections, FP16/BF16, M=2/4/8/16, and ten schedules: 160 independent FP32-dequant
comparisons.

`cuobjdump --dump-resource-usage` showed that ptxas reused lifetimes successfully:

```text
dtype  pipe2 regs  pipe3 regs  static shared  local  stack
FP16       56          56         24 KiB        0      0
BF16       56          56         24 KiB        0      0
```

The first seven-round M8/M16 gate became invalid during execution. Physical GPUs 1 and 2 acquired unrelated,
invisible background contexts using approximately 30-34 GiB apiece and intermittently 0-13% SM utilization.
Forty-five of 48 GPU-1 rows and 43 of 48 GPU-2 rows exceeded 10% complete-range spread; one Marlin row reached
179.5 us. Those medians are excluded from kernel decisions and no absolute Amplin-versus-Marlin claim is made.

The fallback decision gate increased each batched CUDA-event sample from 200 to 1,000 launches and used nine
rounds at M8. Background interference remained visible, but the paired pipe2/pipe3 direction was clear enough
to reject a win. `P2/P3` is pipe2 latency divided by pipe3 latency:

```text
GPU  shape   dtype  pipe2 us  pipe3 us  P2/P3  pipe2 spread  pipe3 spread
 1   Qwen    FP16     38.182    42.050   0.908x      25.9%         25.5%
 1   Qwen    BF16     38.808    44.526   0.872x      23.7%         26.8%
 1   Laguna  FP16     21.704    29.542   0.735x      31.5%         13.6%
 1   Laguna  BF16     26.385    28.307   0.932x      24.5%         30.7%
 2   Qwen    FP16     32.896    42.429   0.775x      30.2%         25.4%
 2   Qwen    BF16     40.601    44.828   0.906x      26.5%         31.1%
 2   Laguna  FP16     27.458    24.126   1.138x      23.7%         33.2%
 2   Laguna  BF16     25.991    37.044   0.702x      25.6%         25.4%
```

Pipe3 is slower in seven of eight cross-device medians. Laguna BF16 has disjoint complete ranges favoring
pipe2 on both devices; the lone opposing median is noisy Laguna FP16 on GPU 2 and does not reproduce on GPU 1.
The candidate therefore has no evidence-backed cell, despite equal static resources and full correctness.

Artifacts:

```text
/tmp/amplin-n32-k12-pipe3-gpu1-r7.json
size 134874 bytes
sha256 35485d373d1f19a9b4f338f6594986c3d0a6afa5740d265c3465892b776cc8a8

/tmp/amplin-n32-k12-pipe3-gpu2-r7.json
size 134963 bytes
sha256 f9f690c4548f1704216797a6dcedf23b7c1f9f1ad06e3c67706f953e26a9e205

/tmp/amplin-n32-k12-pipe3-m8-gpu1-r9x1000.json
size 60405 bytes
sha256 ca001cbdea019246c599b43cf74065a4078e98c60943c377283c7c85cc58201d

/tmp/amplin-n32-k12-pipe3-m8-gpu2-r9x1000.json
size 60382 bytes
sha256 4246ecf3483e6711414e16de0de3983cb53d389094b4f3eec86101e68dce8d7e
```

Decision:

- Reject and remove pipe3 and all of its temporary operator, wrapper, benchmark, and correctness plumbing.
- Keep pipe2 as the only explicit register-pipeline control.
- Do not run NCU on pipe3: it has no reproducible timing win, and profiler replay cannot repair a failed
  ordinary-timing gate.
- Do not issue fresh absolute Amplin-versus-Marlin scores until physical GPUs 1 and 2 are free of background
  work; preserve the contaminated artifacts explicitly rather than reporting them as clean results.

Result: deeper lookahead is correct but slower. More load-to-use distance alone does not improve the K12
schedule, so the next large-shape design should change the amount or ownership of memory work rather than add
another register-prefetch stage.

### 2026-07-24 — Benchmark validity checkpoint: strict physical-GPU idle gate

Revision before infrastructure change: `2d15f3f5`, after merging `origin/main` into `future-1`, resolving the
single `tests/test_extension_load_api.py` conflict by retaining both AdjacentExact and Amplin coverage, and
pushing the merge.

The pipe3 experiment showed that kernel correctness is not enough when unrelated contexts contaminate
microsecond timings. The two Amplin performance entrypoints now run a shared stdlib-only gate before importing
Torch, initializing CUDA, compiling an extension, or allocating device memory. Formal timing defaults to:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<exactly one physical GPU UUID>
3 consecutive samples
required utilization: 0%
maximum unexplained driver baseline: 16 MiB
foreign compute processes: none
```

The thresholds are configurable, while `--skip-gpu-idle-preflight` explicitly marks a run as non-formal. After
each shape's warmup and synchronization, the timing helper re-resolves the physical ID/PCI/UUID, rejects a
foreign PID, and verifies that total residency is the current process's reported allocation plus no more than
the configured driver baseline. The post-warmup check reports utilization but does not require 0%, because the
benchmark's own just-completed warmup legitimately occupies the utilization sampling window.

The first direct-entrypoint smoke failed before Torch or CUDA import:

```text
ModuleNotFoundError: No module named 'gptqmodel'
kernel launches: 0
```

The helper directory had been added to `sys.path`, but the direct script entrypoint still needed the repository
root. Both Amplin scripts now add the explicit script and repository roots before the gated post-bootstrap
imports. `--help` bypasses the hardware query because it cannot produce a timing result.

CPU-only validation:

```text
ruff: pass
py_compile: pass
pytest tests/test_gpu_idle_preflight.py: 7 passed
git diff --check: pass
```

The focused tests cover three-sample acceptance, physical UUID pinning, foreign-process rejection before CUDA,
current-PID memory attribution after warmup, unexplained-memory rejection, the minimum sample count, help
behavior, and a foreign process arriving after warmup.

Live PCI-bus inventory re-resolved the requested devices:

```text
physical GPU 1  PCI 00000000:2B:00.0  GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855  PG506-232  sm_80
physical GPU 2  PCI 00000000:64:00.0  GPU-8be4c651-4058-83df-154b-291f1b86add8  PG506-230  sm_80
```

Both physical GPUs passed independent pre-CUDA probes with three consecutive `0% / 0 MiB` samples. Direct
entrypoint smokes then passed dense-FP32-dequant correctness and both validity gates:

```text
GPU  entrypoint                         initial gate     pre-timing ownership
 1   benchmark_amplin_vs_marlin.py      3 x 0% / 0 MiB  own 946 MiB + driver 9 MiB, foreign 0
 2   benchmark_amplin_model_shapes.py   3 x 0% / 0 MiB  own 674 MiB + driver 9 MiB, foreign 0
```

The smokes used only two timed launches and one round, so their latency values are not retained as performance
evidence.

Artifacts:

```text
/tmp/amplin-idle-gate-smoke-gpu1.json
size 8403 bytes
sha256 1e63bf093eefd37e69e4f5e002e80cb5ca3c7abf9924d3fceba906c2e955f52d

/tmp/amplin-idle-gate-smoke-gpu2.json
size 9232 bytes
sha256 18936d7c204160a9ed38354de9c56ac66f8bce52dcd4ccd4be86cf5b7436d373
```

Decision:

- Retain the strict gate and post-warmup exclusivity check in both Amplin timing entrypoints.
- Treat any initial or pre-timing rejection as an invalid benchmark, without waiting, killing a process, or
  substituting another GPU.
- Continue all formal Amplin-versus-Marlin timing only on UUID-pinned physical GPUs 1 and 2.

Result: benchmark validity infrastructure success, including one recorded pre-CUDA startup failure and its
validated correction. The next large-shape experiment can now fail closed instead of publishing contaminated
latencies.

### 2026-07-24 — Failed experiment: K24 intra-CTA ownership for N32 large-K tiles

Revision before experiment: `faf178ed`, with the temporary K24 candidate in the working tree.

The newly merged ParoQuant sm_80 work was inspected for mechanisms that could transfer to Amplin. Its successful
large-K schedules expose more independent split-K work, while its two-output reuse, second packed-weight
prefetch, and producer-warp metadata staging controls were measured losses. Amplin's bounded analogue kept the
existing packed-int32/scales layout, N32 output tile, MMA instruction mapping, FP32 partials, one-launch contract,
and final reduction order, but changed one ownership fact:

```text
retained K12 pipe2: 12 warps, 64 K16 steps per warp, 384 threads
candidate K24:      24 warps, 32 K16 steps per warp, 768 threads
```

K=12288 contains 96 group-128 blocks, so each K24 warp owned four complete groups. The 24-warp partial buffer
used exactly 48 KiB of static shared memory. The candidate remained an explicit research operator with no
automatic backend route.

JIT fingerprint `555940f78eb701fd` compiled on physical GPU 1. Generated-resource inspection showed:

```text
schedule       FP16 regs  BF16 regs  static shared  local  stack
K12 pipe2          56         56        24 KiB         0      0
K24 candidate      54         54        48 KiB         0      0
```

Exact Qwen K12288xN4096 and Laguna K12288xN3072 correctness passed on UUID-pinned physical GPUs 1 and 2. Each
device run covered FP16/BF16, M=2/4/8/16, a non-default stream, and ten schedules: 160 independent
FP32-dequant comparisons per device. Both runs reported four passed tests and 16 warnings.

Formal timing then ran sequentially on both requested devices. Both initial gates observed three consecutive
`0% / 0 MiB` samples, and all pre-timing rechecks found only the benchmark PID plus a 9 MiB driver baseline.
Each table cell used 50 warmups, 200 launches per batched CUDA-event round, and seven rounds. The slash-separated
columns are M=2/4/8/16 batch-event medians in microseconds:

```text
GPU  shape   dtype  K24 candidate                 K12 pipe2                     Marlin
 1   Qwen    FP16   34.688/35.215/36.306/50.560  25.027/26.220/30.346/46.710  23.276/23.613/23.818/25.354
 1   Qwen    BF16   36.659/37.376/38.548/49.725  26.045/26.952/30.597/45.937  24.274/24.934/24.755/27.162
 1   Laguna  FP16   19.016/19.174/20.434/27.960  17.316/18.668/21.243/29.834  24.586/22.743/24.914/23.398
 1   Laguna  BF16   20.019/20.321/21.371/28.124  17.761/18.964/20.966/29.696  25.441/23.055/24.346/24.806
 2   Qwen    FP16   34.586/35.036/36.132/50.550  25.011/26.179/30.223/46.546  23.229/25.037/22.369/26.220
 2   Qwen    BF16   36.577/37.151/38.641/50.007  26.025/26.941/30.623/45.937  24.320/24.653/24.008/27.167
 2   Laguna  FP16   19.041/19.185/20.454/27.991  17.290/18.560/21.140/30.116  21.929/22.794/21.857/23.921
 2   Laguna  BF16   20.019/20.372/21.366/28.191  17.746/18.888/20.941/29.844  22.697/23.137/21.801/24.868
```

The requested M=1 control was also timed in the same processes. K24 is intentionally not legal at M=1; canonical
Amplin/Marlin medians were:

```text
GPU  shape   dtype  canonical Amplin us  Marlin us
 1   Qwen    FP16          38.835          23.388
 1   Qwen    BF16          45.737          27.054
 1   Laguna  FP16          25.590          23.798
 1   Laguna  BF16          26.020          24.305
 2   Qwen    FP16          38.979          23.357
 2   Qwen    BF16          39.414          24.878
 2   Laguna  FP16          25.580          22.717
 2   Laguna  BF16          25.994          21.560
```

K24 loses every Qwen M=2/4/8/16 cell to K12 pipe2 and Marlin on both GPUs. It is 8-41% slower than K12 pipe2
and 29-50% slower than Marlin there. More warps do help Laguna FP16 M8 by 3.4-4.0% over K12 pipe2 and improve
Laguna M16 over the retained Amplin controls, but M16 still loses Marlin by 11.8-16.3%; Laguna BF16 M8 also
regresses about 2% versus K12 pipe2. This does not solve the user-selected large-shape gap and is not a
cross-dtype niche worth a 768-thread/48-KiB specialization.

Artifacts:

```text
/tmp/amplin-n32-k24-gpu1-r7.json
size 332928 bytes
sha256 5329643de4cbb25ce1589287a51273f1124b8a103dbd6a74dda8e36ddf4628ea

/tmp/amplin-n32-k24-gpu2-r7.json
size 332884 bytes
sha256 9f6613437faef830df058030babc7e7b4cb74ce91927d0ccfb9d3a5b5c785a5e
```

Decision:

- Reject and remove the K24 kernel, operator, wrapper, benchmark flag, and correctness rows.
- Keep K12 pipe2 as the retained large-K N32 control.
- Do not profile K24 with NCU: it has no Qwen timing win for profiler replay to explain.
- Do not infer that more split-K parallelism is generally bad; the failure is specifically intra-CTA K24 with
  doubled shared partials and warp-0 reduction work.

Result: correct, spill-free, and decisively slower on the target Qwen shape. Simply doubling intra-CTA warp
ownership does not hide the packed-weight dependency enough to pay for the larger shared reduction.

### 2026-07-24 — Successful experiment: Ampere lane-pair N32 int32 weight layout

Revision before experiment: `647521a8`, with the candidate implemented only in the working tree.

This pass tested the original data-layout hypothesis directly instead of changing CTA ownership again. The
retained K12 pipe2 kernel fetched the two N16 words used by one lane from locations 32 int32 values (128 bytes)
apart:

```text
retained:  [N64, K128, K16, N16-tile-4, lane-32]
candidate: [N32, K128, K16, lane-32, word-pair-2]
```

The candidate stores the same int32 payload and uses the same packed scales, N32 output tile, 12 split-K warps,
384 threads, K-step pipeline, MMA/dequant math, FP32 partial reduction, and one-launch contract. Only the
qweight order changes. Each lane's two words are now adjacent and 8-byte aligned, so one `uint2` load replaces
two independent 32-bit loads. The candidate remains an explicit research operator with no automatic backend
route; canonical qweight can be restored exactly by the inverse packer.

JIT fingerprint `91d8f6e1b85a72ca` compiled on UUID-pinned physical GPU 1. Generated sm_80 SASS confirmed that
the candidate emits `LDG.E.64.CONSTANT` for the lane-local pair, while the retained pipe2 control emits separate
`LDG.E.CONSTANT` weight loads. Resource usage was unchanged:

```text
schedule                    FP16 regs  BF16 regs  static shared  local  stack
K12 pipe2 retained              56         56        24 KiB         0      0
K12 pipe2 interleaved           56         56        24 KiB         0      0
```

CPU pack/unpack ownership and API coverage passed as part of 24 focused tests. Exact Qwen K12288xN4096 and
Laguna K12288xN3072 CUDA correctness passed on physical GPUs 1 and 2. Each device covered FP16/BF16,
M=2/4/8/16, a non-default stream, and ten schedules: 160 FP32-dequant comparisons per device. Both device runs
reported four passed tests and 16 warnings.

Formal timing ran sequentially on the two requested devices. Each initial gate observed three consecutive
`0% / 0 MiB` samples. Every pre-timing recheck found only the benchmark PID plus a 9 MiB driver baseline. Each
cell used 50 warmups, 200 launches per batched CUDA-event round, and seven rounds. Slash-separated values below
are M=2/4/8/16 batch-event medians in microseconds:

```text
GPU  shape   dtype  interleaved candidate             retained K12 pipe2                Marlin
 1   Qwen    FP16   24.643/25.580/29.650/45.839      25.098/26.214/30.346/46.756      23.209/23.567/22.451/25.344
 1   Qwen    BF16   25.856/26.670/29.716/46.039      26.107/27.023/30.694/46.397      24.356/24.740/23.603/27.218
 1   Laguna  FP16   16.973/18.345/20.572/29.030      17.357/18.744/21.289/30.167      22.021/22.395/25.492/25.687
 1   Laguna  BF16   17.275/18.360/20.429/29.292      17.807/19.057/21.048/29.947      22.876/23.982/21.996/25.073
 2   Qwen    FP16   24.504/25.513/29.558/45.885      24.965/26.132/30.479/46.925      23.107/27.213/22.333/25.569
 2   Qwen    BF16   25.692/26.552/29.716/46.152      25.989/26.911/30.930/46.346      26.245/24.673/24.975/27.238
 2   Laguna  FP16   16.845/18.176/20.383/29.071      17.300/18.555/21.084/29.967      22.308/22.252/26.481/25.559
 2   Laguna  BF16   17.147/18.248/20.306/29.317      17.654/18.862/20.966/29.804      22.794/24.765/22.851/26.189
```

The candidate improves all 32 paired cells against the identical retained control: minimum 0.42%, arithmetic
mean 2.45%, and maximum 4.08%. It beats Marlin in 14 of 32 cells, including every Laguna M2/M4 cell and every
Laguna M8 cell. It does not solve Qwen M8/M16 or Laguna M16, but the uniform cross-device control win proves
that the legacy int32 order was leaving an Ampere load-width opportunity unused.

The requested M=1 control was included in the same processes. This M16/N32 research kernel is intentionally
not legal at M=1, so canonical Amplin and Marlin were compared:

```text
GPU  shape   dtype  canonical Amplin us  Marlin us
 1   Qwen    FP16          38.866          22.216
 1   Qwen    BF16          39.706          23.378
 1   Laguna  FP16          25.656          21.560
 1   Laguna  BF16          26.056          25.390
 2   Qwen    FP16          45.000          25.871
 2   Qwen    BF16          45.798          27.105
 2   Laguna  FP16          25.544          22.523
 2   Laguna  BF16          25.953          21.396
```

Artifacts:

```text
/tmp/amplin-n32-interleaved-gpu1-r7.json
size 333353 bytes
sha256 baf1daf5e36dec5573ebd0feffd7869087c58a8f19d44e5a66c518d11eacc75c

/tmp/amplin-n32-interleaved-gpu2-r7.json
size 333353 bytes
sha256 4b574621d65e0161a3b65f99b771d1ceceed676b7fc253c4fcbb27caebc98f29
```

Decision:

- Retain the explicit N32 lane-pair packer, inverse packer, K12 pipe2 research operator, correctness coverage,
  and formal benchmark flag.
- Do not change automatic routing yet. The candidate requires a different packed representation, and the
  production policy needs a persistent packed-buffer/lifecycle decision rather than a timed-path repack.
- Use the cross-device win as the new data-layout baseline for subsequent large-shape work.
- Continue focusing the next schedule experiment on the unresolved Qwen N4096 and M16 gaps, without adding
  shared-memory or register pressure to this spill-free layout.

Result: first successful large-shape layout redesign. Ampere benefits measurably from lane-adjacent int32 pairs
even when the quantized values, scales, CTA shape, and MMA schedule are otherwise identical.

### 2026-07-24 — Success: N64/K24 vector layout closes most of the Qwen M16 gap

Revision before experiment: `9680fe3a`, with the temporary candidate in the working tree.

The successful N32 lane-pair layout showed that execution-native int32 order is a real Ampere lever, but Qwen
N4096 M8/M16 still lost badly to Marlin because every N32 CTA reread the same activation tile. The next bounded
mega-kernel schedule combined wider output ownership with more K owners:

```text
property                    N32/K12 interleaved        N64/K24 candidate
qweight layout              [N32,G,K16,lane,2]         [N64,G,K16,lane,4]
lane qweight load           aligned uint2 / 64 bit     aligned uint4 / 128 bit
output tile                 M16xN32                    M16xN64
split-K warps / threads     12 / 384                   24 / 768
Qwen / Laguna grid          128 / 96 CTAs              64 / 48 CTAs
activation reads            one per N32 tile           one per N64 tile
FP32 partial storage        24 KiB                     96 KiB dynamic
reduction ownership         warp 0 handles 4 fragments warps 0-7 handle one fragment each
```

K12288 has 96 group-128 blocks, so each K24 warp owns four complete groups. The larger tile halves repeated
activation traffic while preserving total qweight bytes and aggregate launched warps for Qwen. The 96 KiB
partial buffer is explicitly checked against `sharedMemPerBlockOptin` and configured with
`cudaFuncAttributeMaxDynamicSharedMemorySize`. The candidate is a separate research operator; no fallback or
automatic route changed.

JIT fingerprint `c252797ce8c0ecec` compiled on physical GPU 1. Generated sm_80 code showed:

```text
dtype  threads  registers/thread  dynamic shared  local  stack  static qweight loads
FP16     768           80             96 KiB          0      0    8 x LDG.E.128.CONSTANT
BF16     768           80             96 KiB          0      0    8 x LDG.E.128.CONSTANT
```

The 61,440-register block fits the 65,536-register sm_80 block budget without spills. Both dtype kernels
successfully opt in to 96 KiB shared memory on the target cards.

Exact Qwen K12288xN4096 and Laguna K12288xN3072 correctness passed on physical GPUs 1 and 2. Each device
covered FP16/BF16, M=2/4/8/16, a non-default stream, and eleven schedules: 176 FP32-dequant comparisons per
device. Both runs reported four passed tests and 16 warnings.

Formal timing ran sequentially on both UUID-pinned devices. Both initial gates observed three consecutive
`0% / 0 MiB` samples, and every pre-timing recheck found only the benchmark PID plus a 9 MiB driver baseline.
Each cell used 50 warmups, 200 launches per batched CUDA-event round, and seven rounds. Slash-separated values
are M=2/4/8/16 medians in microseconds:

```text
GPU  shape   dtype  N64/K24 candidate                 N32/K12 interleaved               Marlin
 1   Qwen    FP16   26.527/26.639/26.839/28.928      24.658/25.646/29.660/46.321      24.658/28.503/25.052/25.728
 1   Qwen    BF16   27.837/27.930/28.109/30.121      25.887/26.706/29.757/46.444      24.397/24.776/25.728/27.479
 1   Laguna  FP16   26.496/26.593/26.931/29.153      17.039/18.412/20.603/29.476      22.743/22.380/23.332/23.803
 1   Laguna  BF16   27.776/27.924/28.257/30.336      17.321/18.417/20.511/29.655      22.866/23.178/23.378/25.160
 2   Qwen    FP16   26.368/26.450/26.947/28.948      24.535/25.518/29.855/46.653      24.422/23.603/22.743/25.569
 2   Qwen    BF16   27.617/27.822/28.201/30.008      25.718/26.604/29.932/46.541      24.305/24.643/23.757/27.356
 2   Laguna  FP16   26.312/26.440/26.819/29.169      16.870/18.176/20.439/29.501      22.948/22.605/22.666/23.752
 2   Laguna  BF16   27.571/27.791/28.099/30.269      17.172/18.268/20.250/29.343      22.789/23.357/22.088/25.334
```

The schedule crossover reproduces on both devices and dtypes:

```text
regime                         N64/K24 outcome
Qwen M2/M4                     loses N32; do not select
Qwen M8                        wins N32 by 5.9-10.8%
Qwen M16                       wins N32 by 54.2-60.9%
Laguna M2/M4/M8                loses badly because 48 CTAs underfill 124 SMs
Laguna M16                     approximately tied in FP16, loses in BF16; keep N32
```

At Qwen M16, N64/K24 narrows Amplin from approximately 46.3 us to 28.9 us in FP16 and from approximately
46.5 us to 30.1 us in BF16. Marlin remains faster at approximately 25.6 us and 27.4 us. GPU-1 Marlin ranges
contain several noisy outliers, so no overlapping-range Marlin reversal is promoted; GPU 2 cleanly favors
Marlin. This is an Amplin schedule success, not an Amplin-over-Marlin win.

The requested M=1 control was included in the same processes. N64/K24 is intentionally illegal at M=1;
canonical Amplin/Marlin medians were:

```text
GPU  shape   dtype  canonical Amplin us  Marlin us
 1   Qwen    FP16          39.009          23.813
 1   Qwen    BF16          45.604          26.522
 1   Laguna  FP16          25.636          21.074
 1   Laguna  BF16          26.071          21.489
 2   Qwen    FP16          44.954          25.928
 2   Qwen    BF16          39.475          26.967
 2   Laguna  FP16          25.539          22.830
 2   Laguna  BF16          25.979          22.636
```

Artifacts:

```text
/tmp/amplin-n64-k24-gpu1-r7.json
size 206180 bytes
sha256 af6d0fbb0f71655c37fdc14081b084ef28ab85048aeb966ff7710d61a46032da

/tmp/amplin-n64-k24-gpu2-r7.json
size 206245 bytes
sha256 627e0bb6573e00e5fb09e3980a499d5d1f52c764689ead660f96967a173e2037
```

Decision:

- Retain the explicit N64 lane-quad packer/inverse, N64/K24 operator, real-shape correctness coverage, and
  formal benchmark flag.
- Treat it as a Qwen-like N4096, M8/M16 schedule only. N32 interleaved remains the Laguna and low-M control.
- Do not route automatically yet: the alternate packed buffer needs an owned lifecycle, and Marlin still wins
  the target Qwen M16 cells.
- Profile Qwen M16 N64/K24 against Marlin next. The remaining 2.7-3.4 us gap is now small enough for
  instruction, reduction, and memory-stall attribution to guide a narrower optimization.

Result: the first wide-tile Amplin mega-kernel schedule succeeds where isolated K24 failed. K24 becomes useful
only when its added warp supply is paired with N64 activation reuse and a 128-bit execution-native qweight
layout.

### 2026-07-24 — Success: matched N64/K24 and Marlin profile isolates CTA underfill

Revision before profiling: `71deeb60`, with the profiling-harness changes in the working tree.

The profiler entrypoint now exposes the retained interleaved N32/K12 and N64/K24 operators. It also runs the
same stdlib-only physical-GPU idle gate used by the timing harness before importing Torch, repeats the
exclusivity and memory-ownership check after warmup, and records both checks in the emitted metadata. This
prevents a diagnostic replay from silently using a co-resident or incorrectly indexed GPU.

A direct BF16 smoke run on UUID-pinned physical GPU 2 passed the FP32-dequant reference at exact Qwen
`M16 K12288 N4096`:

```text
path                                    max abs error  mean abs error
N64/K24 pipe2 interleaved                   0.008178       0.001073
```

The initial gate observed three consecutive `0% / 0 MiB` samples. The post-warmup smoke recheck found no
foreign process and attributed 1,676 of 1,685 MiB to the benchmark PID, leaving the expected 9 MiB driver
baseline. Ruff, Python compilation, and `git diff --check` passed.

The first FP16 smoke invocation for the newly exposed N32 path failed before profiling because this older
entrypoint still enforced a `0.001` maximum-error cutoff:

```text
AssertionError: n32_splitk12_pipe2_interleaved correctness failed:
finite=True, max_abs=0.0010743141174316406, limit=0.001
```

The kernel test and real-shape benchmark both use the established FP16/BF16 absolute tolerances of
`0.002/0.02`. The profiler was corrected to use those same limits rather than inventing a stricter,
entrypoint-specific threshold. The exact GPU-1 N32 smoke invocation was then repeated and passed with
`max_abs=0.001074` and `mean_abs=0.000134`.

Nsight Compute 2025.3.1 collected one launch and 14 replay passes per kernel on physical GPU 2:

```text
ncu --target-processes all --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section WarpStateStats --section SchedulerStats \
  --section MemoryWorkloadAnalysis --launch-count 1 \
  --export /tmp/<report> -- \
  /usr/bin/env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-8be4c651-4058-83df-154b-291f1b86add8 \
  TORCH_CUDA_ARCH_LIST=8.0 \
  python -m scripts.profile_amplin_vs_marlin \
    --device cuda:0 --path <n64_splitk24_pipe2_interleaved-or-marlin> \
    --dtype bf16 --m 16 --k 12288 --n 4096 \
    --warmup 50 --launches 1 --seed 20260724 --cuda-profiler-api
```

Both captured processes passed the strict initial idle gate and the post-warmup ownership recheck with no
foreign process and only 9 MiB unattributed. The profiler-replay duration is diagnostic and is not substituted
for the seven-round CUDA-event selection result:

```text
metric                                Amplin N64/K24       Marlin
grid / block                              64 / 768         124 / 128
waves per 124-SM device                       0.52               1.00
registers per thread                            80                138
dynamic shared per CTA                    98.304 KiB        166.912 KiB
theoretical / achieved occupancy        37.50/36.75%        6.25/6.18%
active / eligible warps per scheduler     5.91 / 1.68        1.01 / 0.30
issued warps per scheduler                       0.56               0.30
cycles per issued instruction                   10.49               3.34
executed instructions                       5,770,752          4,775,856
SM / memory / DRAM throughput            27.06/36.19/30.24% 25.25/34.65/34.65%
L1/TEX / L2 throughput                   75.05/29.06%       15.29/32.42%
tensor / shared pipe active              14.71/14.71%       16.50/16.50%
replay duration                                35.968 us          32.064 us
```

Amplin's average 10.49 warp cycles between issues divide primarily among wait (2.09), math-pipe throttle
(2.05), not-selected (2.00), dispatch stall (1.28), selected issue (1.00), and long scoreboard (0.95).
It is neither spilling nor saturating DRAM. Higher occupancy and twice Marlin's eligible-warp supply do not
compensate for assigning only 64 N64 tiles to a 124-SM device. Marlin instead assigns exactly one CTA per SM
and executes about 17% fewer aggregate instructions.

The profile changes the next experiment from another register or prefetch tweak to an ownership correction.
An N32/K24 candidate can expose 128 CTAs at Qwen N4096 while preserving K24's four groups per producer warp.
Unlike the previously rejected N32/K24 control, it must combine the successful lane-pair interleaved layout
with fragment-distributed reduction warps; otherwise the old warp-0 reduction bottleneck is knowingly
reintroduced.

Artifacts:

```text
/tmp/amplin-n64-profile-smoke-gpu2.json
size 9624 bytes
sha256 a56cd634785e856b6624a247279e9d8337907cc600ddc6c073fec0c9f309c07a

/tmp/amplin-n32-profile-smoke-gpu1.json
size 5065 bytes
sha256 925e2f7243223780884327bfb40044529596a42491b66eb4f3e1c5570c0c8d7f

/tmp/amplin-qwen-bf16-m16-n64-k24-gpu2-20260724.ncu-rep
size 3458860 bytes
sha256 ed9f440ae980c7871684eb2eeff3abc605209317c0b97a7e2afb99d5c0afd5d7

/tmp/amplin-qwen-bf16-m16-marlin-gpu2-20260724.ncu-rep
size 20853630 bytes
sha256 b7d2b81eccf065bb8e125c8bb77abeb128a74be9237d28409dd0aa3f7524edc7
```

Decision:

- Retain the strict profiling gate and explicit interleaved-path controls.
- Keep the N64/K24 operator as the measured Qwen M8/M16 control, but do not route it automatically.
- Prototype N32/K24 only with the interleaved int32 layout and four-way distributed reduction.
- Keep formal schedule decisions on warmed CUDA-event results; use NCU replay only for attribution.

Result: diagnostic success. The remaining Qwen M16 deficit is primarily an ownership and instruction-work
problem, not evidence that N64/K24 needs more occupancy or another isolated global-load prefetch.

### 2026-07-24 — Failed experiment: interleaved N32/K24 with distributed reduction

Revision before experiment: `e2d948f5`, with the temporary candidate in the working tree.

The matched profile showed that N64/K24 launches only 64 CTAs for Qwen N4096. The bounded follow-up restored an
N32 output tile and 128-CTA grid while retaining the successful lane-pair int32 layout. It differed from the
previously rejected N32/K24 control by distributing its four output fragments across reduction warps 0-3:

```text
property                    retained N64/K24      candidate N32/K24
output tile                 M16xN64               M16xN32
Qwen / Laguna grid          64 / 48 CTAs          128 / 96 CTAs
producer warps / threads    24 / 768              24 / 768
groups per producer warp    4                     4
lane qweight load           uint4 / 128 bit       uint2 / 64 bit
FP32 partial storage        96 KiB                48 KiB
reduction ownership         warps 0-7             warps 0-3
```

The candidate remained a separate explicit research operator with no automatic route. JIT fingerprint
`6659c2567773c4d0` compiled on physical GPU 1. Generated sm_80 resource usage was spill-free:

```text
dtype  registers/thread  static shared  local  stack  qweight load
FP16          64            48 KiB         0      0    LDG.E.64.CONSTANT
BF16          66            48 KiB         0      0    LDG.E.64.CONSTANT
```

Focused correctness passed on UUID-pinned physical GPUs 1 and 2. Each device ran all eight exact large-MLP
shape/dtype cases: Qwen up/down, Laguna up/down, FP16/BF16, M=2/4/8/16, and a non-default stream. The two
K12288 target shapes covered 192 FP32-dequant comparisons per device after adding the candidate. Each device
reported eight passed tests and 16 warnings.

Because the primary Qwen gate was a large regression, timing intentionally stopped at a short two-device
rejection gate rather than spending the formal seven-round budget. Both UUID-pinned devices observed three
initial `0% / 0 MiB` samples and every post-warmup recheck found no foreign process with only 9 MiB
unattributed. Every cell used 50 warmups, 100 launches per batched CUDA-event round, and three rounds:

```text
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 1,2,4,8,16 --dtype both \
  --splitk12-n32-interleaved \
  --splitk24-n32-interleaved \
  --splitk24-n64-interleaved \
  --warmup 50 --iters 100 --rounds 3 \
  --json-out /tmp/amplin-n32-k24-distributed-gpu<1-or-2>-r3.json
```

Slash-separated medians are M=2/4/8/16 in microseconds:

```text
GPU  shape   dtype  N32/K24 candidate       N32/K12 retained        N64/K24 retained        Marlin
 1   Qwen    FP16   40.468/35.543/37.806/48.589  28.774/25.620/29.676/46.469  30.771/26.634/26.829/28.754  27.228/23.839/24.054/25.743
 1   Qwen    BF16   41.677/36.895/38.932/48.333  29.809/26.685/29.706/46.080  31.846/27.924/28.037/30.044  28.180/24.822/23.634/27.474
 1   Laguna  FP16   19.180/19.804/21.576/26.378  16.957/18.319/20.613/29.184  26.429/26.563/26.890/28.979  24.115/22.620/22.016/23.716
 1   Laguna  BF16   19.763/20.552/21.699/26.388  17.244/18.360/20.470/29.266  27.720/27.914/28.201/30.044  23.316/23.183/23.112/25.016
 2   Qwen    FP16   40.581/35.338/37.734/48.323  28.846/25.631/29.655/46.787  30.935/26.563/26.839/29.000  27.412/23.757/23.532/25.825
 2   Qwen    BF16   41.738/36.731/38.810/48.681  29.880/26.655/29.686/46.848  31.949/27.832/28.078/30.280  28.273/24.873/23.777/27.996
 2   Laguna  FP16   19.190/19.866/21.627/26.522  16.998/18.309/20.480/29.573  26.399/26.542/26.860/29.225  22.272/23.235/22.170/24.003
 2   Laguna  BF16   19.814/20.623/21.740/26.808  17.285/18.371/20.367/29.460  27.720/27.873/28.150/30.464  25.057/23.665/26.511/25.559
```

At Qwen M16, N32/K24 is 61-69% slower than retained N64/K24 and 74-89% slower than Marlin across both
devices and dtypes. It even remains 3-5% slower than N32/K12 despite halving each producer warp's K ownership.
The larger grid therefore does not repay the doubled CTA-level fixed work, duplicated activation access,
64-bit rather than 128-bit qweight issue, and larger aggregate reduction schedule.

Laguna M16 improves 10-12% over N32/K12, but still loses Marlin by 5-12% in all four device/dtype cells.
At M2/M4/M8 it loses the retained N32 schedule. This is not a target-wide or cross-dtype niche worth another
operator. M=1 was included as requested but the candidate is intentionally illegal; canonical Amplin/Marlin
medians were:

```text
GPU  shape   dtype  canonical Amplin M1  Marlin M1
 1   Qwen    FP16          45.179          25.928
 1   Qwen    BF16          46.141          27.412
 1   Laguna  FP16          25.661          22.272
 1   Laguna  BF16          26.081          22.333
 2   Qwen    FP16          45.087          26.214
 2   Qwen    BF16          45.660          27.249
 2   Laguna  FP16          25.610          22.938
 2   Laguna  BF16          26.071          23.808
```

Artifacts:

```text
/tmp/amplin-n32-k24-distributed-gpu1-r3.json
size 232215 bytes
sha256 6544b0d5c36315e840da447cc51bab99585eba30b54c5c577488781e71612ced

/tmp/amplin-n32-k24-distributed-gpu2-r3.json
size 232094 bytes
sha256 2e30b7f8948a84178ba534b19c58e24fec7e11b59f9001f24d931da9bb26df72
```

Decision:

- Reject and remove the N32/K24 kernel, operator, wrapper, benchmark flag, profiler path, and correctness rows.
- Do not run NCU or a seven-round gate for a candidate that is decisively slower on the primary Qwen target.
- Keep N32/K12 interleaved for Laguna and low-M Qwen; keep N64/K24 for Qwen M8/M16.
- The next ownership prototype must preserve N64 activation reuse and 128-bit qweight loads while exposing more
  than 64 CTAs. Splitting K across two N64 CTAs is the next hypothesis; simply narrowing N is now rejected twice.

Result: correct and spill-free failure. Filling the 124-SM device with N32/K24 CTAs does not beat the half-wave
N64 schedule, demonstrating that wider activation and vector-load reuse matter more than grid count alone.

### 2026-07-24 — Success: two cooperative N64/K12 CTAs accelerate Laguna M8/M16

Revision before experiment: `b04f8cc7`, with the candidate implemented in the working tree.

The failed N32/K24 experiment showed that filling the GPU by narrowing the output tile gives up too much N64
reuse. This candidate instead preserves the retained `[N64,K128,K16,lane,word-quad]` int32 layout and aligned
128-bit lane loads, but divides the 24 K owners across two cooperative CTAs:

```text
property                       N64/K24 retained       N64/K12x2 cooperative
output ownership               one M16xN64 tile       one M16xN64 tile
CTAs per N64 tile              1                      2
warps / threads per CTA        24 / 768               12 / 384
groups per producer warp       4                      4
Qwen / Laguna grid             64 / 48 CTAs           128 / 96 CTAs
shared FP32 partials per CTA    96 KiB                 48 KiB
cross-CTA state                none                   2 x M x N FP32 scratch
synchronization                CTA barrier            CTA barrier + grid barrier
```

Each CTA owns disjoint K groups. Eight reduction warps write one FP32 partial plane, the cooperative grid
synchronizes, and split-0 CTAs add the two planes and convert once to FP16/BF16. Scratch is 512 KiB for Qwen
M16 N4096 and 384 KiB for Laguna M16 N3072. The runtime:

- probes exact sm_80 and cooperative-launch support on the selected device;
- queries active cooperative residency and rejects a grid larger than the device-wide resident-CTA capacity;
- checks the ordinary 48 KiB shared-memory limit;
- rejects CUDA graph capture before allocating output or scratch;
- leaves all retained non-cooperative paths and automatic routing unchanged.

The final sm_80 JIT fingerprint is `496dba909eba0cfa`. The exact build uses
`-gencode=arch=compute_80,code=sm_80`, `-O3`, `--optimize=3`, `-Xptxas -O3,-dlcm=ca`, `-lineinfo`, BF16
enabled, and no fast-math flag. Generated resources are identical across dtypes:

```text
threads  registers/thread  static shared  local  stack  qweight load
  384           80            48 KiB         0      0    LDG.E.128.CONSTANT
```

The occupancy query and NCU launch data report two resident blocks per SM, so the 124-SM cards have a
248-CTA cooperative capacity. Qwen requires 128 CTAs and Laguna 96; both fit.

Focused correctness passed on physical GPUs 1 and 2. Each device ran all eight exact large-MLP shape/dtype
cases: Qwen up/down, Laguna up/down, FP16/BF16, M=2/4/8/16, and a non-default stream. The two K12288 target
shapes covered 192 FP32-dequant comparisons per device after adding the candidate. Both runs reported eight
passed tests and 16 warnings. The invalid-contract test also passed, and an explicit CUDA-graph capture probe
raised the intended error:

```text
Amplin N64 split-K12x2 cooperative launch does not support CUDA graph capture
```

Two GPU-1 timing processes were invalidated by the strict post-warmup gate and produced no accepted result.
Both had passed three initial idle samples, after which a foreign process appeared:

```text
attempt  requested gate       foreign PID  foreign memory
1        50/100/3 short gate    1078020       4894 MiB
2        50/200/7 formal gate   1082353       1322 MiB
```

Neither process was killed or waited on, and no other GPU was substituted for a GPU-1 result. GPU-1
correctness is valid; its performance cross-check remains pending.

Physical GPU 2 remained exclusive and completed the formal gate with three initial `0% / 0 MiB` samples and
only the benchmark PID plus 9 MiB unattributed at every post-warmup check:

```text
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 1,2,4,8,16 --dtype both \
  --splitk12-n32-interleaved \
  --splitk24-n64-interleaved \
  --splitk12x2-n64-coop \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-n64-k12x2-coop-gpu2-r7.json
```

Slash-separated medians are M=2/4/8/16 in microseconds:

```text
shape   dtype  N64/K12x2 cooperative  N64/K24 retained     N32/K12 retained     Marlin
Qwen    FP16   29.010/29.225/30.131/32.543  26.368/26.511/26.993/28.964  24.504/25.487/29.865/47.068  23.296/23.567/22.548/27.141
Qwen    BF16   30.259/30.490/31.314/33.746  27.622/27.832/28.221/30.008  25.718/26.563/29.947/47.078  24.371/25.037/25.626/27.412
Laguna  FP16   18.371/18.493/19.128/20.449  26.296/26.460/26.839/29.112  16.855/18.196/20.444/29.430  24.356/28.590/24.422/23.926
Laguna  BF16   18.447/18.693/19.379/21.228  27.551/27.796/28.232/30.239  17.198/18.304/20.285/29.363  22.697/23.020/21.734/25.139
```

The cooperative schedule loses every Qwen cell to N64/K24 by approximately 10-12%; it is not a Qwen route.
For Laguna, retained N32 remains faster at M2 and the M4 ranges overlap. At M8/M16 the cooperative path beats
N32 with disjoint ranges in both dtypes. It also beats Marlin:

```text
Laguna regime  FP16 Marlin/cooperative  BF16 Marlin/cooperative  range outcome
M8                    1.277x                   1.122x             disjoint wins
M16                   1.170x                   1.184x             disjoint wins
```

M=1 was included as requested but the candidate is intentionally illegal. Canonical Amplin/Marlin medians were
44.974/25.964 us Qwen FP16, 45.732/27.284 us Qwen BF16, 25.574/23.992 us Laguna FP16, and
25.979/21.443 us Laguna BF16.

Nsight Compute 2025.3.1 captured one 14-pass replay per Laguna BF16 M16 path on physical GPU 2. Replay duration
is diagnostic only:

```text
metric                                  N64/K12x2 coop   N32/K12 retained   Marlin
grid / block                                96 / 384          96 / 384      124 / 128
registers / static shared                 80 / 48 KiB        56 / 24 KiB    138 / 0
achieved occupancy                             18.52%             18.48%         6.17%
active / eligible warps per scheduler       3.21/0.54          3.00/0.22      1.03/0.27
issued warps per scheduler                        0.38               0.19           0.27
cycles per issued instruction                     8.48              15.68           3.84
long-scoreboard cycles per issue                   2.46              11.46           0.29
barrier cycles per issue                           0.95               0.18           0.53
executed instructions                         4,638,090          3,423,264      3,634,644
SM / memory / DRAM throughput             26.35/35.83/28.17% 13.21/40.80/18.91% 20.40/27.16/27.16%
tensor-pipe active                              14.01%              9.07%         13.11%
replay duration                                28.896 us          42.976 us       29.984 us
```

Cooperative N64 pays 35% more executed instructions, FP32 scratch traffic, and a much larger barrier stall than
N32. It nevertheless cuts the dominant long-scoreboard delay by 4.7x and doubles the issue rate. The N64
128-bit layout plus four groups per warp is the mechanism; occupancy is effectively identical to N32.

Artifacts:

```text
/tmp/amplin-n64-k12x2-coop-gpu2-r7.json
size 251699 bytes
sha256 1c5b1d4a5c749b6acbde1d2e6258b3e1920a2e12d45ae170aa7d72213d709133

/tmp/amplin-laguna-bf16-m16-n64-k12x2-coop-gpu2-20260724.ncu-rep
size 3671267 bytes
sha256 fea14c122f16e76aa7e37170732f6a820a8381c7fff9a785413080ca27341876

/tmp/amplin-laguna-bf16-m16-n32-k12-gpu2-20260724.ncu-rep
size 3662997 bytes
sha256 603671b9116b5179a84cee085372e3690cb67276c826be15cf9fcf5d1b858be5

/tmp/amplin-laguna-bf16-m16-marlin-gpu2-20260724.ncu-rep
size 20852454 bytes
sha256 a50c1c44e49cbe4af668cb1a6b689779652c1fa459caacf14b9b8c96d9dedbe3
```

Decision:

- Retain the explicit cooperative N64/K12x2 operator, correctness coverage, benchmark flag, and profiler path.
- Treat it only as a Laguna-like N3072 M8/M16 research schedule; keep N32 for Laguna M2/M4 and N64/K24 for
  Qwen M8/M16.
- Do not add automatic routing. The alternate packed buffer and FP32 scratch need an owned lifecycle, CUDA
  graphs require a non-cooperative fallback, and a clean GPU-1 performance cross-check is still required.
- On the next pass, retry GPU 1 only if the strict initial and post-warmup gates independently pass.

Result: provisional schedule success on GPU 2, with cross-device correctness and a clean profiler explanation.
Preserving N64 reuse while splitting K across CTAs works for Laguna M8/M16; narrowing N does not.

### 2026-07-24 — Success: clean GPU-1 timing confirms the Laguna schedule

Revision: `200fdc57`.

The next GPU-1 retry found the physical PCI-order device idle for all three initial samples and retained
exclusivity through every timed path:

```text
physical id   PCI bus           UUID                                      device
1             00000000:2B:00.0  GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855  NVIDIA PG506-232

initial samples              3 x 0% utilization, 0 MiB
post-warmup foreign PIDs     0
post-warmup unattributed     9 MiB
```

The command matched the accepted GPU-2 formal gate:

```text
python -m scripts.benchmark_amplin_model_shapes \
  --shape 12288x4096 --shape 12288x3072 \
  --m-values 1,2,4,8,16 --dtype both \
  --splitk12-n32-interleaved \
  --splitk24-n64-interleaved \
  --splitk12x2-n64-coop \
  --warmup 50 --iters 200 --rounds 7 \
  --json-out /tmp/amplin-n64-k12x2-coop-gpu1-r7-retry.json
```

Slash-separated medians are M=2/4/8/16 in microseconds:

```text
shape   dtype  N64/K12x2 cooperative       N64/K24 retained           N32/K12 retained           Marlin
Qwen    FP16   29.025/29.245/29.798/32.568  26.414/26.506/26.706/28.836  24.576/25.544/29.568/46.080  23.660/24.535/22.492/25.370
Qwen    BF16   30.264/30.490/31.022/33.787  27.689/27.812/27.955/29.957  25.743/26.609/29.660/46.490  24.315/24.581/23.516/27.960
Laguna  FP16   18.309/18.452/18.949/20.413  26.348/26.460/26.813/29.036  16.850/18.248/20.480/29.348  22.226/22.241/24.515/23.552
Laguna  BF16   18.427/18.637/19.359/21.181  27.638/27.807/28.093/30.259  17.178/18.263/20.372/29.522  22.682/22.994/23.419/24.986
```

The cross-device decision is unchanged:

```text
Laguna regime  GPU-1 FP16 speedup  GPU-1 BF16 speedup  GPU-2 FP16 speedup  GPU-2 BF16 speedup
M8                    1.294x              1.210x              1.277x              1.122x
M16                   1.154x              1.180x              1.170x              1.184x
```

GPU 1 confirms the median win on all four Laguna M8/M16 cells. GPU-1 round ranges are noisier: FP16 M8 and
BF16 M16 remain disjoint from Marlin, while FP16 M16 and BF16 M8 contain an outlier and overlap Marlin's
range. The retained N32 path still wins Laguna M2/M4, and the cooperative path still loses every Qwen cell.
Therefore the explicit research route remains Laguna-like N3072 M8/M16 only, with no automatic selection.

M=1 remained in the run but is intentionally outside the cooperative contract. Canonical Amplin/Marlin
medians were 38.927/25.779 us Qwen FP16, 45.619/27.008 us Qwen BF16, 25.590/22.912 us Laguna FP16, and
26.015/21.484 us Laguna BF16.

Artifact:

```text
/tmp/amplin-n64-k12x2-coop-gpu1-r7-retry.json
size 251643 bytes
sha256 8929f9c584522307dedcfdf68b68972b4558956e7284c834814c22310cdba7ea
```

Result: accepted success. Both permitted devices now independently reproduce the Laguna M8/M16 median win,
while Qwen remains a clear loss and is not a candidate for this schedule.


## 2026-07-24 split-K8 N32 pipe2 sweep and split-K8 N32 pipe2 interleaved experiment

**Revision:** `fbfeefd3`  
**Device:** NVIDIA PG506-230 (GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2), compute capability 8.0, 124 SMs  
**Config:** GPTQ W4 group_size=128 sym=True desc_act=False, BF16, batch-event median  

### `mma_lane_m16_n32_splitk8_pipe2` results (new op) vs Marlin

| model | role | K | N | M | median (us) | speedup vs Marlin | max abs err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glm-5.2 | dense-down | 12288 | 6144 | 1 | 36.37 | 0.852 | 0.007304 |
| glm-5.2 | dense-down | 12288 | 6144 | 2 | 36.59 | 0.871 | 0.007304 |
| glm-5.2 | dense-down | 12288 | 6144 | 4 | 37.03 | 0.870 | 0.008194 |
| glm-5.2 | dense-down | 12288 | 6144 | 6 | 37.34 | 0.869 | 0.008194 |
| glm-5.2 | dense-down | 12288 | 6144 | 8 | 38.49 | 0.820 | 0.008833 |
| glm-5.2 | dense-down | 12288 | 6144 | 16 | 48.68 | 0.733 | 0.008926 |
| glm-5.2 | moe-up | 6144 | 2048 | 1 | 11.38 | 2.002 | 0.004492 |
| glm-5.2 | moe-up | 6144 | 2048 | 2 | 11.40 | 1.966 | 0.004492 |
| glm-5.2 | moe-up | 6144 | 2048 | 4 | 11.99 | 2.112 | 0.005216 |
| glm-5.2 | moe-up | 6144 | 2048 | 6 | 12.23 | 2.005 | 0.005384 |
| glm-5.2 | moe-up | 6144 | 2048 | 8 | 12.36 | 1.952 | 0.005384 |
| glm-5.2 | moe-up | 6144 | 2048 | 16 | 15.29 | 1.616 | 0.005384 |
| glm-5.2 | o-proj | 16384 | 6144 | 1 | 50.54 | 0.813 | 0.008586 |
| glm-5.2 | o-proj | 16384 | 6144 | 2 | 50.67 | 0.833 | 0.008586 |
| glm-5.2 | o-proj | 16384 | 6144 | 4 | 51.12 | 0.844 | 0.010273 |
| glm-5.2 | o-proj | 16384 | 6144 | 6 | 51.47 | 0.827 | 0.010273 |
| glm-5.2 | o-proj | 16384 | 6144 | 8 | 51.92 | 0.803 | 0.010273 |
| glm-5.2 | o-proj | 16384 | 6144 | 16 | 61.76 | 0.737 | 0.010274 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 1 | 11.38 | 2.055 | 0.004148 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 2 | 11.40 | 2.076 | 0.004637 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 4 | 11.99 | 2.099 | 0.004764 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 6 | 12.21 | 1.924 | 0.004764 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 8 | 12.36 | 2.104 | 0.005100 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 16 | 15.28 | 1.594 | 0.005100 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 1 | 58.90 | 0.877 | 0.009195 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 2 | 59.13 | 0.887 | 0.009195 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 4 | 59.38 | 0.887 | 0.009506 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 6 | 59.67 | 0.905 | 0.009597 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 8 | 60.04 | 0.875 | 0.009597 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 16 | 70.24 | 0.808 | 0.011276 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 1 | 20.36 | 1.183 | 0.005133 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 2 | 20.73 | 1.217 | 0.005406 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 4 | 21.04 | 1.206 | 0.005487 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 6 | 21.62 | 1.208 | 0.005487 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 8 | 23.00 | 1.066 | 0.006782 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 16 | 32.07 | 0.883 | 0.007412 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 1 | 19.43 | 1.224 | 0.006025 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 2 | 20.45 | 1.209 | 0.007977 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 4 | 21.39 | 1.155 | 0.007977 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 6 | 22.05 | 1.093 | 0.007977 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 8 | 23.19 | 0.985 | 0.007977 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 16 | 31.92 | 0.786 | 0.008317 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 1 | 14.99 | 1.636 | 0.004511 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 2 | 15.08 | 1.561 | 0.004511 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 4 | 15.32 | 1.557 | 0.004511 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 6 | 16.62 | 1.388 | 0.004548 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 8 | 19.02 | 1.306 | 0.004548 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 16 | 26.42 | 0.940 | 0.004671 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 1 | 11.72 | 2.041 | 0.004089 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 2 | 11.74 | 1.978 | 0.004665 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 4 | 13.15 | 1.847 | 0.004899 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 6 | 13.67 | 1.848 | 0.004899 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 8 | 13.82 | 1.769 | 0.005395 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 16 | 18.08 | 1.413 | 0.005395 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 1 | 15.56 | 1.590 | 0.007415 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 2 | 15.68 | 1.481 | 0.007415 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 4 | 16.07 | 1.698 | 0.007415 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 6 | 16.45 | 1.466 | 0.007415 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 8 | 16.95 | 1.501 | 0.008519 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 16 | 21.86 | 1.112 | 0.008519 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 1 | 9.66 | 2.677 | 0.003950 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 2 | 10.07 | 2.444 | 0.003950 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 4 | 10.27 | 2.201 | 0.004200 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 6 | 11.34 | 2.147 | 0.004200 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 8 | 12.93 | 2.093 | 0.004200 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 16 | 16.51 | 1.576 | 0.004325 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 1 | 12.53 | 1.961 | 0.004102 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 2 | 12.59 | 2.197 | 0.004102 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 4 | 12.70 | 1.837 | 0.004508 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 6 | 13.88 | 1.747 | 0.004508 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 8 | 15.57 | 1.824 | 0.004508 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 16 | 20.57 | 1.285 | 0.004508 |

### `mma_lane_m16_n32_splitk8_pipe2_interleaved` results vs Marlin and vs best Amplin path

| model | role | K | N | M | interleaved (us) | best Amplin path | best (us) | Marlin (us) | interleaved vs best | interleaved vs Marlin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm-5.2 | dense-down | 12288 | 6144 | 1 | 68.13 | amplin_mma_lane_m16_n16_splitk12_raw | 30.53 | 31.00 | +123.2% | 2.198 |
| glm-5.2 | dense-down | 12288 | 6144 | 2 | 68.29 | amplin_mma_lane_m16_n16_splitk12_raw | 31.38 | 31.86 | +117.6% | 2.144 |
| glm-5.2 | dense-down | 12288 | 6144 | 4 | 68.53 | amplin_mma_lane_m16_n16_splitk8_raw | 35.60 | 32.22 | +92.5% | 2.127 |
| glm-5.2 | dense-down | 12288 | 6144 | 6 | 68.59 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 37.34 | 32.44 | +83.7% | 2.115 |
| glm-5.2 | dense-down | 12288 | 6144 | 8 | 68.74 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 38.49 | 31.58 | +78.6% | 2.177 |
| glm-5.2 | dense-down | 12288 | 6144 | 16 | 72.05 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 48.68 | 35.70 | +48.0% | 2.018 |
| glm-5.2 | moe-up | 6144 | 2048 | 1 | 17.36 | amplin_mma_lane_m16_n16_splitk16_raw | 9.89 | 22.78 | +75.5% | 0.762 |
| glm-5.2 | moe-up | 6144 | 2048 | 2 | 17.47 | amplin_mma_lane_m16_n16_splitk12_raw | 9.99 | 22.41 | +74.9% | 0.780 |
| glm-5.2 | moe-up | 6144 | 2048 | 4 | 17.48 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 10.73 | 25.31 | +62.9% | 0.691 |
| glm-5.2 | moe-up | 6144 | 2048 | 6 | 17.59 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 11.19 | 24.51 | +57.3% | 0.718 |
| glm-5.2 | moe-up | 6144 | 2048 | 8 | 17.61 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 11.55 | 24.14 | +52.5% | 0.730 |
| glm-5.2 | moe-up | 6144 | 2048 | 16 | 18.82 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 14.64 | 24.70 | +28.5% | 0.762 |
| glm-5.2 | o-proj | 16384 | 6144 | 1 | 93.12 | amplin_mma_lane_m16_n16_splitk8_raw | 45.60 | 41.08 | +104.2% | 2.267 |
| glm-5.2 | o-proj | 16384 | 6144 | 2 | 92.96 | amplin_mma_lane_m16_n16_splitk8_raw | 46.01 | 42.19 | +102.0% | 2.204 |
| glm-5.2 | o-proj | 16384 | 6144 | 4 | 92.79 | amplin_mma_lane_m16_n16_splitk8_raw | 49.19 | 43.14 | +88.6% | 2.151 |
| glm-5.2 | o-proj | 16384 | 6144 | 6 | 93.58 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 51.47 | 42.56 | +81.8% | 2.199 |
| glm-5.2 | o-proj | 16384 | 6144 | 8 | 92.92 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 51.92 | 41.68 | +79.0% | 2.230 |
| glm-5.2 | o-proj | 16384 | 6144 | 16 | 96.40 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 61.76 | 45.51 | +56.1% | 2.118 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 1 | 17.34 | amplin_mma_lane_m16_n16_splitk16_raw | 9.86 | 23.38 | +75.9% | 0.742 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 2 | 17.44 | amplin_mma_lane_m16_n16_splitk12_raw | 9.99 | 23.66 | +74.5% | 0.737 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 4 | 17.48 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 10.74 | 25.17 | +62.9% | 0.695 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 6 | 17.58 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 11.17 | 23.49 | +57.5% | 0.748 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 8 | 17.62 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 11.55 | 26.00 | +52.6% | 0.678 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 16 | 18.80 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 14.63 | 24.36 | +28.5% | 0.772 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 1 | 104.02 | amplin_mma_lane_m16_n16_splitk8_raw | 53.92 | 51.65 | +92.9% | 2.014 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 2 | 104.13 | amplin_mma_lane_m16_n16_splitk8_raw | 54.79 | 52.47 | +90.1% | 1.985 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 4 | 104.88 | amplin_mma_lane_m16_n16_splitk8_raw | 58.95 | 52.65 | +77.9% | 1.992 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 6 | 104.68 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 59.67 | 53.99 | +75.4% | 1.939 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 8 | 105.56 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 60.04 | 52.52 | +75.8% | 2.010 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 16 | 108.84 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 70.24 | 56.75 | +55.0% | 1.918 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 1 | 35.24 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 20.36 | 24.08 | +73.0% | 1.463 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 2 | 35.47 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 20.73 | 25.23 | +71.1% | 1.406 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 4 | 36.71 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 21.04 | 25.38 | +74.5% | 1.446 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 6 | 36.96 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 21.62 | 26.11 | +71.0% | 1.416 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 8 | 37.38 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 23.00 | 24.52 | +62.5% | 1.524 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 16 | 40.23 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 32.07 | 28.31 | +25.4% | 1.421 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 1 | 31.47 | amplin_mma_lane_m16_n16_splitk16_raw | 15.73 | 23.78 | +100.0% | 1.323 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 2 | 31.65 | amplin_mma_lane_m16_n16_splitk16_raw | 16.27 | 24.73 | +94.6% | 1.279 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 4 | 31.78 | amplin_mma_lane_m16_n16_splitk16_raw | 19.05 | 24.70 | +66.9% | 1.287 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 6 | 32.27 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 20.47 | 24.09 | +57.7% | 1.339 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 8 | 32.87 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 21.27 | 22.84 | +54.5% | 1.439 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 16 | 38.33 | amplin_mma_lane_m16_n32_splitk16_pipe2_raw | 28.69 | 25.09 | +33.6% | 1.528 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 1 | 16.37 | amplin_mma_lane_m16_n16_splitk8_raw | 14.37 | 24.52 | +13.9% | 0.668 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 2 | 16.39 | amplin_mma_lane_m16_n16_splitk8_raw | 14.77 | 23.54 | +11.0% | 0.697 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 4 | 16.71 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 15.32 | 23.86 | +9.0% | 0.700 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 6 | 17.53 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 16.62 | 23.08 | +5.5% | 0.760 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 8 | 19.55 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 18.93 | 24.84 | +3.3% | 0.787 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 16 | 26.28 | amplin_mma_lane_m16_n32_splitk8_pipe2_interleaved_raw | 26.28 | 24.85 | +0.0% | 1.058 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 1 | 17.56 | amplin_mma_lane_m16_n16_splitk16_raw | 10.12 | 23.92 | +73.5% | 0.734 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 2 | 17.69 | amplin_mma_lane_m16_n16_splitk12_raw | 10.02 | 23.21 | +76.5% | 0.762 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 4 | 17.80 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 11.59 | 24.28 | +53.5% | 0.733 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 6 | 17.99 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 13.15 | 25.25 | +36.8% | 0.712 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 8 | 18.01 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 13.48 | 24.45 | +33.6% | 0.737 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 16 | 20.77 | amplin_mma_lane_m16_n32_splitk16_pipe2_raw | 17.05 | 25.55 | +21.8% | 0.813 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 1 | 24.63 | amplin_mma_lane_m16_n16_splitk12_raw | 12.98 | 24.74 | +89.7% | 0.995 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 2 | 24.79 | amplin_mma_lane_m16_n16_splitk12_raw | 13.97 | 23.23 | +77.4% | 1.067 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 4 | 24.88 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 15.23 | 27.28 | +63.4% | 0.912 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 6 | 25.08 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 15.86 | 24.12 | +58.1% | 1.040 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 8 | 25.30 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 16.89 | 25.44 | +49.8% | 0.995 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 16 | 26.68 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 21.86 | 24.32 | +22.0% | 1.097 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 1 | 13.30 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 9.66 | 25.86 | +37.7% | 0.514 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 2 | 11.45 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 9.98 | 24.60 | +14.7% | 0.465 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 4 | 11.60 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 10.15 | 22.59 | +14.3% | 0.513 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 6 | 12.56 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 11.34 | 24.34 | +10.8% | 0.516 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 8 | 14.63 | amplin_mma_lane_m16_n32_splitk12_pipe2_raw | 12.72 | 27.06 | +15.0% | 0.540 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 16 | 17.87 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 16.51 | 26.01 | +8.3% | 0.687 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 1 | 13.28 | amplin_mma_lane_m16_n16_splitk8_raw | 11.55 | 24.57 | +15.0% | 0.541 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 2 | 13.47 | amplin_mma_lane_m16_n16_splitk8_raw | 11.67 | 27.66 | +15.4% | 0.487 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 4 | 13.73 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 12.70 | 23.32 | +8.1% | 0.589 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 6 | 14.81 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 13.88 | 24.24 | +6.7% | 0.611 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 8 | 16.58 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 15.57 | 28.40 | +6.5% | 0.584 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 16 | 21.16 | amplin_mma_lane_m16_n32_splitk8_pipe2_raw | 20.57 | 26.43 | +2.9% | 0.801 |

### Observations

- `splitk8_pipe2` is the fastest Amplin path for several medium/large N32 shapes (GLM o-proj M≥6, Kimi o-proj M≥1, Kimi dense-down M≥6) and beats Marlin on Kimi o-proj for M=1..8.
- It is still slower than Marlin on GLM o-proj (K=16384, N=6144) and Kimi dense-down (K=18432, N=7168).
- `splitk8_pipe2_interleaved` is slower than the non-interleaved `splitk8_pipe2` across all measured target shapes; the uint2 load did not translate to a wall-time win, likely because the two 4-byte loads were already coalesced and the vector load increased register pressure or reduced scheduling flexibility.
- The interleaved variant is therefore **not committed**; this entry records the negative result for future tuning.

## 2026-07-24 Nsight profile of `mma_lane_m16_n32_splitk8_pipe2` (GLM o-proj 16384x6144, M=8, BF16)

**Device:** NVIDIA PG506-230 `GPU-cb9e7784` sm_80 124 SMs  \n**Kernel:** `void <unnamed>::amplin_mma_lane_m16_n32_splitk8_pipe2_kernel<__nv_bfloat16>`  \n**Grid/Block:** `(192, 1, 1)` blocks, `(256, 1, 1)` threads  \n
### Key Nsight Compute metrics

| Section | Metric | Value | Interpretation |
| --- | --- | --- | --- |
| GPU Speed Of Light | Memory Throughput | 49.03% | Moderate memory utilization |
| GPU Speed Of Light | DRAM Throughput | 38.53% | Not DRAM-saturated |
| GPU Speed Of Light | Compute (SM) Throughput | 26.96% | Not compute-saturated |
| GPU Speed Of Light | L1/TEX Hit Rate | 45.28% | Half the requests hit L1 |
| GPU Speed Of Light | L2 Hit Rate | 36.64% | Poor L2 reuse |
| Launch Statistics | Registers Per Thread | 64 | High register pressure |
| Launch Statistics | Static Shared Memory Per Block | 16,384 bytes | 16 KB fixed per block |
| Launch Statistics | Waves Per SM | 0.39 | Grid too small to fill 124 SMs |
| Occupancy | Theoretical Occupancy | 50.0% | Limited by registers |
| Occupancy | Achieved Occupancy | 19.40% | Low; warps not being kept active |
| Occupancy | Block Limit Registers | 4 blocks/SM | 64 regs * 256 threads = 16,384 regs/block, 4 per SM |
| Occupancy | Block Limit Shared Mem | 5 blocks/SM | 16 KB/block fits 5 per SM |

### Observations

- The kernel is **latency/occupancy-bound**, not memory or compute bound: SOL% in both memory and compute is below 50%, but occupancy is far below theoretical (19.4% achieved vs 50% theoretical).
- The grid of 192 blocks is too small for 124 SMs; ncu reports only 0.39 waves per SM and warns the grid is too small to fill the device.
- Register usage (64 per thread) limits occupancy to 4 blocks per SM. Shared memory would allow 5, so registers are the binding constraint.
- The L2 hit rate is low (36.6%), suggesting the weight streaming footprint for M=8 does not reuse cached scales/weights across K groups; this is consistent with M=8 streaming K large and each warp owning a small output tile.

### Hypothesis for next step

Raising the number of resident blocks per SM by reducing register pressure should improve achieved occupancy and hide latency. The next experiment is to change `__launch_bounds__(kMmaLaneSplitK8Threads)` to `__launch_bounds__(kMmaLaneSplitK8Threads, 5)` for the split-K8 pipe2 N32 kernel, which caps registers to 51 per thread (65536 / (256*5)) and allows up to 5 blocks per SM, matching the shared-memory limit.

## 2026-07-24 M-split experiment on `mma_lane_m16_n32_splitk8_pipe2`

**Motivation:** The N32 split-K8 pipe2 kernel processes at most 16 rows per block; for large M this leaves occupancy low and also serializes the 16 rows. Splitting M into 8-row tiles should double the grid for M=16 and give the device more blocks to hide latency.

**Change:** In `amplin_mma_lane_m16_n32_splitk8_pipe2_kernel`, compute `tile_m = blockIdx.y * 8` and `tile_m_size = min(8, size_m - tile_m)`, then pass `input + tile_m*size_k`, `output + tile_m*size_n`, and `tile_m_size` to the generic `splitk12_body`. Launch as `dim3(grid_n, (size_m+7)/8, 1)`.

**Verification:** `pytest -q tests/kernels/test_amplin.py` passed (78 passed), so the tiled indexing is numerically correct.

**Benchmark result on GLM o-proj 16384x6144 BF16:**

| M | original splitk8_pipe2 (us) | M-split splitk8_pipe2 (us) | splitk16_pipe2 (us) | Marlin (us) |
|---|---|---|---|---|
| 1 | 55.5 | ~120 | - | - |
| 2 | - | ~120 | - | - |
| 4 | 52.0 | 112-116 | 54.3 | 42.5 |
| 6 | 53.0 | 112-113 | 55.3 | 42.8 |
| 8 | 57.0 | 112-123 | 56.6 | 42.2 |
| 16 | 62.0 | 112-120 | 63.7 | 45.3 |

Even M=8 (where `blockIdx.y == 0` and `tile_m == 0`) regressed ~2x, so the slowdown is not extra blocks or tile overhead.

**Nsight Compute on the M-split kernel (M=8):**

- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` dropped from ~49% to ~22%.
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` dropped from ~39% to ~18%.
- Registers per thread went from 64 to 63, occupancy stayed ~18.8% (achieved).
- ncu flagged **uncoalesced global loads/stores** and shared-memory bank conflicts for the tiled kernel.
- The dominant warp-stall reason was still L1/TEX scoreboard wait.

**Interpretation:** Offsetting the `input`/`output` pointers inside the split-K8 pipe2 kernel (or replacing the `size_m` guard with a locally-computed `tile_m_size`) caused the compiler to emit much less efficient global access, likely because the `__restrict__`/no-alias information on the original parameters was lost and the uniform `size_m` guard could no longer be exploited. Reverting the offset restored the original ~57 us performance.

**Conclusion:** M-split via pointer offset/local `tile_m_size` is not a viable path. A future M-split must keep the original `__restrict__` `input`/`output` pointers and pass the tile offset/count into the body itself, or it must specialize the kernel by tile size so the compiler can constant-fold the guard. This experiment is **reverted** and recorded here as a negative result.

## 2026-07-24 Mega-kernel route: relax splitk24 N64 group-divisibility constraint
**Change:** remove the `num_groups %% 24 == 0` check from `amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved`.
**Rationale:** the body loop already skips out-of-range groups, so warps that own no group contribute zero partials and the reduction stays correct.
**Verification:** `pytest -q tests/kernels/test_amplin.py` passes (78 passed) after updating the reject-inputs test.
**BF16 wall-time summary** (median µs, speedup ratio < 1.0 means Amplin is faster than Marlin):

### GLM 5.2
| role | M | K | N | best_amplin | splitk24 µs | marlin µs | ratio | vs splitk8 µs |
|---|---|---|---|---|---|---|---|---|
| dense-down | 1 | 12288 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 29.21 | 30.52 | 0.957 | 53.97 |
| dense-down | 2 | 12288 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 29.25 | 31.62 | 0.925 | 89.23 |
| dense-down | 4 | 12288 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 29.89 | 32.07 | 0.932 | 159.04 |
| dense-down | 6 | 12288 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 31.09 | 32.76 | 0.949 | 230.20 |
| dense-down | 8 | 12288 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 31.66 | 31.95 | 0.991 | 301.58 |
| dense-down | 16 | 12288 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 34.86 | 36.26 | 0.961 | 599.57 |
| dense-up | 1 | 6144 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 30.31 | 30.34 | 0.999 | 49.61 |
| dense-up | 2 | 6144 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 30.44 | 31.08 | 0.979 | 81.40 |
| dense-up | 4 | 6144 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 31.87 | 31.63 | 1.008 | 155.74 |
| dense-up | 6 | 6144 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 33.34 | 31.72 | 1.051 | 232.43 |
| dense-up | 8 | 6144 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 33.10 | 31.98 | 1.035 | 305.71 |
| dense-up | 16 | 6144 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 39.15 | 35.68 | 1.097 | 604.55 |
| indexer-wq-b | 1 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.30 | 23.93 | 0.347 | 9.65 |
| indexer-wq-b | 2 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.32 | 21.26 | 0.391 | 13.51 |
| indexer-wq-b | 4 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.57 | 22.23 | 0.386 | 20.44 |
| indexer-wq-b | 6 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.49 | 22.44 | 0.378 | 27.72 |
| indexer-wq-b | 8 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.29 | 21.19 | 0.391 | 35.10 |
| indexer-wq-b | 16 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.09 | 21.02 | 0.432 | 65.52 |
| kv-a-proj | 1 | 6144 | 128 | amplin | 15.68 | 55.28 | 0.284 | 13.19 |
| kv-a-proj | 2 | 6144 | 128 | amplin | 15.71 | 69.24 | 0.227 | 13.52 |
| kv-a-proj | 4 | 6144 | 128 | amplin | 15.85 | 71.40 | 0.222 | 13.81 |
| kv-a-proj | 6 | 6144 | 128 | amplin | 15.90 | 71.73 | 0.222 | 13.85 |
| kv-a-proj | 8 | 6144 | 128 | amplin | 15.94 | 58.21 | 0.274 | 13.89 |
| kv-a-proj | 16 | 6144 | 128 | amplin | 16.94 | 66.74 | 0.254 | 16.42 |
| kv-a-proj-mqa | 1 | 6144 | 576 | amplin | 15.76 | 23.35 | 0.675 | 13.86 |
| kv-a-proj-mqa | 2 | 6144 | 576 | amplin | 15.84 | 23.13 | 0.685 | 14.01 |
| kv-a-proj-mqa | 4 | 6144 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.89 | 22.80 | 0.697 | 16.75 |
| kv-a-proj-mqa | 6 | 6144 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.94 | 22.77 | 0.700 | 16.78 |
| kv-a-proj-mqa | 8 | 6144 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.22 | 21.60 | 0.751 | 22.41 |
| kv-a-proj-mqa | 16 | 6144 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.31 | 22.35 | 0.774 | 33.20 |
| kv-b-proj | 1 | 512 | 28672 | amplin | 16.45 | 20.94 | 0.786 | 11.91 |
| kv-b-proj | 2 | 512 | 28672 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.45 | 20.96 | 0.785 | 18.33 |
| kv-b-proj | 4 | 512 | 28672 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.48 | 20.93 | 0.788 | 31.81 |
| kv-b-proj | 6 | 512 | 28672 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.72 | 21.90 | 0.763 | 44.91 |
| kv-b-proj | 8 | 512 | 28672 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.12 | 21.58 | 0.793 | 57.90 |
| kv-b-proj | 16 | 512 | 28672 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.03 | 21.30 | 0.846 | 111.21 |
| lm-head | 1 | 6144 | 154880 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 308.77 | 328.92 | 0.939 | 490.47 |
| lm-head | 2 | 6144 | 154880 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 311.87 | 331.97 | 0.939 | 963.80 |
| lm-head | 4 | 6144 | 154880 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 323.79 | 334.55 | 0.968 | 1917.59 |
| lm-head | 6 | 6144 | 154880 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 331.19 | 336.65 | 0.984 | 2883.34 |
| lm-head | 8 | 6144 | 154880 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 337.89 | 337.53 | 1.001 | 3841.40 |
| lm-head | 16 | 6144 | 154880 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 387.67 | 378.18 | 1.025 | 7666.92 |
| moe-down | 1 | 2048 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.84 | 22.23 | 0.398 | 11.67 |
| moe-down | 2 | 2048 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.06 | 22.79 | 0.398 | 17.16 |
| moe-down | 4 | 2048 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.81 | 22.50 | 0.391 | 27.88 |
| moe-down | 6 | 2048 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.45 | 22.66 | 0.417 | 38.90 |
| moe-down | 8 | 2048 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.76 | 21.41 | 0.409 | 49.85 |
| moe-down | 16 | 2048 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 11.30 | 23.48 | 0.481 | 95.20 |
| moe-up | 1 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.90 | 22.24 | 0.715 | 16.00 |
| moe-up | 2 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.93 | 21.57 | 0.738 | 22.12 |
| moe-up | 4 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.97 | 22.90 | 0.698 | 32.88 |
| moe-up | 6 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.06 | 22.60 | 0.711 | 43.51 |
| moe-up | 8 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.19 | 22.45 | 0.721 | 53.52 |
| moe-up | 16 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.14 | 22.76 | 0.753 | 95.70 |
| o-proj | 1 | 16384 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 39.75 | 41.18 | 0.965 | 81.01 |
| o-proj | 2 | 16384 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 39.82 | 42.30 | 0.941 | 119.36 |
| o-proj | 4 | 16384 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 40.69 | 42.78 | 0.951 | 212.64 |
| o-proj | 6 | 16384 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 42.40 | 43.33 | 0.979 | 312.09 |
| o-proj | 8 | 16384 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 42.63 | 42.64 | 1.000 | 413.35 |
| o-proj | 16 | 16384 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 45.56 | 46.55 | 0.979 | 817.33 |
| q-a-proj | 1 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.32 | 24.92 | 0.735 | 18.68 |
| q-a-proj | 2 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.90 | 21.69 | 0.733 | 22.12 |
| q-a-proj | 4 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.94 | 22.24 | 0.717 | 32.86 |
| q-a-proj | 6 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.08 | 21.10 | 0.762 | 43.47 |
| q-a-proj | 8 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.17 | 21.06 | 0.768 | 53.47 |
| q-a-proj | 16 | 6144 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.13 | 22.60 | 0.758 | 95.66 |
| q-b-proj | 1 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.15 | 20.64 | 0.395 | 9.66 |
| q-b-proj | 2 | 2048 | 4096 | amplin | 62.43 | 170.01 | 0.367 | 58.51 |
| q-b-proj | 4 | 2048 | 4096 | amplin | 62.74 | 172.62 | 0.363 | 58.39 |
| q-b-proj | 6 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.57 | 21.10 | 0.454 | 32.30 |
| q-b-proj | 8 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.27 | 21.43 | 0.386 | 35.04 |
| q-b-proj | 16 | 2048 | 4096 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.06 | 20.91 | 0.433 | 65.46 |
| q-b-proj-large | 1 | 2048 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.93 | 22.96 | 0.824 | 20.37 |
| q-b-proj-large | 2 | 2048 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.95 | 21.46 | 0.883 | 35.06 |
| q-b-proj-large | 4 | 2048 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 19.18 | 21.40 | 0.896 | 65.52 |
| q-b-proj-large | 6 | 2048 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 19.43 | 21.63 | 0.898 | 95.04 |
| q-b-proj-large | 8 | 2048 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 19.67 | 21.07 | 0.933 | 125.44 |
| q-b-proj-large | 16 | 2048 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 23.36 | 23.22 | 1.006 | 245.78 |

### Kimi K2.5
| role | M | K | N | best_amplin | splitk24 µs | marlin µs | ratio | vs splitk8 µs |
|---|---|---|---|---|---|---|---|---|
| dense-down | 1 | 18432 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 44.04 | 52.13 | 0.845 | 93.50 |
| dense-down | 2 | 18432 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 44.48 | 53.30 | 0.835 | 157.93 |
| dense-down | 4 | 18432 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 46.04 | 53.94 | 0.854 | 284.44 |
| dense-down | 6 | 18432 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 46.95 | 54.87 | 0.856 | 413.18 |
| dense-down | 8 | 18432 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 47.72 | 53.81 | 0.887 | 540.41 |
| dense-down | 16 | 18432 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 51.75 | 59.80 | 0.865 | 1073.48 |
| dense-up | 1 | 7168 | 18432 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 56.56 | 51.81 | 1.092 | 88.72 |
| dense-up | 2 | 7168 | 18432 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 57.07 | 52.65 | 1.084 | 143.15 |
| dense-up | 4 | 7168 | 18432 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 58.56 | 52.98 | 1.105 | 275.45 |
| dense-up | 6 | 7168 | 18432 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 59.39 | 53.31 | 1.114 | 404.96 |
| dense-up | 8 | 7168 | 18432 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 60.42 | 53.26 | 1.135 | 538.34 |
| dense-up | 16 | 7168 | 18432 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 66.65 | 58.42 | 1.141 | 1065.40 |
| kv-a-proj-mqa | 1 | 7168 | 576 | amplin | 18.15 | 21.47 | 0.845 | 17.06 |
| kv-a-proj-mqa | 2 | 7168 | 576 | amplin | 18.15 | 22.02 | 0.824 | 17.13 |
| kv-a-proj-mqa | 4 | 7168 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.22 | 23.44 | 0.777 | 20.20 |
| kv-a-proj-mqa | 6 | 7168 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.30 | 22.71 | 0.806 | 20.26 |
| kv-a-proj-mqa | 8 | 7168 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.50 | 21.40 | 0.865 | 26.50 |
| kv-a-proj-mqa | 16 | 7168 | 576 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 19.92 | 21.28 | 0.936 | 39.52 |
| kv-b-proj | 1 | 512 | 16384 | amplin | 12.43 | 23.41 | 0.531 | 8.98 |
| kv-b-proj | 2 | 512 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.46 | 23.35 | 0.534 | 12.75 |
| kv-b-proj | 4 | 512 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.49 | 22.51 | 0.555 | 20.78 |
| kv-b-proj | 6 | 512 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.57 | 21.26 | 0.591 | 27.97 |
| kv-b-proj | 8 | 512 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.81 | 21.10 | 0.607 | 35.54 |
| kv-b-proj | 16 | 512 | 16384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 13.45 | 21.47 | 0.626 | 65.80 |
| lm-head | 1 | 7168 | 163840 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 401.97 | 403.41 | 0.996 | 606.93 |
| lm-head | 2 | 7168 | 163840 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 403.75 | 406.81 | 0.992 | 1201.52 |
| lm-head | 4 | 7168 | 163840 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 414.32 | 409.95 | 1.011 | 2389.13 |
| lm-head | 6 | 7168 | 163840 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 427.73 | 414.33 | 1.032 | 3579.28 |
| lm-head | 8 | 7168 | 163840 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 436.77 | 414.56 | 1.054 | 4793.67 |
| lm-head | 16 | 7168 | 163840 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 493.73 | 466.64 | 1.058 | 9603.96 |
| o-proj | 1 | 8192 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 20.25 | 24.04 | 0.842 | 38.74 |
| o-proj | 2 | 8192 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 20.30 | 25.29 | 0.803 | 67.57 |
| o-proj | 4 | 8192 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 20.35 | 25.28 | 0.805 | 117.84 |
| o-proj | 6 | 8192 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 20.84 | 27.31 | 0.763 | 168.03 |
| o-proj | 8 | 8192 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 20.84 | 24.73 | 0.843 | 220.51 |
| o-proj | 16 | 8192 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 22.49 | 28.67 | 0.784 | 436.22 |
| q-a-proj | 1 | 7168 | 1536 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.21 | 21.00 | 0.867 | 20.08 |
| q-a-proj | 2 | 7168 | 1536 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.21 | 21.92 | 0.831 | 20.30 |
| q-a-proj | 4 | 7168 | 1536 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.31 | 22.06 | 0.830 | 33.19 |
| q-a-proj | 6 | 7168 | 1536 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.59 | 22.54 | 0.825 | 39.52 |
| q-a-proj | 8 | 7168 | 1536 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.61 | 21.52 | 0.865 | 51.49 |
| q-a-proj | 16 | 7168 | 1536 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 61.46 | 174.77 | 0.352 | 88.43 |
| q-b-proj | 1 | 1536 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.11 | 22.61 | 0.536 | 14.31 |
| q-b-proj | 2 | 1536 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.12 | 21.03 | 0.577 | 22.18 |
| q-b-proj | 4 | 1536 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.22 | 20.96 | 0.583 | 38.96 |
| q-b-proj | 6 | 1536 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.51 | 20.72 | 0.604 | 56.23 |
| q-b-proj | 8 | 1536 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 13.35 | 22.64 | 0.590 | 73.10 |
| q-b-proj | 16 | 1536 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.89 | 23.45 | 0.678 | 141.15 |
| router-gate | 1 | 7168 | 384 | amplin | 18.12 | 35.25 | 0.514 | 17.06 |
| router-gate | 2 | 7168 | 384 | amplin | 18.13 | 43.10 | 0.421 | 17.08 |
| router-gate | 4 | 7168 | 384 | amplin | 18.19 | 44.55 | 0.408 | 17.22 |
| router-gate | 6 | 7168 | 384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.30 | 46.09 | 0.397 | 20.11 |
| router-gate | 8 | 7168 | 384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.44 | 37.97 | 0.486 | 20.21 |
| router-gate | 16 | 7168 | 384 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 19.82 | 41.91 | 0.473 | 32.97 |
| shared-down | 1 | 2048 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.34 | 21.81 | 0.382 | 11.71 |
| shared-down | 2 | 2048 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.87 | 26.82 | 0.368 | 18.85 |
| shared-down | 4 | 2048 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.48 | 26.77 | 0.392 | 31.51 |
| shared-down | 6 | 2048 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.80 | 22.45 | 0.392 | 44.23 |
| shared-down | 8 | 2048 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.61 | 21.35 | 0.403 | 56.89 |
| shared-down | 16 | 2048 | 7168 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 12.04 | 22.19 | 0.542 | 109.85 |
| shared-up | 1 | 7168 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.23 | 22.11 | 0.824 | 19.06 |
| shared-up | 2 | 7168 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.24 | 21.69 | 0.841 | 26.16 |
| shared-up | 4 | 7168 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.20 | 21.67 | 0.840 | 39.05 |
| shared-up | 6 | 7168 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.26 | 21.20 | 0.862 | 51.21 |
| shared-up | 8 | 7168 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.36 | 21.58 | 0.851 | 62.76 |
| shared-up | 16 | 7168 | 2048 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 19.66 | 22.84 | 0.861 | 112.04 |

### Laguna S 2.1
| role | M | K | N | best_amplin | splitk24 µs | marlin µs | ratio | vs splitk8 µs |
|---|---|---|---|---|---|---|---|---|
| dense-down | 1 | 12288 | 3072 | amplin | 27.49 | 21.23 | 1.295 | 25.97 |
| dense-down | 2 | 12288 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 27.50 | 22.46 | 1.225 | 50.60 |
| dense-down | 4 | 12288 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 27.72 | 22.90 | 1.211 | 81.94 |
| dense-down | 6 | 12288 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 27.92 | 23.11 | 1.208 | 112.18 |
| dense-down | 8 | 12288 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 27.98 | 22.74 | 1.231 | 143.16 |
| dense-down | 16 | 12288 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 30.27 | 26.83 | 1.128 | 273.92 |
| dense-up | 1 | 3072 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.39 | 21.77 | 0.799 | 24.65 |
| dense-up | 2 | 3072 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.46 | 22.04 | 0.792 | 40.24 |
| dense-up | 4 | 3072 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.67 | 21.50 | 0.822 | 72.48 |
| dense-up | 6 | 3072 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.92 | 22.70 | 0.789 | 106.64 |
| dense-up | 8 | 3072 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.74 | 22.34 | 0.839 | 139.30 |
| dense-up | 16 | 3072 | 12288 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 21.83 | 21.36 | 1.022 | 274.83 |
| expert-down | 1 | 1024 | 3072 | amplin | 9.15 | 22.68 | 0.404 | 8.18 |
| expert-down | 2 | 1024 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.10 | 20.85 | 0.389 | 9.03 |
| expert-down | 4 | 1024 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.16 | 20.70 | 0.394 | 11.08 |
| expert-down | 6 | 1024 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.56 | 24.09 | 0.355 | 13.76 |
| expert-down | 8 | 1024 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.74 | 22.58 | 0.387 | 16.68 |
| expert-down | 16 | 1024 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 8.73 | 21.86 | 0.399 | 31.20 |
| kv/expert-up | 1 | 3072 | 1024 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 11.35 | 21.09 | 0.538 | 11.54 |
| kv/expert-up | 2 | 3072 | 1024 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.10 | 27.14 | 0.372 | 11.16 |
| kv/expert-up | 4 | 3072 | 1024 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.88 | 22.09 | 0.448 | 13.61 |
| kv/expert-up | 6 | 3072 | 1024 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.97 | 22.30 | 0.447 | 16.87 |
| kv/expert-up | 8 | 3072 | 1024 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.06 | 20.99 | 0.479 | 19.54 |
| kv/expert-up | 16 | 3072 | 1024 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.74 | 21.21 | 0.506 | 29.80 |
| o-proj-6144 | 1 | 6144 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.71 | 20.69 | 0.759 | 16.16 |
| o-proj-6144 | 2 | 6144 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.74 | 23.76 | 0.662 | 27.32 |
| o-proj-6144 | 4 | 6144 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 15.90 | 22.28 | 0.713 | 43.10 |
| o-proj-6144 | 6 | 6144 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.15 | 22.65 | 0.713 | 58.29 |
| o-proj-6144 | 8 | 6144 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 16.21 | 22.79 | 0.711 | 73.97 |
| o-proj-6144 | 16 | 6144 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.54 | 25.87 | 0.678 | 137.67 |
| o-proj-9216 | 1 | 9216 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 21.63 | 23.00 | 0.940 | 23.53 |
| o-proj-9216 | 2 | 9216 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 21.63 | 22.55 | 0.959 | 40.04 |
| o-proj-9216 | 4 | 9216 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 21.68 | 22.81 | 0.951 | 63.78 |
| o-proj-9216 | 6 | 9216 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 21.87 | 25.02 | 0.874 | 86.44 |
| o-proj-9216 | 8 | 9216 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 22.02 | 25.08 | 0.878 | 109.71 |
| o-proj-9216 | 16 | 9216 | 3072 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 23.85 | 23.78 | 1.003 | 205.80 |
| q-proj-6144 | 1 | 3072 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.01 | 21.55 | 0.465 | 16.70 |
| q-proj-6144 | 2 | 3072 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.07 | 21.18 | 0.476 | 24.62 |
| q-proj-6144 | 4 | 3072 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.20 | 21.45 | 0.476 | 40.51 |
| q-proj-6144 | 6 | 3072 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.45 | 21.02 | 0.497 | 56.57 |
| q-proj-6144 | 8 | 3072 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.57 | 22.92 | 0.461 | 72.62 |
| q-proj-6144 | 16 | 3072 | 6144 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 11.65 | 22.80 | 0.511 | 139.08 |
| q-proj-9216 | 1 | 3072 | 9216 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.12 | 22.80 | 0.751 | 19.47 |
| q-proj-9216 | 2 | 3072 | 9216 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.16 | 22.84 | 0.752 | 32.25 |
| q-proj-9216 | 4 | 3072 | 9216 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.40 | 25.93 | 0.671 | 56.52 |
| q-proj-9216 | 6 | 3072 | 9216 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 17.56 | 21.83 | 0.804 | 80.63 |
| q-proj-9216 | 8 | 3072 | 9216 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 18.10 | 22.75 | 0.796 | 106.58 |
| q-proj-9216 | 16 | 3072 | 9216 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 20.91 | 22.32 | 0.937 | 207.34 |
| router-gate | 1 | 3072 | 256 | amplin | 9.74 | 29.76 | 0.327 | 9.62 |
| router-gate | 2 | 3072 | 256 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.76 | 36.58 | 0.267 | 9.81 |
| router-gate | 4 | 3072 | 256 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.85 | 37.12 | 0.265 | 9.89 |
| router-gate | 6 | 3072 | 256 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 9.94 | 38.14 | 0.261 | 10.00 |
| router-gate | 8 | 3072 | 256 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.04 | 31.37 | 0.320 | 11.08 |
| router-gate | 16 | 3072 | 256 | mma_lane_m16_n64_splitk24_pipe2_interleaved | 10.73 | 35.61 | 0.301 | 13.60 |

### 2026-07-24 — Success: split-K12x2 cooperative N64 no longer requires num_groups % 24

Revision: `f26a01d8`
Target: GPTQ 4-bit group_size=128 desc_act=False sym=True, batch sizes 1/2/4/6/8/16 on PG506-230/232 (sm_80, 124 SMs), BF16.

The `mma_lane_m16_n64_splitk12x2_coop_interleaved` kernel was gated to `num_groups % 24 == 0` and the benchmark script mirrored that. The kernel body already handles warps with no assigned K group by writing zero partials and participating in the cooperative grid reduction, so the guard was removed. `test_amplin` now asserts the kernel runs and matches the deterministic reference on a non-multiple-of-24 group count.

A full-shape sweep with `splitk24` + `splitk12x2` enabled shows the cooperative mega-kernel closes the remaining small-N gaps and is now selected for many shapes:

| model | role | M | K | N | best Amplin path | batch median us | Marlin us | vs Marlin |
|---|---|---|---|---|---|---|---|---|
| Laguna | dense-down | 1-16 | 12288 | 3072 | splitk12x2 | 18.4-21.3 | 22.4-25.4 | 1.19-1.27x |
| Laguna | o-proj-6144 | 1-16 | 6144 | 3072 | splitk12x2 | 11.8-14.3 | 21.8-22.9 | 1.56-1.85x |
| Laguna | o-proj-9216 | 1-16 | 9216 | 3072 | splitk12x2 | 15.1-18.1 | 21.7-23.4 | 1.24-1.48x |
| Laguna | q-a-proj | 1-16 | 6144 | 2048 | splitk12x2 | 13.2-16.4 | 22.9-26.1 | 1.71-2.24x |
| Laguna | moe-up | 1-16 | 6144 | 2048 | splitk12x2 | 11.7-13.2 | 21.4-24.4 | 1.63-2.03x |
| GLM | q-a-proj | 1-16 | 6144 | 2048 | splitk12x2 | 11.6-16.4 | 22.9-26.1 | 1.78-2.24x |
| GLM | moe-up | 1-16 | 6144 | 2048 | splitk12x2 | 11.6-15.4 | 21.4-25.6 | 1.63-2.15x |
| Kimi | q-a-proj | 1-16 | 7168 | 1536 | splitk12x2 | 13.9-16.2 | 21.9-23.8 | 1.40-1.69x |
| Kimi | shared-up | 1-16 | 7168 | 2048 | splitk12x2 | 14.0-15.8 | 21.7-23.4 | 1.48-1.62x |
| Kimi | kv-a-proj-mqa | 1-16 | 7168 | 576 | splitk12x2 | 13.7-15.6 | 22.4-29.0 | 1.51-1.86x |
| Kimi | router-gate | 1-16 | 7168 | 384 | splitk12x2 | 13.7-15.6 | 34.9-45.9 | 2.54-3.25x |

Remaining Amplin losses after this change (splitk24 is the best available path for these; splitk12x2 grid is too large):

| model | role | M | K | N | vs Marlin |
|---|---|---|---|---|---|
| GLM | dense-down | 8 | 12288 | 6144 | 0.974x |
| GLM | dense-up | 6/8/16 | 6144 | 12288 | 0.974/0.951/0.902x |
| GLM | lm-head | 16 | 6144 | 154880 | 0.980x |
| GLM | o-proj | 8 | 16384 | 6144 | 0.991x |
| Kimi | dense-up | 1-16 | 7168 | 18432 | 0.850-0.916x |
| Kimi | lm-head | 4-16 | 7168 | 163840 | 0.944-0.987x |

Observation: the losses are now concentrated in `dense-up` and `lm-head` shapes with very large N, where the splitk24 24-warp N64 kernel is the fastest Amplin path but still trails Marlin by 2-15%. A shared-A mega-kernel that reuses A across N tiles or a larger-K-split layout is the next hypothesis. M=32 is still uncovered and is the target after M=16 gaps are closed.

### 2026-07-24 — Failure: shared-A M16/M32 N64 mega-kernel is correct but slower than Marlin

Revision: working tree on `devin/1784898993-amplin-batches` after `f26a01d8`
Target: GPTQ 4-bit group_size=128 desc_act=False sym=True, batch 16 and 32, FP16 on PG506-230/232 (sm_80, 124 SMs).

Implemented `amplin_mma_lane_mN_n64_shared_a_kernel<Scalar, BlockM>` (BlockM=16 and 32) in `gptqmodel_ext/amplin/amplin_kernel.cu` plus host wrappers `mma_lane_m16_n64_shared_a` and `mma_lane_m32_n64_shared_a`. The kernel loads the full A tile for the block into shared memory once per K group, then all N warps reuse that A tile while streaming the lane-packed weights. Output is guarded by `store_mma_fragment_guard_m` so it supports any `size_m <= BlockM`.

Correctness: verified against the same `(codes - 8) * scales` FP32 dequantized reference; max abs error is ~2e-3 for FP16, well inside the FP16 quantization noise floor.

Performance (FP16, median CUDA-event us, GPU 0, warmup 10 iters 30):

| M | K | N | Marlin | m16 shared-A | m32 shared-A | splitk24 M16 | m32 global-a |
|---|---|---|---|---|---|---|---|---|
| 16 | 6144 | 12288 | 69.2 | 170.8 | - | 80.9 | - |
| 16 | 7168 | 18432 | 91.8 | 229.1 | - | 110.7 | - |
| 16 | 7168 | 163840 | 452.8 | 1025.2 | - | 472.1 | - |
| 32 | 6144 | 12288 | 74.9 | - | 247.2 | - | 251.1 |
| 32 | 7168 | 18432 | 100.1 | - | 370.5 | - | 301.7 |
| 32 | 7168 | 163840 | 499.2 | - | 1995.9 | - | 1134.0 |

Result: the shared-A mega-kernel is 2.0-4.0x slower than Marlin and also loses to `splitk24` for M=16 and to the existing `m32_global_a` for most M=32 cases.

Nsight Compute on `M=32 K=7168 N=18432` (FP16) shows the kernel is not memory bound:

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed: 15.09%
dram__throughput.avg.pct_of_peak_sustained_elapsed: 8.86%
smsp__cycles_elapsed.avg: 399168 cycles
launch: (288,1,1) x (256,1,1)
```

The bottleneck is low SM utilization because the block only has 8 warps and processes the entire K dimension sequentially. Each thread issues a scalar `MmaLaneDequant` + `MmaInstruction` chain per K step with little independent work to hide latency. By comparison `splitk24` on the same shape achieves ~34% SM throughput and 40% DRAM throughput by splitting K across 24 warps.

Conclusion: a shared-A N64 kernel is not enough; it must also K-split across warps and use vector `uint4` weight loads to expose the same instruction-level parallelism as `splitk24`. M=32 support additionally requires handling two 16-row A tiles, which doubles accumulator/partial storage. The next candidate is a K-split shared-A N64 kernel (e.g. 2 K-slices x 4 N warps) for M<=32, or a cooperative multi-CTA variant that keeps partials within the device-wide shared budget.

### 2026-07-24 — Failure: K-split shared-A M16/M32 N64 mega-kernel is correct but still slower than Marlin

Revision: working tree on `devin/1784898993-amplin-batches` after `f26a01d8`
Target: GPTQ 4-bit group_size=128 desc_act=False sym=True, batch 16 and 32, FP16 on PG506-230/232 (sm_80, 124 SMs).

Implemented `amplin_mma_lane_mN_n64_shared_a_kernel<Scalar, BlockM>` as a K-split shared-A N64 mega-kernel.  Each block uses `NWarp=2` (each warp owns two adjacent N16 tiles) and `KSplit=10` for `BlockM=16` / `KSplit=5` for `BlockM=32` so the combined shared-A + partials scratch fits in the 96 KB opt-in shared-memory limit.  The kernel loads `KSplit` K-group A tiles into shared memory, has `MWarpGroups * KSplit * NWarp` warps per block, and performs a per-warp K-slice accumulation followed by a shared-memory reduction across K-slices before storing.  Weight packing was switched to `pack_mma_lane_n64_qweight` (`[N/64, K/128, 8, 32, 4]`) so each warp can `uint4`-load the four N-group words at once.  Host wrappers call `cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize` and pass the total dynamic shared bytes at launch.

Correctness: verified against the FP32 `(codes - 8) * scales` dequantized reference for both M=16 and M=32 on `6144x12288`, `7168x18432`, and `7168x163840`.  Max abs error is ~1e-3 for FP16, inside the quantization noise floor.  `pytest -q tests/kernels/test_amplin.py` passes (78/78).

Performance (FP16, median CUDA-event us, GPU 0, warmup 10 iters 30):

| M | K | N | Marlin | m16 shared-A K-split | m32 shared-A K-split | splitk24 M16 | m32 global-a |
|---|---|---|---|---|---|---|---|---|
| 16 | 6144 | 12288 | 70.3 | 133.1 | - | 82.0 | - |
| 16 | 7168 | 18432 | 93.0 | 196.3 | - | 112.0 | - |
| 16 | 7168 | 163840 | 454.2 | 959.7 | - | 475.3 | - |
| 32 | 6144 | 12288 | 70.9 | - | 194.6 | - | 234.9 |
| 32 | 7168 | 18432 | 93.6 | - | 303.1 | - | 277.3 |
| 32 | 7168 | 163840 | 501.1 | - | 1872.3 | - | 1154.0 |

Result: the K-split shared-A mega-kernel is 1.9-3.7x slower than Marlin for M=16 and 2.7-3.7x slower for M=32.  It also loses to the existing `splitk24` M16 path and to the existing `m32_global_a` M32 path for the tested shapes.

`pytest -q tests/kernels/test_amplin.py`: 78 passed.

Conclusion: K-split + shared-A does not close the M=16/32 dense-up/lm-head gap on its own.  The extra reduction and dynamic shared-memory traffic outweigh the benefit of A reuse, and the 96 KB shared limit caps the number of independent warps below `splitk24`.  The next mega-kernel route should either avoid the shared-A reduction (e.g. a cooperative multi-CTA K-split that keeps partials in device memory) or target M=32 with a different tile geometry (e.g. M64 with global A and more N warps).

### 2026-07-24 — Cooperative multi-CTA K-split N64 for M<=32 (device-memory partials)

Revision: working tree on `devin/1784898993-amplin-batches` after the K-split shared-A failure entry.
Target: GPTQ 4-bit group_size=128 desc_act=False sym=True, batches 1,2,4,6,8,16,32, FP16/BF16 on PG506-230/232 (sm_80, 124 SMs).

Implemented `amplin_mma_lane_mN_n64_splitk12x2_coop_interleaved_kernel<Scalar, BlockM>` in `gptqmodel_ext/amplin/amplin_kernel.cu`, registered as `mma_lane_m32_n64_splitk12x2_coop_interleaved` in `gptqmodel_ext/amplin/amplin.cpp`, `gptqmodel/utils/amplin.py`, and exposed via `--m32-splitk12x2-n64-coop` in `scripts/benchmark_amplin_model_shapes.py`.

Design:
- 12-warps per CTA, 2 CTAs per cooperative pair (24 K-slices total).
- Each CTA is split into `MWarpGroups = BlockM / 16` M-groups (1 for M=16, 2 for M=32) with `WarpsPerMGroup = 6`.
- Each M-group processes the same K-slice partition; each warp accumulates `MWarpGroups` consecutive K-groups (1 for M=16, 2 for M=32) before reduction, so every M-group covers the full K dimension.
- The grid loops over N64 tiles: `gridDim.x = min(N/64, active_blocks_per_sm * sm_count / 2)`, escaping the original `(N/64)*2 <= active_blocks_per_sm * sm_count` cooperative-residency limit.
- Per-CTA partials are reduced in shared memory; the two CTAs then reduce through a `float[2][M][N]` device scratch buffer and `cooperative_groups::this_grid().sync()`.
- Weight layout uses `pack_mma_lane_n64_qweight` (`[N/64, K/128, 8, 32, 4]`) for aligned `uint4` lane loads; scales use `pack_hmma_scales`.

Correctness: verified against the FP32 `(codes - 8) * scales` dequantized reference for M=32 on `6144x12288`, `7168x18432`, and `7168x163840`. Max abs error is ~1e-3 for FP16, inside the quantization noise floor. `pytest -q tests/kernels/test_amplin.py`: 78 passed.

Performance (FP16, median CUDA-event us, GPU 0, warmup 20 iters 50) on the Laguna S 2.1 / GLM 5.2 / Kimi K2.5 shape list at M=32:

| model | role | K | N | Marlin | m32_global_a | m32_splitk12x2_coop | coop err |
|---|---|---|---|---|---|---|---|
| laguna-s-2.1 | expert-down | 1024 | 3072 | 56.8 | 64.3 | 67.2 | 0.0003 |
| laguna-s-2.1 | kv/expert-up | 3072 | 1024 | 62.6 | 92.3 | 66.0 | 0.0005 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 56.1 | 94.9 | 67.7 | 0.0006 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 57.6 | 98.2 | 87.2 | 0.0007 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 57.8 | 98.6 | 87.9 | 0.0006 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 57.0 | 143.8 | 69.4 | 0.0007 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 60.8 | 193.1 | 74.6 | 0.0011 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 61.6 | 225.7 | 81.8 | 0.0012 |
| laguna-s-2.1 | router-gate | 3072 | 256 | 80.2 | 86.1 | 65.7 | 0.0005 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 58.1 | 135.4 | 68.6 | 0.0010 |
| glm-5.2 | q-b-proj | 2048 | 4096 | 55.6 | 74.9 | 66.9 | 0.0005 |
| glm-5.2 | q-b-proj-large | 2048 | 16384 | 56.8 | 91.5 | 88.1 | 0.0006 |
| glm-5.2 | kv-a-proj | 6144 | 128 | 122.3 | 128.2 | 66.8 | 0.0006 |
| glm-5.2 | kv-a-proj-mqa | 6144 | 576 | 57.5 | 131.6 | 67.1 | 0.0007 |
| glm-5.2 | kv-b-proj | 512 | 28672 | 55.8 | 63.7 | 83.8 | 0.0004 |
| glm-5.2 | o-proj | 16384 | 6144 | 80.0 | 523.1 | 130.4 | 0.0013 |
| glm-5.2 | dense-up | 6144 | 12288 | 69.3 | 229.3 | 115.6 | 0.0009 |
| glm-5.2 | dense-down | 12288 | 6144 | 71.1 | 385.9 | 108.3 | 0.0012 |
| glm-5.2 | moe-up | 6144 | 2048 | 57.2 | 128.9 | 68.0 | 0.0010 |
| glm-5.2 | moe-down | 2048 | 6144 | 56.1 | 73.4 | 67.6 | 0.0006 |
| glm-5.2 | indexer-wq-b | 2048 | 4096 | 55.7 | 73.3 | 66.8 | 0.0005 |
| glm-5.2 | lm-head | 6144 | 154880 | 420.7 | 982.6 | 751.1 | 0.0010 |
| kimi-k2.5 | q-a-proj | 7168 | 1536 | 58.8 | 148.4 | 71.5 | 0.0008 |
| kimi-k2.5 | q-b-proj | 1536 | 12288 | 56.1 | 68.9 | 74.4 | 0.0005 |
| kimi-k2.5 | kv-a-proj-mqa | 7168 | 576 | 56.6 | 140.6 | 69.9 | 0.0006 |
| kimi-k2.5 | kv-b-proj | 512 | 16384 | 55.2 | 61.5 | 73.9 | 0.0003 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 63.3 | 183.7 | 89.7 | 0.0010 |
| kimi-k2.5 | dense-up | 7168 | 18432 | 92.4 | 276.3 | 175.7 | 0.0011 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 91.4 | 595.0 | 135.2 | 0.0014 |
| kimi-k2.5 | shared-up | 7168 | 2048 | 56.9 | 143.5 | 70.0 | 0.0007 |
| kimi-k2.5 | shared-down | 2048 | 7168 | 56.1 | 76.0 | 67.3 | 0.0005 |
| kimi-k2.5 | router-gate | 7168 | 384 | 84.5 | 136.1 | 69.6 | 0.0006 |
| kimi-k2.5 | lm-head | 7168 | 163840 | 499.3 | 1148.7 | 978.2 | 0.0011 |

Result: the cooperative multi-CTA kernel is the fastest Amplin path for M=32 across the tested model shapes (2-6x faster than `m32_global_a`). It also beats Marlin for small-N M=32 cases (`router-gate`, `kv-a-proj`, `kv-a-proj-mqa` with N <= 576). It is still slower than Marlin for `dense-up` and `lm-head` large-N shapes, where Marlin's scheduling remains superior.

`pytest -q tests/kernels/test_amplin.py`: 78 passed.
`ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_model_shapes.py tests/kernels/test_amplin.py`: passed.
`git diff --check`: passed.

Conclusion: the cooperative multi-CTA K-split N64 mega-kernel is a verified M=32 improvement for Amplin and closes the gap on small-N M=32 shapes, but it does not yet beat Marlin on `dense-up` / `lm-head` at M=16/32. The remaining large-N losses likely need a different tile geometry or a persistent/cross-SM schedule that can hide latency with more independent warps than the 12-warps-per-CTA design allows.

## 2026-07-24: M32 N64 single-CTA K24 split (pipe2 interleaved)

Revision: working tree on `devin/1784898993-amplin-batches`.
Target: same as above.

Hypothesis: the cooperative M32 kernel is limited by device-memory partial traffic and a small grid (`gridDim.x` capped at ~124) for very large N. A single-CTA K24 split that keeps partials in shared memory and launches one CTA per N64 tile should expose more memory-level parallelism and reduce scratch traffic.

Implementation:
- Added `amplin_mma_lane_mN_n64_splitk24_pipe2_interleaved_body<Scalar>` and `amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_kernel<Scalar>` in `gptqmodel_ext/amplin/amplin_kernel.cu`.
- Uses 24 warps per CTA, 2 M groups of 12 warps each; each M group has a 12-way K-split.
- Each warp owns one K-slice and computes a full N64 tile for both M groups; partials are accumulated in shared memory and reduced before a single store to global output.
- Dynamic shared memory: 96 KB (48 KB partials + 48 KB reserved; the compiler currently allocates only the partials array through the extern pointer and the launch requests 96 KB to match the sm_80 opt-in limit).
- Registered as `mma_lane_m32_n64_splitk24_pipe2_interleaved` in `gptqmodel_ext/amplin/amplin.cpp`, `gptqmodel/utils/amplin.py`, and `scripts/benchmark_amplin_model_shapes.py` (`--m32-splitk24-n64`).
- Added `test_amplin_mma_lane_m32_n64_splitk24_matches_fp32_reference` in `tests/kernels/test_amplin.py` for M=32 on the fixture shape.

Correctness: `pytest -q tests/kernels/test_amplin.py`: 80 passed. FP16 max abs error on `7168x163840` is ~2e-3, inside the quantization noise floor. Manual checks on M=17,24,32 (small K=1024,N=256) also pass.

Nsight Compute on `kimi-k2.5 lm-head M=32 K=7168 N=163840`:
- Duration: ~921 us (cooperative was ~1030 us; Marlin is ~502 us).
- Memory Throughput: 65.9%, DRAM Throughput: 27.5%, Max Bandwidth: 41.3%, Compute (SM) Throughput: 48.1%.
- Occupancy: 37.5%, 1 block per SM due to 96 KB shared memory and 80 registers per thread.

Focused micro-benchmark (FP16, median CUDA-event us, GPU 0):

| M | K | N | cooperative | splitk24-single-CTA | speedup vs coop | Marlin (wall) |
|---|---|---|-------------|---------------------|-----------------|---------------|
| 32 | 7168 | 163840 | 1038.8 | 866.3 | 1.20x | 555.2 |
| 32 | 7168 | 18432  | 142.5  | 124.9 | 1.14x | 73.1 |
| 32 | 6144 | 12288  | 77.4   | 71.8  | 1.08x | 45.6 |
| 32 | 4096 | 4096   | 41.4   | 36.5  | 1.13x | 25.2 |
| 32 | 6144 | 4096   | 40.6   | 37.5  | 1.08x | - |

Result: the single-CTA K24 split is a consistent improvement over the cooperative kernel for M=32 (8-20% faster), but it still loses to Marlin on large-N `dense-up` / `lm-head` shapes. The kernel is memory-bound with ~41% of peak memory bandwidth; the remaining gap suggests the next route should reduce activation traffic (e.g., reuse A across multiple N tiles via shared memory) rather than increase K-split alone.

`pytest -q tests/kernels/test_amplin.py`: 80 passed.
`ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_model_shapes.py tests/kernels/test_amplin.py`: passed.
`git diff --check`: passed.

## 2026-07-24: A-reuse across multiple N64 tiles (failed)

Hypothesis: the M32 large-N losses are memory-bound; loading the activation tile A into shared memory once and reusing it across several N64 tiles should cut activation traffic while keeping the warp count high enough to saturate memory.

Prototype: `amplin_mma_lane_mN_n64_multitile_shared_a` in `gptqmodel_ext/amplin/amplin_kernel.cu`. A single 768-thread CTA uses 24 warps to process `NTiles = 12` N64 tiles for `BlockM=32` (or 24 tiles for `BlockM=16`). The full A tile for the current K group is copied into shared memory once, then each warp reads its own `fragment_a` via `ldmatrix` and processes one N64 tile for all `num_groups`. A is re-loaded every K group.

Correctness: passes for `M=1..32, K=4096, N=256` and `M=32, K=7168, N=18432/163840` against the FP32 dequantized reference; max FP16 error ~1.3e-3.

Focused micro-benchmark (FP16, median CUDA-event us, GPU 0, seed 1234, 3 warmup + 20 iters):

| M | K | N | multi-tile shared-A | splitk24 single-CTA | Marlin |
|---|---|---|--------------------:|--------------------:|-------:|
| 16 | 7168 | 18432  | 479.6 |  -  | 120.9 |
| 32 | 7168 | 18432  | 475.3 | 175.6 | 127.8 |
| 32 | 7168 | 163840 | 833.1 / 921.5 | 847.1 | 529.3 / 530.8 |

Nsight Compute on `M=32 K=7168 N=163840`:

| Metric | Value |
| --- | --- |
| Duration | ~885 us |
| Memory Throughput | 703.82 GB/s |
| Max Bandwidth | 28.77 % |
| Mem Busy | 67.55 % |
| L1/TEX Hit Rate | 49.11 % |
| L2 Hit Rate | 33.89 % |
| dram__sectors_read.sum | ~608 MB |

Conclusion: the multi-N-tile shared-A kernel is 2-3x slower than the current `splitk24` path and ~1.6-1.8x slower than Marlin. The Nsight profile shows the kernel is memory-bound but only achieves ~29% of max bandwidth; the L1 global-load hit rate is low and DRAM read traffic is much larger than the nominal activation+weight size. The root cause is that mapping one warp per N tile reduces the total warp count by 12x versus `splitk24`, which lowers memory-level parallelism and prevents the memory pipeline from saturating. Activation reuse alone is not enough; the next route must combine activation reuse with K-split across warps (more independent memory requests per N tile) and/or share the weight load across the two M groups to avoid duplicating the 4-bit weights for M=32.

## 2026-07-25: M16 shared-A N64 tile4 / tile8 full-k kernels

Hypothesis: for M<=16, the activation tile is small enough to be kept in shared memory while one warp processes a full N64 tile over all K groups. Loading the `BlockM x K` activation once and reusing it across several consecutive N64 tiles should reduce A traffic and kernel overhead on very large-N shapes (lm-head and kv-b-proj) without the synchronization cost of a K-split reduction.

Prototype: `amplin_mma_lane_mN_n64_tiled_fullk_kernel<Scalar, BlockM, NTiles>` in `gptqmodel_ext/amplin/amplin_kernel.cu` and host wrappers `amplin_mma_lane_m16_n64_tile4_shared_a_cuda` / `amplin_mma_lane_m16_n64_tile8_shared_a_cuda`. The kernel copies `BlockM x 128` of A into shared memory per K group. One warp owns one N64 tile and accumulates the full K for that tile; the A tile is reused for `NTiles` consecutive N64 tiles in the CTA. No K-split is used, so there is no partial reduction. Registered as `mma_lane_m16_n64_tile4_shared_a` and `mma_lane_m16_n64_tile8_shared_a` in `amplin.cpp` with Python wrappers in `gptqmodel/utils/amplin.py`.

A K-split variant (`NTiles=4, KSplit=2`) was implemented and tested; it was not faster than the no-K-split tile4 path, so its public wrapper was removed before commit and only the template remains as a future building block.

Correctness: `tests/kernels/test_amplin.py` `test_amplin_padded_m16_large_mlp_projection_matches_fp32_dequant_reference` now exercises tile4 and tile8 for the Laguna/Qwen MLP shapes. `pytest -q tests/kernels/test_amplin.py`: 80 passed. Max FP16 error is ~1e-3 against the FP32 dequantized reference, matching the existing `splitk24` error.

Focused benchmark script: `scripts/benchmark_amplin_m16_tile4.py` (FP16, GPU 0, seed 20260724, 20 warmup, 100 iters, 5 rounds, median CUDA-event us).

M=16 results across Laguna S 2.1, GLM 5.2, and Kimi K2.5 shapes:

| model | role | K | N | splitk24 | tile4 | tile8 | marlin | fastest |
|---|---|---|---:|---:|---:|---:|---|
| laguna-s-2.1 | expert-down | 1024 | 3072 | 62.3 | 63.5 | 72.8 | 84.6 | splitk24 |
| laguna-s-2.1 | kv/expert-up | 3072 | 1024 | 61.8 | 94.3 | 117.5 | 85.1 | splitk24 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 62.5 | 90.3 | 111.8 | 86.2 | splitk24 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 62.6 | 88.8 | 108.7 | 84.9 | splitk24 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 62.5 | 89.0 | 109.3 | 86.1 | splitk24 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 62.0 | 130.8 | 169.3 | 85.2 | splitk24 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 63.2 | 173.6 | 230.1 | 83.9 | splitk24 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 69.4 | 216.4 | 291.7 | 84.2 | splitk24 |
| laguna-s-2.1 | router-gate | 3072 | 256 | 62.6 | 86.7 | - | 92.2 | splitk24 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 62.9 | 129.6 | 168.4 | 86.3 | splitk24 |
| glm-5.2 | q-b-proj | 2048 | 4096 | 62.1 | 73.1 | 86.8 | 84.1 | splitk24 |
| glm-5.2 | q-b-proj-large | 2048 | 16384 | 63.6 | 74.0 | 87.6 | 84.7 | splitk24 |
| glm-5.2 | kv-a-proj | 6144 | 128 | 62.2 | - | - | 123.9 | splitk24 |
| glm-5.2 | kv-a-proj-mqa | 6144 | 576 | 62.0 | - | - | 83.1 | splitk24 |
| glm-5.2 | kv-b-proj | 512 | 28672 | 62.7 | 62.2 | 62.1 | 84.6 | tile8 |
| glm-5.2 | o-proj | 16384 | 6144 | 83.2 | 354.6 | 438.2 | 101.0 | splitk24 |
| glm-5.2 | dense-up | 6144 | 12288 | 76.3 | 158.8 | 195.4 | 90.9 | splitk24 |
| glm-5.2 | dense-down | 12288 | 6144 | 71.6 | 271.7 | 338.0 | 90.4 | splitk24 |
| glm-5.2 | moe-up | 6144 | 2048 | 61.8 | 129.3 | 168.4 | 83.7 | splitk24 |
| glm-5.2 | moe-down | 2048 | 6144 | 61.5 | 73.4 | 88.7 | 83.2 | splitk24 |
| glm-5.2 | indexer-wq-b | 2048 | 4096 | 61.5 | 73.5 | 86.4 | 84.7 | splitk24 |
| glm-5.2 | lm-head | 6144 | 154880 | 393.4 | 326.1 | - | 409.2 | tile4 |
| kimi-k2.5 | q-a-proj | 7168 | 1536 | 64.8 | 144.7 | 190.7 | 84.9 | splitk24 |
| kimi-k2.5 | q-b-proj | 1536 | 12288 | 64.5 | 68.3 | 79.8 | 85.5 | splitk24 |
| kimi-k2.5 | kv-a-proj-mqa | 7168 | 576 | 65.2 | - | - | 86.0 | splitk24 |
| kimi-k2.5 | kv-b-proj | 512 | 16384 | 64.9 | 64.5 | 64.0 | 85.2 | tile8 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 65.2 | 161.5 | 211.1 | 85.9 | splitk24 |
| kimi-k2.5 | dense-up | 7168 | 18432 | 103.7 | 186.8 | 226.5 | 112.6 | splitk24 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 86.7 | 395.8 | 492.3 | 111.2 | splitk24 |
| kimi-k2.5 | shared-up | 7168 | 2048 | 62.8 | 143.1 | 187.9 | 84.9 | splitk24 |
| kimi-k2.5 | shared-down | 2048 | 7168 | 63.4 | 74.2 | 87.6 | 85.6 | splitk24 |
| kimi-k2.5 | router-gate | 7168 | 384 | 64.5 | - | - | 99.5 | splitk24 |
| kimi-k2.5 | lm-head | 7168 | 163840 | 502.1 | 477.2 | 496.6 | 484.5 | tile4 |

Key wins at M=16:
- `glm-5.2 lm-head` 6144x154880: tile4 326us vs splitk24 393us (-17%) vs Marlin 409us (-20%).
- `kimi-k2.5 lm-head` 7168x163840: tile4 477us vs splitk24 502us (-5%) vs Marlin 484us (-1%).
- `glm-5.2 kv-b-proj` 512x28672: tile8 62us vs splitk24 63us vs Marlin 85us.
- `kimi-k2.5 kv-b-proj` 512x16384: tile8 64us vs splitk24 65us vs Marlin 85us.

Dense-up M=16 shapes remain fastest with `splitk24` (laguna 62.5us vs Marlin 86.1us; glm 76.3us vs Marlin 90.9us; kimi 103.7us vs Marlin 112.6us), so the M16 large-N gap is now essentially the lm-head class, where tile4 is the new best.

Conclusion: tile4 is the best Amplin kernel for M=16 `lm-head` shapes. The remaining M=16 opportunity outside lm-head is small, and the tile4 path should only be selected for very large N (lm-head / kv-b-proj). The next gap is M=32 large-N (dense-up, lm-head, dense-down), where neither the current single-CTA K24 split nor shared-A tile4 closes the Marlin gap.

`ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py scripts/benchmark_amplin_m16_tile4.py`: passed.
`git diff --check`: passed.

## 2026-07-25: Extend shared-A N64 tile4 / tile8 full-k kernel to M=32

Goal: close the M=32 large-N gap by reusing the M16 shared-A tile4/tile8 N64 kernel for BlockM=32.

Change: `gptqmodel_ext/amplin/amplin_kernel.cu`:
- Fixed `__launch_bounds__` for `amplin_mma_lane_mN_n64_tiled_fullk_kernel` to `(BlockM / kMmaM) * NTiles * kMmaLanes`, so M=32 uses 2 warp rows (16 warps for tile4, 32 for tile8).
- Fixed the host wrapper `amplin_mma_lane_mN_n64_tiled_fullk_cuda_impl` `Threads` to match.
- Added host wrappers `amplin_mma_lane_m32_n64_tile4_shared_a_cuda` and `amplin_mma_lane_m32_n64_tile8_shared_a_cuda`.
- Added `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` registrations in `gptqmodel_ext/amplin/amplin.cpp`.
- Added Python wrappers and `required_ops`/`__all__` entries in `gptqmodel/utils/amplin.py`.
- Added correctness test `test_amplin_mma_lane_m32_n64_tile4_matches_fp32_reference` in `tests/kernels/test_amplin.py` and a fixture `packed_case_n256` with N=512.
- Added focused benchmark `scripts/benchmark_amplin_m32_tile4.py`.

Correctness: `pytest -q tests/kernels/test_amplin.py`: 82 passed. Max FP16 error vs FP32 dequantized reference is ~1e-3.

Focused M=32 benchmark (FP16, GPU UUID `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, seed 20260724, 20 warmup, 100 iters, 5 rounds, median CUDA-event us):

| model | role | K | N | splitk24 | tile4 | tile8 | marlin | fastest |
|---|---|---|---:|---:|---:|---:|---|
| laguna-s-2.1 | expert-down | 1024 | 3072 | 62.3 | 73.5 | 94.6 | 83.5 | splitk24 |
| laguna-s-2.1 | kv/expert-up | 3072 | 1024 | 63.3 | 120.9 | 173.8 | 84.2 | splitk24 |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | 63.0 | 111.5 | 159.3 | 82.9 | splitk24 |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | 77.3 | 110.8 | 158.6 | 83.5 | splitk24 |
| laguna-s-2.1 | dense-up | 3072 | 12288 | 78.9 | 111.1 | 159.5 | 83.1 | splitk24 |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | 72.9 | 174.7 | 266.2 | 83.0 | splitk24 |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | 85.8 | 238.8 | 372.9 | 83.9 | splitk24 |
| laguna-s-2.1 | dense-down | 12288 | 3072 | 98.7 | 302.5 | 479.7 | 83.8 | splitk24 |
| laguna-s-2.1 | router-gate | 3072 | 256 | 62.1 | 108.4 | - | 100.8 | splitk24 |
| glm-5.2 | q-a-proj | 6144 | 2048 | 72.3 | 172.8 | 263.6 | 83.1 | splitk24 |
| glm-5.2 | q-b-proj | 2048 | 4096 | 62.0 | 88.2 | 120.0 | 82.7 | splitk24 |
| glm-5.2 | q-b-proj-large | 2048 | 16384 | 79.9 | 88.9 | 120.5 | 83.6 | splitk24 |
| glm-5.2 | kv-a-proj | 6144 | 128 | 71.7 | - | - | 140.2 | splitk24 |
| glm-5.2 | kv-a-proj-mqa | 6144 | 576 | 72.3 | - | - | 83.0 | splitk24 |
| glm-5.2 | kv-b-proj | 512 | 28672 | 65.2 | 62.6 | 69.5 | 83.2 | tile4 |
| glm-5.2 | o-proj | 16384 | 6144 | 122.8 | 450.3 | 658.0 | 104.0 | splitk24 |
| glm-5.2 | dense-up | 6144 | 12288 | 107.2 | 198.8 | 280.8 | 93.2 | splitk24 |
| glm-5.2 | dense-down | 12288 | 6144 | 103.9 | 350.1 | 506.7 | 94.4 | splitk24 |
| glm-5.2 | moe-up | 6144 | 2048 | 72.0 | 172.5 | 263.0 | 83.4 | splitk24 |
| glm-5.2 | moe-down | 2048 | 6144 | 61.7 | 89.7 | 122.6 | 83.1 | splitk24 |
| glm-5.2 | indexer-wq-b | 2048 | 4096 | 61.5 | 88.2 | 120.1 | 83.1 | splitk24 |
| glm-5.2 | lm-head | 6144 | 154880 | 731.7 | 684.8 | - | 455.6 | tile4 |
| kimi-k2.5 | q-a-proj | 7168 | 1536 | 78.5 | 196.7 | 300.8 | 83.6 | splitk24 |
| kimi-k2.5 | q-b-proj | 1536 | 12288 | 64.9 | 79.5 | 105.2 | 83.0 | splitk24 |
| kimi-k2.5 | kv-a-proj-mqa | 7168 | 576 | 76.4 | - | - | 83.4 | splitk24 |
| kimi-k2.5 | kv-b-proj | 512 | 16384 | 62.5 | 61.8 | 66.7 | 82.4 | tile4 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 83.6 | 220.2 | 335.2 | 87.4 | splitk24 |
| kimi-k2.5 | dense-up | 7168 | 18432 | 156.4 | 230.2 | 321.2 | 117.6 | splitk24 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 132.1 | 504.5 | 736.0 | 116.5 | splitk24 |
| kimi-k2.5 | shared-up | 7168 | 2048 | 77.1 | 193.9 | 298.5 | 83.1 | splitk24 |
| kimi-k2.5 | shared-down | 2048 | 7168 | 61.8 | 88.9 | 121.9 | 83.0 | splitk24 |
| kimi-k2.5 | router-gate | 7168 | 384 | 76.8 | - | - | 108.9 | splitk24 |
| kimi-k2.5 | lm-head | 7168 | 163840 | 901.5 | 809.7 | 912.1 | 554.1 | tile4 |

Key wins at M=32:
- `glm-5.2 kv-b-proj` 512x28672: tile4 62.6us vs Marlin 83.2us (-25%).
- `kimi-k2.5 kv-b-proj` 512x16384: tile4 61.8us vs Marlin 82.4us (-25%).
- `glm-5.2 lm-head` 6144x154880: tile4 685us vs splitk24 732us (-6%) but still behind Marlin 456us.
- `kimi-k2.5 lm-head` 7168x163840: tile4 810us vs splitk24 902us (-10%) but still behind Marlin 554us.

M=32 dense-up/dense-down/o-proj shapes remain fastest with `splitk24`; the M=32 large-N `lm-head` and `dense-up` gaps to Marlin are smaller than before for `lm-head` but still open.

`ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py scripts/benchmark_amplin_m32_tile4.py`: passed.
`git diff --check`: passed.

## 2026-07-25: Comprehensive M=32 head-to-head (splitk24, coop, tile4/tile8, global_a, n32_global_a, Marlin)

Updated `scripts/benchmark_amplin_m32_tile4.py` to also time `mma_lane_m32_global_a`, `mma_lane_m32_n32_global_a`, and `mma_lane_m32_n64_splitk12x2_coop_interleaved` alongside splitk24, tile4/tile8, and Marlin.

FP16 results (GPU UUID `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, 20 warmup, 100 iters, 5 rounds, median CUDA-event us). Only the fastest Amplin path and Marlin are shown for each shape:

| model | role | K | N | fastest_amplin | us | marlin_us | result |
|---|---|---|---:|---|---:|---:|---|
| laguna-s-2.1 | expert-down | 1024 | 3072 | splitk24 | 64.0 | 84.1 | win |
| laguna-s-2.1 | kv/expert-up | 3072 | 1024 | splitk24 | 63.1 | 83.9 | win |
| laguna-s-2.1 | q-proj-6144 | 3072 | 6144 | splitk24 | 64.1 | 82.5 | win |
| laguna-s-2.1 | q-proj-9216 | 3072 | 9216 | splitk24 | 77.8 | 82.5 | win |
| laguna-s-2.1 | dense-up | 3072 | 12288 | splitk24 | 79.9 | 83.4 | win |
| laguna-s-2.1 | o-proj-6144 | 6144 | 3072 | splitk24 | 74.3 | 83.7 | win |
| laguna-s-2.1 | o-proj-9216 | 9216 | 3072 | splitk24 | 87.3 | 82.8 | loss |
| laguna-s-2.1 | dense-down | 12288 | 3072 | splitk24 | 99.8 | 84.1 | loss |
| laguna-s-2.1 | router-gate | 3072 | 256 | splitk24 | 64.0 | 101.5 | win |
| glm-5.2 | q-a-proj | 6144 | 2048 | splitk24 | 73.9 | 82.6 | win |
| glm-5.2 | q-b-proj | 2048 | 4096 | splitk24 | 62.8 | 83.9 | win |
| glm-5.2 | q-b-proj-large | 2048 | 16384 | splitk24 | 81.5 | 84.4 | win |
| glm-5.2 | kv-a-proj | 6144 | 128 | splitk24 | 73.2 | 141.2 | win |
| glm-5.2 | kv-a-proj-mqa | 6144 | 576 | splitk24 | 73.7 | 85.4 | win |
| glm-5.2 | kv-b-proj | 512 | 28672 | global_a | 64.6 | 85.1 | win |
| glm-5.2 | o-proj | 16384 | 6144 | splitk24 | 124.8 | 105.3 | loss |
| glm-5.2 | dense-up | 6144 | 12288 | splitk24 | 109.1 | 94.7 | loss |
| glm-5.2 | dense-down | 12288 | 6144 | splitk24 | 106.0 | 95.5 | loss |
| glm-5.2 | moe-up | 6144 | 2048 | splitk24 | 73.9 | 84.6 | win |
| glm-5.2 | moe-down | 2048 | 6144 | splitk24 | 63.9 | 83.6 | win |
| glm-5.2 | indexer-wq-b | 2048 | 4096 | splitk24 | 63.9 | 84.7 | win |
| glm-5.2 | lm-head | 6144 | 154880 | tile4 | 690.9 | 460.6 | loss |
| kimi-k2.5 | q-a-proj | 7168 | 1536 | splitk24 | 79.9 | 85.5 | win |
| kimi-k2.5 | q-b-proj | 1536 | 12288 | splitk24 | 66.1 | 84.2 | win |
| kimi-k2.5 | kv-a-proj-mqa | 7168 | 576 | splitk24 | 78.1 | 85.0 | win |
| kimi-k2.5 | kv-b-proj | 512 | 16384 | global_a | 62.1 | 83.4 | win |
| kimi-k2.5 | o-proj | 8192 | 7168 | splitk24 | 84.9 | 87.7 | win |
| kimi-k2.5 | dense-up | 7168 | 18432 | splitk24 | 157.3 | 117.9 | loss |
| kimi-k2.5 | dense-down | 18432 | 7168 | splitk24 | 133.6 | 116.7 | loss |
| kimi-k2.5 | shared-up | 7168 | 2048 | splitk24 | 78.3 | 84.4 | win |
| kimi-k2.5 | shared-down | 2048 | 7168 | splitk24 | 64.0 | 84.4 | win |
| kimi-k2.5 | router-gate | 7168 | 384 | splitk24 | 78.8 | 109.9 | win |
| kimi-k2.5 | lm-head | 7168 | 163840 | tile4 | 814.4 | 558.8 | loss |

Revised M=32 status:
- `splitk24` is the fastest Amplin path for the majority of M=32 shapes and already beats Marlin on most small/medium-N modules.
- `m32_global_a` / `m32_n32_global_a` win the `kv-b-proj` shapes (large N, tiny K).
- `tile4` is the fastest Amplin path for `lm-head` M=32 and is closer to Marlin than `splitk24`, but still loses.
- Remaining Marlin wins at M=32 are concentrated in `dense-up`, `dense-down`, `lm-head`, and a few `o-proj` shapes. The `dense-up`/`dense-down` losses are the largest in absolute time and are currently best served by `splitk24`.
- The gap on `dense-up`/`dense-down` is likely because the no-K-split shared-A tile kernels do not create enough independent warps per N tile for moderate N; a small K-split (2/4/8) shared-A N64 tile kernel for M=32 is the next candidate.

`ruff check scripts/benchmark_amplin_m32_tile4.py`: passed.
`git diff --check`: passed.

## 2026-07-25: Failed attempt — M32 N64 16-warp K-split (`splitk16`)

Motivated by Nsight Compute on `kimi-k2.5 dense-up` M=32 K=7168 N=18432:

- `m32_n64_splitk24_pipe2_interleaved`: grid 288, block 768 threads, dynamic shared 98.30 KB, Waves/SM 2.32, Memory Throughput 50.53%, Compute 36.93%, Duration 135.4 us.
- NCU estimates a 33% tail-wave cost and notes low compute/memory utilization (latency-bound), suggesting occupancy is the bottleneck.

A 16-warp variant (`splitk16`) was prototyped to reduce shared memory from 96 KB to 64 KB and target 2 blocks/SM occupancy. It compiled, passed `test_amplin_mma_lane_m32_n64_splitk16_matches_fp32_reference`, and was benchmarked, but it is slower than `splitk24` on nearly all shapes:

| model | role | K | N | splitk24_us | splitk16_us | marlin_us |
|---|---|---|---:|---:|---:|---:|
| laguna-s-2.1 | dense-down | 12288 | 3072 | 98.9 | 151.9 | 84.0 |
| glm-5.2 | o-proj | 16384 | 6144 | 123.2 | 236.3 | 104.2 |
| glm-5.2 | dense-up | 6144 | 12288 | 107.4 | 163.2 | 93.5 |
| glm-5.2 | lm-head | 6144 | 154880 | 731.3 | 1398.7 | 461.1 |
| kimi-k2.5 | o-proj | 8192 | 7168 | 84.3 | 140.8 | 86.9 |
| kimi-k2.5 | dense-up | 7168 | 18432 | 155.9 | 272.1 | 117.0 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 132.8 | 265.3 | 116.1 |
| kimi-k2.5 | lm-head | 7168 | 163840 | 902.5 | 1705.7 | 555.9 |

The prototype was reverted. Losing K-split warp count (24 -> 16) reduced memory-level parallelism enough to outweigh the occupancy gain. The next candidate is to keep 24 K-split warps but reduce shared memory via a tree reduction, or to profile Marlin's scheduling for `dense-up` to identify a different high-level schedule.

`ruff check gptqmodel/utils/amplin.py tests/kernels/test_amplin.py scripts/benchmark_amplin_m32_tile4.py`: passed.
`git diff --check`: passed.

## 2026-07-25: Failed attempt — Marlin-style M32 N64 2-tile A-reuse K-split24

Marlin profiles on `kimi-k2.5 dense-up` (M=32 K=7168 N=18432, 72.77 us, grid 124x128, 167 KB shared, 4 pipeline stages) show it uses a single wave of small blocks that each process a large contiguous N chunk and amortize A loads. Hypothesis: keep Amplin's 24 K-split warps but process 2 contiguous N64 tiles per block, loading A once per K step and reusing it across tiles.

Prototype `m32_n64_splitk24_2tile_pipe2_interleaved` was added to `gptqmodel_ext/amplin/amplin_kernel.cu`, registered in `amplin.cpp`/`amplin.py`, tested in `test_amplin.py`, and benchmarked in `scripts/benchmark_amplin_m32_tile4.py`. It compiled and passed correctness (`84 passed` in `tests/kernels/test_amplin.py`, `ruff` and `git diff --check` clean). Timing (FP16, cuda:0, 100 iters x 5 rounds) on Kimi K2.5 M=32 shapes:

| role | K | N | splitk24_us | splitk24_2tile_us | marlin_us |
|---|---|---:|---:|---:|---:|
| q-a-proj | 7168 | 1536 | 84.7 | 226.4 | 85.6 |
| q-b-proj | 1536 | 12288 | 65.1 | 104.1 | 84.8 |
| kv-b-proj | 512 | 16384 | 64.4 | 89.9 | 84.3 |
| o-proj | 8192 | 7168 | 85.2 | 275.8 | 88.0 |
| dense-up | 7168 | 18432 | 157.3 | 549.2 | 117.4 |
| dense-down | 18432 | 7168 | 133.4 | 545.4 | 117.3 |
| shared-up | 7168 | 2048 | 77.3 | 198.8 | 83.5 |
| shared-down | 2048 | 7168 | 63.6 | 108.1 | 85.6 |
| router-gate | 7168 | 384 | 78.3 | 200.1 | 109.2 |
| lm-head | 7168 | 163840 | 903.0 | 3631.0 | 556.5 |

The 2-tile kernel is 1.5–4× slower than the single-tile `splitk24` and never beats Marlin. The likely causes are extra register pressure from `tile_accumulators[2][8]`, per-tile `__syncthreads` serialization, and a 50% smaller grid that under-utilizes memory-level parallelism. A-reuse is not the binding bottleneck; the `m32_splitk24` kernel is already latency-bound, not A-bandwidth-bound.

This prototype was not committed. The remaining M32 large-N gap requires a different schedule: smaller blocks (4–8 warps), a 2–4 stage `cp.async` pipeline for A/B, and a larger contiguous N chunk per CTA, i.e. a Marlin-style micro-kernel rather than incremental tuning of the 24-warp lane-MMA design.
