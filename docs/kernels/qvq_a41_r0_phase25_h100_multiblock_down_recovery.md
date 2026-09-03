# QVQ A41/R0 Phase 25: multiblock H100 down recovery

## Decision

Phase 25 is promoted for the measured physical NVIDIA H100 Llama 3.2 1B
down projection. It factorizes the exact fused split-16 N=2048 recovery into
eight independent 256-column low-stage blocks and four high-stage blocks per
logical row. The complete MLP improves **1.0461x** geometric mean over Phase
23 and all 20 W2-W3.5 by M1/M2/M4/M8/M16 cells improve.

The production executable change is commit `10d0ee32`. H200, non-H100
devices, and unsupported shapes retain the preceding recovery path.

## Why another launch wins

Phase 23 used one 1,024-thread block per logical row for all work after the
tensor-core kernel:

```text
16 ordered partial planes
  -> deterministic reduction
  -> 11 N=2048 Hadamard stages
  -> output scale and bias
  -> FP16 output
```

At M1 that is one useful block on a 132-SM H100. Phase 25 restores one global
workspace boundary but exposes the independent part of the transform:

```text
low stage:  grid = (8, M), block = 256
  ordered reduction for one 256-column tile
  initial FP16-emulation boundary
  Hadamard bits 1,2,4,8,16,32,64,128
  write FP32 workspace

high stage: grid = (4, M), block = 64
  one thread owns the same within-tile column in all eight tiles
  Hadamard bits 256,512,1024
  output scale, bias, and FP16 store
```

This raises useful M1 work from one block to twelve blocks. The extra launch
is worthwhile because it removes the much larger single-block dependency and
barrier tail.

## Exact math

For output column `j`, both paths first execute the fixed split reducer in the
same order:

\[
r_j=((((0+P_{0,j})+P_{1,j})+\cdots)+P_{15,j}).
\]

The initial historical FP16 boundary and optional first normalization are
then applied exactly as Phase 23. A Walsh-Hadamard stage with distance `b`
only pairs `j` and `j xor b`. For every `b < 256`, both indices are in the
same aligned 256-column tile. Therefore the first eight stages are independent
across the eight tiles.

The FP32 workspace is a pure ownership boundary after bit 128; it does not
round or transform a value. In the high kernel, one thread loads:

\[
(x_{0,c},x_{1,c},\ldots,x_{7,c})
\]

for fixed within-tile column `c`, then performs the remaining tile-index
butterflies in ascending order. Every call to
`round_fp16_unless_overflow`, final normalization, output-scale multiply,
bias add, and FP16 store remains in the same mathematical location.

The production contract is consequently bit equality, not a tolerance.

## Specialization and storage budget

This phase adds two fixed device kernels. There is no multiplication by rate,
M, dtype, alternative bank, or split policy:

```text
new device specializations = 2 fixed N=2048 kernels
new host launch routes      = 1 bool-selected operator route
```

Persistent VRAM addition is zero. Transient FP32 workspace is:

\[
4MN = 8\text{ KiB at M1} \ldots 128\text{ KiB at M16}.
\]

The existing sixteen `[16,2048]` FP32 split planes remain 2 MiB and are not
duplicated. CUDA Graph capture retains the transient allocation in its graph
pool as expected.

## Isolated H100 benchmark

Both implementations were present in the same JIT binary. Each row used 30
warmups, 100 samples, and 50 CUDA Graph replays per sample, timed with CUDA
events. The physical H100 UUID was
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`; the formal idle gate required
0% utilization and zero MiB in use.

| Rate | M | MKN | Phase-23 recovery us | Phase-25 recovery us | Recovery speedup | Phase-23 down site us | Phase-25 down site us | Site speedup | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x8192x2048 | 7.312 | 4.463 | 1.6385x | 20.914 | 17.877 | 1.1699x | Yes |
| W2 | 2 | 2x8192x2048 | 6.866 | 4.510 | 1.5223x | 20.429 | 17.693 | 1.1547x | Yes |
| W2 | 4 | 4x8192x2048 | 6.991 | 4.546 | 1.5379x | 20.124 | 17.742 | 1.1343x | Yes |
| W2 | 8 | 8x8192x2048 | 6.993 | 4.828 | 1.4484x | 20.383 | 18.141 | 1.1236x | Yes |
| W2 | 16 | 16x8192x2048 | 7.218 | 4.853 | 1.4873x | 20.557 | 17.988 | 1.1428x | Yes |
| W2.5 | 1 | 1x8192x2048 | 6.857 | 4.369 | 1.5697x | 20.205 | 17.662 | 1.1440x | Yes |
| W2.5 | 2 | 2x8192x2048 | 7.026 | 4.584 | 1.5328x | 20.455 | 17.716 | 1.1546x | Yes |
| W2.5 | 4 | 4x8192x2048 | 6.874 | 4.665 | 1.4735x | 20.253 | 17.816 | 1.1368x | Yes |
| W2.5 | 8 | 8x8192x2048 | 7.104 | 4.774 | 1.4882x | 20.450 | 18.036 | 1.1338x | Yes |
| W2.5 | 16 | 16x8192x2048 | 7.145 | 4.956 | 1.4416x | 20.495 | 18.178 | 1.1275x | Yes |
| W3 | 1 | 1x8192x2048 | 6.918 | 4.577 | 1.5113x | 19.285 | 16.896 | 1.1414x | Yes |
| W3 | 2 | 2x8192x2048 | 6.880 | 4.470 | 1.5391x | 19.287 | 16.813 | 1.1472x | Yes |
| W3 | 4 | 4x8192x2048 | 6.989 | 4.510 | 1.5497x | 19.202 | 16.883 | 1.1374x | Yes |
| W3 | 8 | 8x8192x2048 | 7.005 | 4.587 | 1.5272x | 19.396 | 17.158 | 1.1304x | Yes |
| W3 | 16 | 16x8192x2048 | 7.186 | 4.746 | 1.5140x | 19.381 | 17.157 | 1.1296x | Yes |
| W3.5 | 1 | 1x8192x2048 | 6.900 | 4.293 | 1.6073x | 20.270 | 17.697 | 1.1454x | Yes |
| W3.5 | 2 | 2x8192x2048 | 6.885 | 4.511 | 1.5262x | 20.075 | 17.654 | 1.1372x | Yes |
| W3.5 | 4 | 4x8192x2048 | 6.818 | 4.452 | 1.5312x | 20.241 | 17.889 | 1.1315x | Yes |
| W3.5 | 8 | 8x8192x2048 | 6.971 | 4.493 | 1.5516x | 20.348 | 18.014 | 1.1295x | Yes |
| W3.5 | 16 | 16x8192x2048 | 7.113 | 4.714 | 1.5090x | 20.588 | 18.307 | 1.1247x | Yes |

Geometric means are **1.5247x** for recovery and **1.1388x** for the down
decode-plus-recovery site. All 20 rows are bit-exact and faster.

## Complete Llama 3.2 1B MLP

The production runtime was measured with 30 warmups, 200 samples, and 50
warmed CUDA Graph replays per sample. Marlin W4 and Machete W4 were measured
freshly in the same process. `Better` is the strict median comparison against
the committed Phase-23 production artifact.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 23 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 59.039 | 1.705 | 0.492x | 0.852x | 1.0437x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 57.996 | 3.471 | 0.537x | 0.862x | 1.0474x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 57.897 | 6.955 | 0.542x | 0.870x | 1.0450x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 58.247 | 13.826 | 0.504x | 0.867x | 1.0481x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 59.083 | 27.260 | 0.550x | 0.857x | 1.0394x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 59.925 | 1.680 | 0.484x | 0.839x | 1.0473x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 58.975 | 3.414 | 0.529x | 0.848x | 1.0444x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 58.797 | 6.848 | 0.534x | 0.857x | 1.0449x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 59.126 | 13.620 | 0.496x | 0.854x | 1.0491x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 59.997 | 26.845 | 0.542x | 0.844x | 1.0428x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 57.033 | 1.765 | 0.509x | 0.882x | 1.0470x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 56.126 | 3.587 | 0.555x | 0.891x | 1.0442x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 55.843 | 7.210 | 0.562x | 0.902x | 1.0491x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 56.182 | 14.334 | 0.522x | 0.899x | 1.0482x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 56.922 | 28.295 | 0.571x | 0.890x | 1.0473x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 59.644 | 1.688 | 0.487x | 0.843x | 1.0465x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 58.809 | 3.423 | 0.530x | 0.850x | 1.0455x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 58.662 | 6.864 | 0.535x | 0.859x | 1.0513x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 58.845 | 13.685 | 0.498x | 0.858x | 1.0458x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 59.787 | 26.939 | 0.544x | 0.847x | 1.0459x | Yes |

Geometric means are **1.0461x** versus Phase 23, **0.5254x** versus Marlin
W4, **0.8636x** versus Machete W4, and **2.3997x** versus ordinary per-module
QVQ. The W4 columns are figurative dense-equivalent throughput baselines, not
equal-rate or equal-quality comparisons.

## Nsight Compute and SASS evidence

The matched commands used Nsight Compute 2026.2.1 with 19 hardware-counter
replay passes:

```text
ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --section InstructionStats --section MemoryWorkloadAnalysis \
  -o <report> -- python scripts/profile_qvq_phase25_down_recovery.py \
  --variant scalar|multiblock --m 1 --idle-memory-mib 4
```

| Metric | Phase-23 scalar | Phase-25 low | Phase-25 high |
|:--|--:|--:|--:|
| Grid x block | 1 x 1024 | 8 x 256 | 4 x 64 |
| Registers/thread | 32 | 31 | 31 |
| Shared memory/block | 8.448 KiB dynamic | 1.056 KiB static | 0 |
| Local spills | 0 | 0 | 0 |
| Executed warp instructions | 29,664 | 15,168 | 2,144 |
| NCU replay duration | 9.184 us | 3.552 us | 4.096 us |
| Eligible warps/scheduler/cycle | 1.315 | 0.180 | 0.070 |
| Active warps | 46.79% | 11.72% | 3.08% |
| Long-scoreboard cycles/issued instruction | 4.619 | 3.845 | 9.745 |
| DRAM throughput | 0.714% | 1.610% | 0.349% |
| Combined memory throughput | 3.666% | 9.440% | 8.011% |

The two new kernels execute **17,312** warp instructions combined, 41.6%
fewer than the scalar kernel. Source-correlated SASS accounts for the retained
work: the low stage executes 1,856 `FADD`, 1,024 `LDG`, 960 `F2FP`, 960
`HADD2`, and 896 each of `LDS`/`STS`; the high stage executes 256 `FADD`, 192
`LDG`, 384 `F2FP`, and 320 `HADD2`. These are executed profiler counts, not
source estimates.

Neither path is HBM-bound. The high stage is now a tiny four-block,
long-scoreboard-limited tail, but the normal CUDA-event result proves that the
additional device parallelism still wins end to end.

Reports are kept outside Git:

```text
/root/qvq-profiler-artifacts/phase25-down-recovery/scalar.ncu-rep
/root/qvq-profiler-artifacts/phase25-down-recovery/multiblock.ncu-rep
```

## Nsight Systems graph evidence

Nsight Systems 2026.4.1 captured the production W3/M1 CUDA Graph with node
tracing enabled:

```text
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --resolve-symbols=false --wait=primary -o <report> -- \
  python scripts/profile_qvq_phase19_full_mlp.py \
  --bits 3 --m 1 --warmup 20 --cuda-graph --idle-memory-mib 4
```

The Phase-25 graph has nine visible QVQ nodes, one more than Phase 23. Its
instrumented down tail is:

| Node | Nsight Systems duration |
|:--|--:|
| split-16 down WGMMA | 12.768 us |
| multiblock ordered-reduction/low transform | 1.824 us |
| multiblock high transform/recovery | 1.791 us |

The full nine-node instrumented GPU sum is 58.495 us. Phase 23's corresponding
fused recovery was 6.688 us in its matched historical trace; Phase 25's two
stages sum to 3.615 us here. These trace durations explain graph structure and
are not substituted for CUDA-event production timing.

Report:

```text
/root/qvq-profiler-artifacts/phase25-down-recovery/full_mlp_graph_nodes.nsys-rep
```

## Validation

- 20 rate/M isolated rows are bit-exact under repeated CUDA Graph replay;
- 20 complete MLP rows are bit-exact to ordinary per-module QVQ;
- scale modes 3 and 4, bias present/absent, and all five target M values pass;
- a late-FP16-overflow rescue case passes on a non-default CUDA stream;
- 130 focused CUDA/grouped-runtime/Hopper-plan tests pass (71 plus the
  independent 59-case grouped-P32 suite);
- the real Llama layer test preserves logits within the existing 2e-3 dense
  gate, exact repeated logits, exact generated tokens, and cache lifecycle;
- runtime telemetry proves the specialized multiblock path fires;
- CUDA Graph capture/replay is bit-exact;
- compilation was limited to Ninja `-j4`, one NVCC host thread, and one split
  compile partition.

Compute Sanitizer is not installed on this host, so memcheck/synccheck were not
run. This is recorded as a validation limitation rather than a pass.

Artifacts:

- `artifacts/a41_phase25_h100/multiblock_down_recovery_w3_probe.json`
- `artifacts/a41_phase25_h100/multiblock_down_recovery_experiment.json`
- `artifacts/a41_phase25_h100/production_mlp_multiblock_down_recovery_vs_phase23.json`
