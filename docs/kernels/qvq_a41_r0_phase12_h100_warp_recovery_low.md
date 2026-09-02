# QVQ A41/R0 Phase 12: H100 warp-local recovery low stage

Phase 12 keeps the first five butterflies of the exact paired gate/up output
recovery inside each H100 warp. The accepted kernel replaces five
shared-memory exchanges and block-wide barriers with XOR lane shuffles, while
preserving every historical FP16-emulation rounding boundary and the
overflow-preserving FP32 workspace contract.

The retained low stage removes **30.5%** of its executed warp instructions and
improves isolated paired recovery at every M by **1.0573x** geometric mean.
The complete W2--W3.5 Llama 3.2 1B MLP improves all 20 rate/M cells and is
**1.0052x** faster than accepted Phase 11, with a latency range of
**64.46--70.21 us**.

This is an execution-only change. Canonical P32 payloads, quantized weights,
scales, output ordering, FP16 rounding, down preconditioning and decode,
tensor layouts, CUDA Graph node count, operation-local global workspace, and
all non-H100 fallbacks are unchanged.

## Exact math and lane ownership

The N=8192 Hadamard is factored into 32 independent 256-column low tiles and
one 32-tile high transform. For every low-stage butterfly pair `(a,b)`, the
required result remains:

\[
s=R(a+b), \qquad d=R(a-b),
\]

where `R` is the existing `round_fp16_unless_overflow` operation: round to
FP16 when finite, but retain an overflowing intermediate in FP32. The result
therefore cannot be represented by unconditional half arithmetic.

Each 256-thread block still owns one 256-column tile. For butterfly bits
`1,2,4,8,16`, both members of a pair are in the same warp. Each lane obtains
the peer without shared memory:

```cpp
peer = __shfl_xor_sync(0xffffffffu, value, bit);
value = (lane & bit) == 0
    ? round_fp16_unless_overflow(value + peer)
    : round_fp16_unless_overflow(peer - value);
```

The upper lane uses `peer-value`, not `value-peer`, so it reproduces the
original lower-minus-upper orientation. The same rounding function executes
after every stage, and values stay FP32 throughout.

Only bits `32,64,128` cross warp boundaries. Values are then written to the
same bank-padded shared array as before, and the last three stages execute in
the original ascending order. The unchanged high kernel consumes the same
FP32 `[2*M,8192]` workspace and performs the remaining five tile butterflies,
normalization, `SV`, optional bias, and final FP16 store.

## Candidate history

Phase 12 first tested the boundary deletion proposed by Phase 11. Both
designs were exact and CUDA Graph safe, but synchronization cost was larger
than the launch/workspace cost they removed.

### Cooperative-grid low/high fusion

All `32*M` low blocks wrote the workspace, synchronized the cooperative grid,
and only four blocks per row executed the high stage in place.

| M | Existing two-kernel us | Cooperative us | Speedup | Decision |
|---:|---:|---:|---:|:--|
| 1 | 4.351 | 5.316 | 0.818x | rejected |
| 2 | 4.474 | 5.505 | 0.813x | rejected |
| 4 | 4.494 | 5.485 | 0.819x | rejected |
| 8 | 4.543 | 5.670 | 0.801x | rejected |
| 16 | 4.859 | 6.325 | 0.768x | rejected |

### Row-local completion-counter fusion

Low blocks issued a device fence and incremented a row-local atomic counter;
the last block executed the high transform. This avoided a cooperative grid
barrier but added a counter clear, fence, atomic, and the union of low/high
register pressure.

| M | Existing two-kernel us | Completion-counter us | Speedup | Decision |
|---:|---:|---:|---:|:--|
| 1 | 4.296 | 5.131 | 0.837x | rejected |
| 2 | 4.320 | 5.227 | 0.826x | rejected |
| 4 | 4.463 | 5.265 | 0.848x | rejected |
| 8 | 4.513 | 5.635 | 0.801x | rejected |
| 16 | 4.785 | 6.316 | 0.757x | rejected |

The rejected source was fully reverted. The promoted design instead removes
intra-block synchronization from the existing low kernel and retains the
proven two-kernel handoff.

## Matched SASS and scheduler analysis

Nsight Compute 2026.2.1 captured one M1 original low launch and one M1
warp-local low launch from the same binary at production source commit
`c55cc361`. The reports, raw metric exports, and source-correlated SASS
exports are stored outside Git under:

```text
/root/qvq-profiler-artifacts/phase12-low-high-fusion/
```

The total is the raw NCU `inst_executed` metric. Opcode rows are
source-correlated executed warp instructions; NCU leaves 512 instructions in
each launch without a source-correlated opcode row.

| Metric/opcode | Shared low | Warp-local low | Change |
|:--|--:|--:|--:|
| total executed warp instructions | 107,520 | 74,752 | **-30.5%** |
| `LDS` | 7,168 | 2,048 | **-71.4%** |
| `STS` | 7,168 | 2,048 | **-71.4%** |
| `BAR` | 4,608 | 2,048 | **-55.6%** |
| `BSYNC` | 7,168 | 2,048 | **-71.4%** |
| `LEA` | 9,728 | 4,608 | **-52.6%** |
| `SHFL` | 0 | 2,560 | register exchange |
| `FADD` | 6,656 | 6,656 | unchanged |
| `F2FP` | 7,680 | 7,680 | unchanged |
| `HADD2` | 7,680 | 7,680 | unchanged |
| registers/thread | 28 | 18 | **-35.7%** |
| static shared memory | 1,056 B | 1,056 B | unchanged |
| local/shared spills | 0 / 0 | 0 / 0 | unchanged |
| NCU duration | 3.552 us | 3.456 us | -2.7% |

The arithmetic instruction counts are intentionally unchanged because the
optimization changes data exchange, not math or narrowing. Active warps are
nearly equal (12.44% versus 12.43%), while eligible warps per cycle fall from
0.181 to 0.153 and long-scoreboard wait rises from 3.46 to 4.88 instructions
per issue-active cycle. That dependency tradeoff explains why removing 30.5%
of instructions yields a 5.7% isolated geometric-mean win rather than a 30%
latency win. DRAM active cycles remain negligible (0.85% versus 0.86%); this
stage is instruction/dependency limited, not HBM limited.

The matched profiler commands used the physical H100 UUID, strict zero-MiB
idle admission, `--profile-from-start off`, one named-kernel launch, and the
`SpeedOfLight`, `LaunchStats`, `Occupancy`, `SchedulerStats`,
`WarpStateStats`, and `InstructionStats` section sets. The profile driver
places CUDA profiler markers around only the requested recovery variant.

## Launches, traffic, and VRAM

| Property | Phase 11 | Phase 12 |
|:--|--:|--:|
| low-stage threads/block | 256 | 256 |
| low-stage warps/block | 8 | 8 |
| low-stage blocks/row/projection | 32 | 32 |
| low shared butterfly stages | 8 | **3** |
| high stage | half2 | unchanged |
| CUDA Graph nodes | 2 | 2 |
| global workspace | `2*M*8192*4` bytes | unchanged |
| persistent VRAM added | 0 | 0 |

The existing transient paired-recovery workspace is 64 KiB at M1 and 1 MiB
at M16. Phase 12 neither changes its size nor introduces a persistent buffer,
checkpoint field, or duplicate packed representation.

## Isolated paired-recovery benchmark

Both paths are compiled into the same extension and timed in one process.
Times are warmed CUDA Graph replays measured by CUDA events: 30 warmups, 300
samples, and 50 replays per sample. The table includes both low and unchanged
high stages. Every output is FP16 bit-exact.

| M | Shared-low p50 us | Warp-local p50 us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 6.166 | 5.778 | 1.067x | Yes |
| 2 | 6.220 | 6.050 | 1.028x | Yes |
| 4 | 6.321 | 6.167 | 1.025x | Yes |
| 8 | 6.859 | 6.402 | 1.071x | Yes |
| 16 | 7.795 | 7.108 | 1.097x | Yes |

Geometric-mean speedup is **1.0573x**.

## Complete Llama 3.2 1B MLP

The formal matrix uses 30 warmups, 150 CUDA-event samples, and 50 CUDA Graph
replays per sample. It includes gate/up P32, paired recovery, exact SiLU,
FP16 product/down precondition, and down P32/recovery. `vs` is comparator
latency divided by QVQ latency, so a value below one means the W4 comparator
remains faster. `Better` compares against the committed Phase-11 artifact;
`No` would be recorded as a regression.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 11 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 67.988 | 1.481 | 0.435x | 0.745x | 1.0026x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 67.718 | 2.973 | 0.460x | 0.739x | 1.0010x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 68.439 | 5.883 | 0.458x | 0.735x | 1.0025x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.164 | 11.643 | 0.423x | 0.729x | 1.0028x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.458 | 24.987 | 0.505x | 0.771x | 1.0077x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.714 | 1.465 | 0.430x | 0.737x | 1.0058x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.847 | 2.924 | 0.452x | 0.726x | 1.0030x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.249 | 5.815 | 0.452x | 0.727x | 1.0043x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.040 | 11.498 | 0.418x | 0.720x | 1.0051x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.231 | 24.691 | 0.499x | 0.762x | 1.0098x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 69.208 | 1.454 | 0.427x | 0.732x | 1.0006x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 69.052 | 2.916 | 0.451x | 0.724x | 1.0040x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.508 | 5.793 | 0.451x | 0.724x | 1.0055x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.206 | 11.471 | 0.417x | 0.718x | 1.0048x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.422 | 24.619 | 0.497x | 0.759x | 1.0098x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.459 | 1.470 | 0.432x | 0.740x | 1.0070x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.789 | 2.927 | 0.453x | 0.727x | 1.0048x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.388 | 5.803 | 0.451x | 0.725x | 1.0048x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.625 | 11.566 | 0.420x | 0.724x | 1.0070x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.908 | 24.814 | 0.501x | 0.765x | 1.0119x | Yes |

Geometric means:

- **1.0052x** versus accepted Phase 11;
- **2.0667x** versus ordinary per-module QVQ;
- **0.4507x** versus Marlin W4;
- **0.7364x** versus Machete W4.

The W4 comparisons are figurative dense-equivalent efficiency baselines.
They do not claim equal quantization quality or equal compressed decode work.

## Correctness and validation

- 60 random exactness cases cover M=1/2/4/8/16, three seeds, scale modes 3
  and 4, and ten repetitions for both shared and warp-local low paths.
- CUDA Graph replay, argument guards, and overflow-preserving finite-FP16
  cases pass for both normalization modes.
- The focused native paired-recovery suite reports 66 passed tests.
- Runtime lifecycle and the real Llama 3.2 layer/logits/cache test report two
  passed tests and assert warp-local launch telemetry.
- Python compilation, Ruff, and the changed-file whitespace check pass.

Compilation used no more than four Ninja jobs, one NVCC host thread, and one
CUDA 13 split-compile partition:

```text
MAX_JOBS=4
NINJAFLAGS=-j4
CMAKE_BUILD_PARALLEL_LEVEL=4
NVCC_THREADS=1
GPTQMODEL_QVQ_NVCC_THREADS=1
GPTQMODEL_NVCC_SPLIT_COMPILE=1
```

Artifacts and drivers:

- `artifacts/a41_phase12_h100/warp_recovery_low_experiment.json`
- `artifacts/a41_phase12_h100/production_mlp_warp_recovery_vs_baselines.json`
- `scripts/benchmark_qvq_phase12_warp_recovery_low.py`
- `scripts/profile_qvq_phase12_warp_recovery_low.py`

Both committed benchmark artifacts identify production source commit
`c55cc361`. The binary Nsight reports and CSV exports remain outside Git.

## Next experiment

The two exact low/high fusion attempts show that deleting this particular
launch boundary is not profitable on H100 when it requires a grid barrier or
device-fence/atomic completion protocol. Further recovery work should target
a representation-level reduction inside the unchanged high stage, or a
larger producer/consumer fusion where synchronization disappears naturally.
Small arithmetic substitutions and wide DSM reductions remain low priority.
