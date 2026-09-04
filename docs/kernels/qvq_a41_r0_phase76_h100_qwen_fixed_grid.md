# Phase 76: fixed H100 Qwen gate/up grid

Phase 76 replaces the generic flattened-work mapping in Qwen3.8-27B's
equal-width grouped gate/up P32 launch with a compile-time rectangular grid.
It improves all 15 W2--W3 by M1--M16 complete-MLP cells and lowers geometric
latency by 3.41 percent.

## Mapping

Phase 75 has two equal children, 272 N64 blocks per child, and five K splits.
The generic ordered kernel flattened those dimensions into 2720 work items
and reconstructed them inside every thread:

$$
segment=search(work),
$$

$$
split=\left\lfloor local\_work/272\right\rfloor,
\qquad
n64=local\_work-272\,split.
$$

That mapping is needed for unequal QKV segments, but it is redundant for
equal gate/up children.  Phase 76 launches the exact rectangular grid

$$
grid=(272,2,5)
$$

and obtains the same coordinates directly:

$$
n64=blockIdx.x,\qquad segment=blockIdx.y,\qquad split=blockIdx.z.
$$

The child-local ordered partial address is

$$
offset=segment\,(5\cdot16\cdot17408)
      +split\,(16\cdot17408)
      +row\,(17408)+column.
$$

No P32 state, selector, pseudo-random code, level lookup, WGMMA issue, FP32
addition, or recovery order changes.  CTA count and shared-memory payload are
also unchanged.  `h100_qwen_fixed_ordered_grid_launches` exposes selection.

## Matched Nsight Compute result

Nsight Compute 2026.4.1 profiled W3/M1 on physical H100 with the same section
set before and after the mapping specialization.

| Metric | Phase 75 | Phase 76 | Change |
|:--|--:|--:|--:|
| Gate/up P32 duration | 99.904 us | 97.120 us | -2.79% |
| Executed warp instructions | 59,754,794 | 57,845,250 | -3.20% |
| Registers/thread | 73 | 71 | -2 |
| Grid blocks | 2,720 | 2,720 | unchanged |
| Shared memory/block | 46.848 KiB | 46.848 KiB | unchanged |
| SM throughput | 62.59% | 63.01% | +0.42 points |
| DRAM throughput | 32.05% | 31.50% | -0.55 points |
| Long-scoreboard cycles/issued instruction | 1.685 | 1.689 | effectively unchanged |

The measured latency tracks the instruction reduction, while memory and
scoreboard behavior stay stable.  This validates removal of uniform mapping
work rather than a cache or clock artifact.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays/sample.
`last/new` compares with Phase 75.  Marlin and Machete are figurative W4
projection-sum baselines; ratios below one mean the W4 baseline is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 185.677 | 0.309x | 0.606x | 1.0342x | yes |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 186.918 | 0.313x | 0.593x | 1.0350x | yes |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 187.537 | 0.313x | 0.591x | 1.0357x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 187.350 | 0.308x | 0.592x | 1.0409x | yes |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 186.736 | 0.361x | 0.595x | 1.0445x | yes |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 184.548 | 0.311x | 0.609x | 1.0385x | yes |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 186.005 | 0.315x | 0.596x | 1.0384x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 186.520 | 0.314x | 0.594x | 1.0381x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 187.059 | 0.309x | 0.593x | 1.0393x | yes |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 187.538 | 0.359x | 0.592x | 1.0421x | yes |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 187.149 | 0.307x | 0.601x | 1.0292x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 189.342 | 0.309x | 0.585x | 1.0256x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 189.675 | 0.309x | 0.585x | 1.0260x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 189.728 | 0.304x | 0.584x | 1.0304x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 190.194 | 0.354x | 0.584x | 1.0321x | yes |

The geometric speedup is **1.0353x over Phase 75**, with 15 of 15 wins.
Mean absolute error remains at most `1.544e-8` and maximum absolute error is
`4.838e-8` against the same-payload dense-P32 Torch oracle.  The raw distilled
result is `artifacts/a41_phase76_h100/qwen38_27b_fixed_gate_up_grid.json`.

## Rejected adjacent-output probe

An ordered W3 N128 probe halved the block count by adding a second consumer
warpgroup to every CTA.  It was exact, but all five M rows regressed by
0.15--0.34 percent.  The extra resource and synchronization cost exceeded
the shared input/TMA setup savings, so no N128 source was retained.

## Next target

The fixed W3 kernel still executes 57.85 million warp instructions at only
about 32 percent peak DRAM throughput.  Further progress must reduce the
state/PGC/level-address instruction chain without adding a level-index
dependency or enlarging the per-block shared table.
