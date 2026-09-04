# Phase 79: H100 Qwen W2/W2.5 down decoded-level prefetch

Phase 79 enables the packed decoded-level prefetch schedule for the W2 and
W2.5 Qwen3.8-27B down projection at the exact physical-H100 geometry

$$
(K,N,split)=(17408,5120,34).
$$

It improves every affected complete-MLP row while retaining the existing
split-34 atomic accumulation and dense-P32 accuracy contract.

## Schedule

The 1088 K16 tiles divide into 34 partitions of 32 K16 tiles, or two K256
stages per block.  With 80 N64 output blocks the launch contains

$$
80\times34=2720\text{ blocks}.
$$

The candidate decodes eight FP16 level values into four packed register pairs
before the fragment-reuse dependency wait.  It does not change state, bank,
level, WGMMA, or atomic-add math.  Dispatch requires unordered split output,
transition width four or five, the exact shape and split above, and physical
device name `NVIDIA H100`.  W3 retains Phase 77's deterministic ordered
split-17 path.

## CUDA Graph contract

The low-level P32 operation and complete MLP capture and replay successfully.
W2/W2.5 down already use unordered floating-point atomic accumulation, so two
complete-MLP executions are not guaranteed bit-identical.  The all-rate graph
test therefore requires finite replay and the established dense-P32 tolerance
for W2/W2.5, while W3's ordered path remains bit-exact across repeated replay.
This distinction is pre-existing and is not hidden by the prefetch schedule.

Runtime telemetry exposes `h100_qwen_down_decode_prefetch_launches`, retaining
the more specific W3 counter as well.

## H100 benchmark

The physical device UUID was
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`.  Timings use 20 warmups, 60
samples, and 50 CUDA Graph replays per sample, measured with CUDA events.
Marlin and Machete are figurative W4 projection-sum baselines.  `last/new`
compares with Phase 78.

| W | M | MKN: gate/up x2; down | new us | effective TFLOP/s | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2; 1x17408x5120 | 181.856 | 2.941 | 0.3156x | 0.6185x | 1.0099x | yes |
| 2 | 2 | 2x5120x17408 x2; 2x17408x5120 | 183.188 | 5.839 | 0.3196x | 0.6050x | 1.0099x | yes |
| 2 | 4 | 4x5120x17408 x2; 4x17408x5120 | 183.738 | 11.642 | 0.3192x | 0.6034x | 1.0128x | yes |
| 2 | 8 | 8x5120x17408 x2; 8x17408x5120 | 183.674 | 23.292 | 0.3145x | 0.6035x | 1.0100x | yes |
| 2 | 16 | 16x5120x17408 x2; 16x17408x5120 | 184.037 | 46.493 | 0.3660x | 0.6034x | 1.0099x | yes |
| 2.5 | 1 | 1x5120x17408 x2; 1x17408x5120 | 183.019 | 2.922 | 0.3136x | 0.6146x | 1.0055x | yes |
| 2.5 | 2 | 2x5120x17408 x2; 2x17408x5120 | 184.088 | 5.810 | 0.3180x | 0.6020x | 1.0077x | yes |
| 2.5 | 4 | 4x5120x17408 x2; 4x17408x5120 | 184.484 | 11.595 | 0.3179x | 0.6010x | 1.0077x | yes |
| 2.5 | 8 | 8x5120x17408 x2; 8x17408x5120 | 184.826 | 23.147 | 0.3125x | 0.5998x | 1.0072x | yes |
| 2.5 | 16 | 16x5120x17408 x2; 16x17408x5120 | 185.308 | 46.174 | 0.3635x | 0.5993x | 1.0060x | yes |

The ten cells improve **1.00864x geometrically**.  W2 improves 1.01047x and
W2.5 improves 1.00681x, with ten of ten wins.  Mean absolute error is at most
`1.533e-8` and maximum absolute error is `4.838e-8` against the same-payload
dense-P32 Torch oracle.

The distilled benchmark is
`artifacts/a41_phase79_h100/qwen38_27b_w2_w25_down_prefetch.json`.
