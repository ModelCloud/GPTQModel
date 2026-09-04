# Phase 77: H100 Qwen W3 down decoded-level prefetch

Phase 77 enables the existing exact register-prefetched W3 decoder for the
Qwen3.8-27B down projection at the measured H100 geometry

$$
(K,N,split)=(17408,5120,17).
$$

It improves all five complete-MLP row counts without changing P32 payloads,
split arithmetic, FP32 reduction order, recovery order, or graph topology.

## Schedule

The down projection contains 1088 K16 tiles.  Ordered split 17 assigns

$$
1088/17=64\text{ K16 tiles}=4\text{ K256 stages}
$$

to every block.  Its 80 N64 output blocks launch

$$
80\times17=1360\text{ blocks}.
$$

Before Phase 77, this shape decoded each eight-value WGMMA input fragment
after waiting for the old fragment register.  The accepted path computes the
four state values, pseudo-random-code products, and eight shared level loads
into an independent register fragment first.  Only the final packed fragment
write remains behind the exact fragment-reuse wait.  WGMMA issue and ordered
split accumulation remain unchanged.

The dispatch is restricted to transition width six, the physical NVIDIA H100,
and the exact Qwen down geometry above.  Other rates, devices, shapes, and
split counts retain their prior kernels.  Runtime telemetry exposes
`h100_qwen_w3_down_decode_prefetch_launches`.

## H100 benchmark

The physical H100 UUID was
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`.  Timings use 20 warmups, 60
samples, and 50 CUDA Graph replays per sample, measured entirely with CUDA
events.  Marlin and Machete are figurative W4 projection-sum baselines.
`last/new` compares with Phase 76.

| W | M | MKN: gate/up x2; down | new us | effective TFLOP/s | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|
| 3 | 1 | 1x5120x17408 x2; 1x17408x5120 | 185.685 | 2.880 | 0.3091x | 0.6058x | 1.0079x | yes |
| 3 | 2 | 2x5120x17408 x2; 2x17408x5120 | 187.711 | 5.698 | 0.3119x | 0.5904x | 1.0087x | yes |
| 3 | 4 | 4x5120x17408 x2; 4x17408x5120 | 188.570 | 11.344 | 0.3110x | 0.5879x | 1.0059x | yes |
| 3 | 8 | 8x5120x17408 x2; 8x17408x5120 | 188.708 | 22.671 | 0.3061x | 0.5874x | 1.0054x | yes |
| 3 | 16 | 16x5120x17408 x2; 16x17408x5120 | 188.622 | 45.363 | 0.3571x | 0.5887x | 1.0083x | yes |

The complete W3 MLP improves **1.00723x geometrically**, with five of five
cells winning.  Mean absolute error is at most `1.529e-8` and maximum absolute
error is `4.760e-8` against the same-payload dense-P32 Torch oracle.  The
complete runtime test is bit-exact and CUDA Graph replay exact.

## Rejected precursor probes

- Explicit bit-field high-byte extraction regressed four of five rows and is
  not retained.  Earlier SASS analysis also showed that Hopper selects the
  same byte permutation for this spelling.
- Hoisting a lane base around the eight shared level loads left the static
  instruction count unchanged and increased `IMAD` count; it is not retained.
- Gate/up split four regressed every row by 5.3--7.3 percent.
- Qwen-specific W3 decode depth three regressed every row by 2.1--2.6 percent;
  depth four remains production.

The distilled result is
`artifacts/a41_phase77_h100/qwen38_27b_w3_down_prefetch.json`.
