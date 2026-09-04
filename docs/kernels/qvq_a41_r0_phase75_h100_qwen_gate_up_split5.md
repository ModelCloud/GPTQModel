# Phase 75: H100 Qwen gate/up split five

Phase 75 changes the grouped Qwen3.8-27B gate/up P32 schedule from split ten
to split five and extends the exact fused gate/up reduction and W3 decoded
level prefetch to the new schedule.  It improves all 15 W2--W3 by M1--M16
complete-MLP cells and lowers geometric latency by 5.73 percent.

## Schedule math

Each Qwen gate/up child has

$$
K=5120,\qquad N=17408,\qquad K_{tiles}=5120/16=320.
$$

The P32 Hopper kernel consumes K256 stages, or 16 K16 tiles per stage.  The
previous split-ten schedule assigned

$$
320/10=32\ K16\ tiles=2\ K256\ stages
$$

to each CTA.  Across both N=17408 children it launched

$$
2\times(17408/64)\times10=5440\ CTAs.
$$

Split five instead assigns

$$
320/5=64\ K16\ tiles=4\ K256\ stages
$$

and launches 2720 CTAs.  That is still more than twenty blocks per H100 SM,
while halving repeated CTA setup, TMA-pipeline setup, P32 state initialization,
and ordered-partial traffic.

## Exact fused reduction

For each child and output element the kernel retains the canonical ordered
FP32 reduction

$$
s=((((p_0+p_1)+p_2)+p_3)+p_4).
$$

Gate/up recovery then preserves the established boundaries exactly:

$$
g_h=\operatorname{FP16}(s_gS_{V,g}+b_g),
\qquad
u_h=\operatorname{FP16}(s_uS_{V,u}+b_u),
$$

$$
d=\operatorname{FP16}
\left(
  \operatorname{FP16}(\operatorname{SiLU}_{FP16}(g_h)u_h)S_{U,down}
\right).
$$

The fused consumer now has compile-time specializations for split five and
split ten.  Unit tests require bit-exact equality to the separately reduced
reference at both split counts, M1/M16, bias/no-bias, and after CUDA Graph
replay.

W3's register-prefetched decoded-level path is also enabled for split five.
It is the same exact decoder used by split ten; only the validated split gate
changes.  Runtime telemetry continues to expose both the selected split
counts and W3 prefetch launches.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays/sample.
`last/new` compares with Phase 74.  Marlin and Machete are figurative W4
projection-sum baselines; ratios below one mean the W4 baseline is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 192.033 | 0.299x | 0.586x | 1.0508x | yes |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 193.465 | 0.303x | 0.573x | 1.0525x | yes |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 194.241 | 0.302x | 0.571x | 1.0540x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 195.019 | 0.296x | 0.568x | 1.0600x | yes |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 195.046 | 0.345x | 0.569x | 1.0762x | yes |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 191.660 | 0.299x | 0.587x | 1.0483x | yes |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 193.152 | 0.303x | 0.574x | 1.0506x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 193.617 | 0.303x | 0.573x | 1.0540x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 194.408 | 0.297x | 0.570x | 1.0611x | yes |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 195.439 | 0.345x | 0.568x | 1.0724x | yes |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 192.605 | 0.298x | 0.584x | 1.0577x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 194.197 | 0.301x | 0.571x | 1.0607x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 194.606 | 0.301x | 0.570x | 1.0653x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 195.498 | 0.295x | 0.567x | 1.0690x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 196.293 | 0.343x | 0.566x | 1.0807x | yes |

The geometric speedup is **1.0608x over Phase 74**, with 15 of 15 wins.
Mean absolute error remains at most `1.544e-8` and maximum absolute error is
`4.838e-8` against the same-payload dense-P32 Torch oracle.  The raw distilled
result is `artifacts/a41_phase75_h100/qwen38_27b_gate_up_split5.json`.

## Next target

The grouped gate/up P32 launch remains the largest component.  The next
profile should measure whether adjacent N64 outputs can share each split-five
input stage in an ordered N128 consumer without increasing shared-memory bank
conflicts or register stalls.
