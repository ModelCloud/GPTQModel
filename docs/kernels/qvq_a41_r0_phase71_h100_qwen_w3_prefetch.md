# Phase 71: H100 Qwen3.8 W3 ordered decode prefetch

Phase 71 promotes the register-prefetched P32 level decode for the exact
Qwen3.8-27B grouped gate/up schedule at W3.  It improves all five target row
counts, remains exact, and is CUDA Graph safe.  W2 and W2.5 retain the Phase-70
kernel.

## Scheduling change

Qwen gate and up each execute as an independent ordered split-10 segment:

```text
gate: K=5120, N=17408, split=10
up:   K=5120, N=17408, split=10
```

Before this phase, the reusable WGMMA source fragment stayed live until the
dependency barrier, leaving the next eight shared level-table loads behind
that wait.  At W3 only, the decoder now builds an independent decoded fragment:

$$
D_k=L\!\left(\operatorname{PGC}(S_k,B_k)\right),\qquad k=0,\ldots,7,
$$

then waits for the old WGMMA operand and copies the already decoded packed
values into the reusable fragment.  P32 state extraction, selector math,
child-local split order, WGMMA issue order, FP32 accumulation, and all FP16
rounding boundaries are unchanged.

Promotion is restricted to the physical NVIDIA H100, grouped ordered partial
output, `K=5120`, two `N=17408` children, split ten for both children, and W3.
The W2.5 probe was not promoted because M8 and M16 regressed.  Runtime telemetry
reports `h100_qwen_w3_ordered_decode_prefetch_launches`.

## Correctness

- The complete Qwen MLP is exact across eager execution and CUDA Graph replay.
- Ordered grouped partials retain the established child-local reduction order.
- Mean absolute error is `1.503e-8` to `1.544e-8` against the same-payload
  dense-P32 Torch oracle; the maximum across the matrix is `4.838e-8`.
- No checkpoint, persistent VRAM, or transient workspace size changes.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays/sample.
`last/new` compares with committed Phase 70.  Marlin and Machete are figurative
W4 projection-sum baselines; ratios below one mean the W4 baseline is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 222.188 | 0.258x | 0.506x | 1.0007x | yes |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 225.826 | 0.259x | 0.491x | 1.0007x | yes |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 229.332 | 0.256x | 0.483x | 1.0002x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 240.098 | 0.241x | 0.462x | 1.0001x | yes |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 252.666 | 0.267x | 0.440x | 0.9998x | no |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 221.452 | 0.259x | 0.508x | 0.9982x | no |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 224.833 | 0.260x | 0.493x | 1.0005x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 228.389 | 0.257x | 0.485x | 1.0007x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 238.976 | 0.242x | 0.464x | 0.9968x | no |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 251.316 | 0.268x | 0.442x | 0.9971x | no |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 225.675 | 0.254x | 0.498x | 1.0154x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 229.190 | 0.255x | 0.484x | 1.0151x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 232.575 | 0.252x | 0.477x | 1.0156x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 243.827 | 0.237x | 0.455x | 1.0134x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 256.268 | 0.263x | 0.433x | 1.0125x | yes |

The targeted W3 geometric speedup is **1.0144x**, with five of five wins.
Across the complete W2--W3 matrix the geometric speedup is **1.0044x**; the
untouched-rate movements are cross-run noise.  Raw distilled result:
`artifacts/a41_phase71_h100/qwen38_27b_w3_ordered_prefetch.json`.

## Next target

The Qwen down path still performs an atomic split-K accumulation followed by
a graph-time dual FP16/FP32 composite output transform.  Returning ordered
down partials and fusing their deterministic reduction into a native
5120-wide composite recovery can remove that reducer/materialization boundary
and the graph-only rescue duplication.
