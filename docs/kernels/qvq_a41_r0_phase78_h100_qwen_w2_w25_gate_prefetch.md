# Phase 78: H100 Qwen W2/W2.5 gate/up decoded-level prefetch

Phase 78 extends the exact packed decoded-level prefetch schedule to the W2
and W2.5 Qwen3.8-27B grouped gate/up kernel.  Phase 77 already uses it for W3.
The production Qwen gate/up schedule is now prefetched for every target rate
W2--W3.

## Exact schedule

Both children have K=5120, N=17408, and ordered split five.  Each block owns
four K256 stages.  For each decoded fragment the kernel computes four P32
states, the pseudo-random-code products, and eight shared level loads into
four packed FP16 register pairs before the old WGMMA source must be reused.
The existing compile-time wait then protects the final fragment overwrite.

This changes register scheduling only.  State extraction, selectors, level
values, WGMMA issue order, child-local split planes, left-to-right FP32
reduction, and recovery boundaries are identical.  Dispatch remains gated to
the exact physical H100 Qwen `(5120, (17408,17408), (5,5))` plan.  The broader
telemetry counter is `h100_qwen_ordered_decode_prefetch_launches`; the existing
W3-specific counter remains available.

W2 and W2.5 retain their prior atomic down path.  The Phase-77 ordered
split-17 down prefetch remains W3-only.

## H100 benchmark

The physical device UUID was
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`.  Timings use 20 warmups, 60
samples, and 50 CUDA Graph replays per sample, measured with CUDA events.
Marlin and Machete are figurative W4 projection-sum baselines.  `last/new`
compares with the last executable affecting these rows, Phase 76.

| W | M | MKN: gate/up x2; down | new us | effective TFLOP/s | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2; 1x17408x5120 | 183.648 | 2.912 | 0.3125x | 0.6125x | 1.0110x | yes |
| 2 | 2 | 2x5120x17408 x2; 2x17408x5120 | 184.992 | 5.782 | 0.3164x | 0.5991x | 1.0104x | yes |
| 2 | 4 | 4x5120x17408 x2; 4x17408x5120 | 186.083 | 11.495 | 0.3151x | 0.5958x | 1.0078x | yes |
| 2 | 8 | 8x5120x17408 x2; 8x17408x5120 | 185.518 | 23.061 | 0.3113x | 0.5975x | 1.0099x | yes |
| 2 | 16 | 16x5120x17408 x2; 16x17408x5120 | 185.851 | 46.039 | 0.3624x | 0.5975x | 1.0048x | yes |
| 2.5 | 1 | 1x5120x17408 x2; 1x17408x5120 | 184.024 | 2.906 | 0.3119x | 0.6112x | 1.0028x | yes |
| 2.5 | 2 | 2x5120x17408 x2; 2x17408x5120 | 185.504 | 5.766 | 0.3156x | 0.5974x | 1.0027x | yes |
| 2.5 | 4 | 4x5120x17408 x2; 4x17408x5120 | 185.910 | 11.506 | 0.3154x | 0.5963x | 1.0033x | yes |
| 2.5 | 8 | 8x5120x17408 x2; 8x17408x5120 | 186.151 | 22.982 | 0.3103x | 0.5955x | 1.0049x | yes |
| 2.5 | 16 | 16x5120x17408 x2; 16x17408x5120 | 186.417 | 45.899 | 0.3613x | 0.5957x | 1.0060x | yes |

The ten affected cells improve **1.00636x geometrically**.  W2 improves
1.00878x and W2.5 improves 1.00394x; every cell wins.  Mean absolute error is
at most `1.533e-8` and maximum absolute error is `4.838e-8` against the
same-payload dense-P32 Torch oracle.

The Qwen full-MLP graph test now runs W2, W2.5, and W3 and requires exact
CUDA Graph replay plus the expected rate-generic and W3-only telemetry.
The distilled benchmark is
`artifacts/a41_phase78_h100/qwen38_27b_w2_w25_gate_prefetch.json`.
