# Phase 69: H100 Qwen3.8 folded MLP bridge

Phase 69 extends the accepted A41/R0 H100 path to Qwen3.8-27B's folded
17,408-wide MLP intermediate.  It is a production promotion: every W2--W3
decode cell improves, the operation remains CUDA Graph safe, and output is
bit-exact to the previous runtime boundary.

## Exact operation

Qwen3.8 intentionally disables the gate/up output Hadamards and the down
input Hadamard because 17,408 is `17 * 1024` and the implementation has no
supported exact 17-point base.  Before this phase the runtime materialized:

```text
gate_fp32 * SV_gate [+ bias_gate] -> FP16 gate
up_fp32   * SV_up   [+ bias_up]   -> FP16 up
SiLU(gate) -> FP16 activated gate
activated gate * up -> FP16 product
product * SU_down -> FP16 transformed input
zero-fill/copy -> M16 P32 input
```

The fused kernel computes the same elementwise chain directly:

$$
g_h = \operatorname{fp16}(\operatorname{rn}(gS_g)+b_g),
\qquad
u_h = \operatorname{fp16}(\operatorname{rn}(uS_u)+b_u),
$$

$$
a_h = \operatorname{fp16}\!\left(
\frac{\operatorname{fp32}(g_h)}
     {1+\exp(-\operatorname{fp32}(g_h))}
\right),
$$

$$
p_h = \operatorname{fp16}(\operatorname{fp32}(a_h)
                            \operatorname{fp32}(u_h)),
\qquad
x_h = \operatorname{fp16}(\operatorname{fp32}(p_h)
                            \operatorname{fp32}(SU_{down})).
$$

Explicit round-to-nearest multiply/add instructions prevent compiler FMA
contraction from changing the former separate-kernel FP32 boundary.  The
SiLU expression is the exhaustively validated FP16 implementation already
used by the H100 Llama path.  Logical rows are written into an M16 output;
the remaining rows are cleared before launch.

## Kernel and runtime design

- One 256-thread elementwise grid replaces the two recoveries, casts, SiLU,
  product, down scaling, and copy.
- The native operation accepts optional independent gate/up biases.
- Dispatch is restricted to physical H100 FP16 execution with the exact
  Qwen3.8 geometry `5120 -> 17408 x2 -> 5120`.
- Allocation, memset, and launch are stream ordered and CUDA Graph safe.
- `h100_folded_qwen_fused_precondition_launches` proves production dispatch.
- Unsupported devices, dtypes, and geometries retain the previous exact path.

## Correctness

The low-level unit test covers M=1/2/4/8/16, with and without biases.  It
requires FP16 bit equality for every logical value, exact zero padded rows,
and bit-identical CUDA Graph replay.  The complete grouped Qwen MLP lifecycle
test also requires graph replay equality and the new telemetry counter.

The 15-cell benchmark's mean absolute error against the same-payload dense-P32
Torch oracle is `1.503e-8`--`1.544e-8`; global maximum absolute error is
`4.838e-8`.

## H100 benchmark

Timing uses CUDA Graph replay and CUDA events on physical H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`: 20 warmups, 60 samples, and 50
replays per sample.  `last/new` compares with merged PR #98.  Baselines are
the existing W4 Marlin and Machete projection sums; ratios below one mean the
baseline remains faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 235.502 | 0.244x | 0.478x | 1.0382x | yes |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 238.745 | 0.245x | 0.464x | 1.0496x | yes |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 241.761 | 0.243x | 0.459x | 1.0506x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 250.537 | 0.231x | 0.442x | 1.0497x | yes |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 259.151 | 0.260x | 0.429x | 1.0419x | yes |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 235.438 | 0.244x | 0.478x | 1.0386x | yes |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 238.545 | 0.245x | 0.465x | 1.0490x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 240.959 | 0.243x | 0.460x | 1.0542x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 249.288 | 0.232x | 0.445x | 1.0551x | yes |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 258.020 | 0.261x | 0.430x | 1.0480x | yes |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 242.417 | 0.237x | 0.464x | 1.0415x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 245.417 | 0.239x | 0.452x | 1.0543x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 248.248 | 0.236x | 0.447x | 1.0561x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 257.489 | 0.224x | 0.431x | 1.0536x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 265.740 | 0.253x | 0.418x | 1.0477x | yes |

Geometric speedup over merged PR #98 is **1.0485x**, with **15/15 wins**.
The new geometric baseline ratios are `0.242x` Marlin and `0.450x` Machete.
This phase removes avoidable lifecycle traffic; the remaining MLP gap is
dominated by the grouped gate/up and down P32 kernels.

Raw result: `artifacts/a41_phase69_h100/qwen38_27b_folded_mlp.json`.
