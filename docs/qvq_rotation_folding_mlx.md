# QVQ offline rotation-folding search on Apple M4 Max

Date: 2026-08-31. Model: `ModelCloud/Llama3.2-1B-Instruct` (16 decoder
layers, hidden size 2048, MLP size 8192, 32 query heads, 8 KV heads, head
dimension 64). Quantizer: standard P32 PGC16-v1 with fixed binary bank
selectors. Hardware: 48 GB Apple M4 Max in high-power mode.

This report distinguishes exact dense graph rewrites from post-quantization
accuracy. A transform is called folded only when the unquantized model passes
the numerical dense-equivalence checks below. Passing dense parity does not
imply that the transformed basis quantizes equally well.

## Current conclusion

A0 remains the best-quality and production control. A25 is the only
transform-removal arm that passed the propagated W2 gate: it folds the
head-local V output basis into V and the inverse O input basis into O, reducing
14 to 12 online Hadamards per block. Its final KL is `0.03080` versus A0's
`0.02973`, its logits relative L2 is lower, and its layer error is 5.4% higher.

Actual runtime improvement is within repeatability. A25's projected M=1
QuantLinear sum is `1.64054` ms versus A0's `1.65950` ms (`1.012x`), while its
p95 is slightly worse. Therefore A25 is the mathematical/quality Pareto point,
but A0 remains the recommended production architecture until the V/O fold is
integrated end to end and demonstrates a material full-model tokens/s gain.

The zero-Hadamard A6 candidate is not acceptable. At W2 its seven-role mean
local output relative L2 is `0.30523`, versus `0.23710` for A0, and its summed
local KL is over 12 times A0. Removing the down transform alone also causes a
large down-projection regression (`0.1371` to `0.4198` local output relative
L2 when comparing A22 with A23).

| Arm | Description | Online H/block | Other online | Folded sides/block | W2 EBPW | Final KL | Top-1 | Top-5 | Top-10 | M1 ms | tok/s | Dense parity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A0 | two-sided RHT | 14 | 0 | 0 | `2.05442` | `0.029735` | `.71795` | `1.0` | `1.0` | `1.6595` | — | exact control | production |
| A1 | residual + V/O folds | 5 | 0 | 9 | `2.05442` | — | — | — | — | `1.3680` | — | rel `1.82e-6` | local reject |
| A3 | A1 + RoPE Q/K | 3 | 0 | 11 | `2.05442` | — | — | — | — | `1.3722` | — | rel `1.87e-6` | reject |
| A4 | A3 + SwiGLU fold | 1 | 0 | 13 | `2.05442` | — | — | — | — | `1.3566` | — | rel `1.82e-6` | reject |
| A6 | zero H | 0 | 0 | 13 | `2.05442` | — | — | — | — | `1.3507` | — | rel `1.82e-6` | reject |
| A22 | three-H aggressive | 3 | 0 | 11 | `2.05442` | `0.057908` | `.84615` | `1.0` | `1.0` | `1.3524` | — | rel `1.75e-6` | reject |
| A25 | V/O only | 12 | 0 | 2 | `2.05442` | `0.030801` | `.87179` | `1.0` | `1.0` | `1.6405` | — | rel `1.15e-6` | Pareto/pass |
| A27 | permutation SwiGLU | 12 | 0 | 2 | `2.05442` | `0.034170` | `.84615` | `1.0` | `1.0` | `1.6453` | — | rel `1.13e-6` | reject |
| A29 | identity gate/up output | 12 | 0 | 0 | `2.05442` | `0.034276` | `.94872` | `1.0` | `1.0` | `1.6453` | — | exact zero delta | reject |

`tok/s` is intentionally blank: the repository does not yet have an MLX Llama
container that persists these graph bases, so only real-shape QuantLinear sums
can be measured without making a false end-to-end claim.

## Exact folds

Weights use PyTorch's `[out,in]` storage and row activations. For an orthogonal
residual basis `R`, `x_tilde=xR`. Input-side projections are rewritten as
`W'=WR`; output-side residual projections use `W'=R^T W`. Therefore
`x_tilde W'^T=xW^T` on input-side maps and `zW'^T=(zW^T)R` on output-side
maps. Embeddings become `ER`, the LM head becomes `W_head R`, and each
RMSNorm scale is fused into its consumers before its scale is reset to one.
RMS is invariant under orthogonal `R`, so residual additions remain in one
common basis.

For each KV head, V uses a head-local orthogonal `R_v` and the matching O
columns use its inverse. Attention weights act on token positions, hence
`A(VR_v)=(AV)R_v`; the O rewrite cancels `R_v` without mixing heads. The same
KV-head basis is repeated across the four associated GQA query heads.

For a rotary pair, the tested A3 map is `T_k=sR(phi)` and
`T_q=s^-1R(phi)`. Two-dimensional rotations commute with that pair's RoPE
rotation and `T_q T_k^T=I`, so attention scores are unchanged. Llama 3.2 uses
split-half pairs `(i,i+head_dim/2)`, which the implementation handles
explicitly. The rewrite fails closed if q_norm or k_norm is present. A20
removes the reciprocal scale and A21 uses the identity pair map; both remain
exact but both still quantize worse than retaining the online Q/K Hadamards.

For SwiGLU, gate rows are permuted, up rows receive the same permutation and
a diagonal scale, and down columns receive the inverse permutation/scale:
`SiLU(gP) * (uPD) = (SiLU(g)*u)PD`. The rewritten down weight cancels `PD`.
Only powers-of-two scales are used in the current exact implementation.

## Dense equivalence on the complete 1B model

These are complete-model MPS logits on a held-out sentence, relative to A0.
All tested exact arms retain 100% top-1 identity.

| Arm | Max abs logits delta | Relative L2 | Top-1 identity |
| --- | ---: | ---: | ---: |
| A1 | `5.5313e-5` | `1.8182e-6` | `1.000` |
| A3 | `5.3406e-5` | `1.8661e-6` | `1.000` |
| A4 | `6.2943e-5` | `1.8164e-6` | `1.000` |
| A6 | `6.2943e-5` | `1.8164e-6` | `1.000` |
| A20 | `5.3406e-5` | `1.8661e-6` | `1.000` |
| A21 | `5.5313e-5` | `1.8182e-6` | `1.000` |
| A22 | `5.3406e-5` | `1.7464e-6` | `1.000` |
| A23 | `5.3406e-5` | `1.7464e-6` | `1.000` |
| A24 | `3.3379e-5` | `1.1311e-6` | `1.000` |
| A25 | `2.5511e-5` | `1.1544e-6` | `1.000` |
| A26 | `2.8610e-5` | `1.1354e-6` | `1.000` |
| A27 | `3.3379e-5` | `1.1311e-6` | `1.000` |
| A28 | `2.8610e-5` | `1.1354e-6` | `1.000` |
| A29 | `0` | `0` | `1.000` |
| A30 | `2.5511e-5` | `1.1544e-6` | `1.000` |

A0 is also covered by a state-dict byte-preservation test when the planner is
disabled. Tiny-Llama tests independently exercise final-logit parity for all
of the arms above and the q_norm/k_norm blocker.

## Stage-1 real-weight quantization screen

The screen uses aligned 256-by-256 crops from all seven real layer-0 matrices.
Hessians come from four calibration texts; three disjoint validation texts
provide local outputs. Arm selection never sees the dense-parity sentence.
The table reports the mean validation output relative L2 over all roles and
the sum of role-local KL values. It is a screening metric, not final-model KL.

| Arm | Online H/block | W1.5 output L2 / KL | W2 output L2 / KL | W2.5 output L2 / KL | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 | 14 | `0.33435 / 0.002833` | `0.23710 / 0.001367` | `0.16787 / 0.000685` | quality control |
| A1 | 5 | `0.35976 / 0.007473` | `0.25996 / 0.003955` | `0.18465 / 0.002127` | quality regression |
| A3 | 3 | `0.38073 / 0.018495` | `0.28831 / 0.013677` | `0.21868 / 0.010177` | reject: Q/K |
| A4 | 1 | `0.34373 / 0.019401` | `0.26483 / 0.014150` | `0.20481 / 0.010471` | reject: inherited Q/K |
| A6 | 0 | `0.40054 / 0.024241` | `0.30523 / 0.016989` | `0.23588 / 0.012098` | reject |
| A20 | 3 | `0.36762 / 0.008898` | `0.26878 / 0.005283` | `0.19486 / 0.003001` | reject: Q/K |
| A21 | 3 | `0.37147 / 0.009317` | `0.27377 / 0.005541` | `0.20165 / 0.003472` | reject: Q/K |
| A22 | 3 | `0.32251 / 0.008376` | `0.23560 / 0.004423` | `0.17078 / 0.002422` | promoted to layer test |
| A23 | 2 | `0.37931 / 0.013217` | `0.27599 / 0.007262` | `0.20184 / 0.004049` | reject: down |
| A25 | 12 | `0.34454 / 0.002835` | `0.24375 / 0.001370` | `0.17458 / 0.000687` | V/O-only candidate |
| A27 | 12 | `0.30554 / 0.002967` | `0.21754 / 0.001446` | `0.15524 / 0.000737` | promoted to layer test |
| A28 | 10 | `0.31581 / 0.002970` | `0.22423 / 0.001449` | `0.16195 / 0.000739` | dominated locally by A27 |
| A29 | 12 | `0.33441 / 0.002833` | `0.23737 / 0.001369` | `0.16866 / 0.000688` | promoted; within local noise |
| A30 | 10 | `0.34479 / 0.002837` | `0.24408 / 0.001372` | `0.17537 / 0.000690` | dominated by A29 |

All six required rates W1 through W3.5 were run for A0, A1, A3, A4, and A6.
The promoted A25 and A27 arms were also run at all six rates. Other evidence-driven AX
branches were screened at the priority rates W1.5, W2, and W2.5. The raw
per-role values, top-1/5/10 containment, weight error, fit
times, and actual tensor storage are in the JSON artifacts linked below.

## Promoted one-layer propagation

Every projection in real decoder layer 0 is quantized at its complete Llama
shape, then three held-out texts run through the remaining dense model. The
reference for each arm is its own exact dense rewrite. A22 is rejected because
its apparently favorable cropped reconstruction does not survive propagation.

| Arm | Online H/block | Final KL | Logits relative L2 | Layer relative L2 | Top-1 | Top-5 | Top-10 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | 14 | `0.029735` | `0.09691` | `0.04165` | `0.71795` | `1.0` | `1.0` | control |
| A22 | 3 | `0.057908` | `0.13963` | `0.12841` | `0.84615` | `1.0` | `1.0` | reject: KL/layer error |
| A27 | 12 | `0.034170` | `0.10533` | `0.13099` | `0.84615` | `1.0` | `1.0` | reject: layer error |
| A29 | 12 | `0.034276` | `0.11357` | `0.12848` | `0.94872` | `1.0` | `1.0` | reject: layer error |
| A25 | 12 | `0.030801` | `0.09337` | `0.04389` | `0.87179` | `1.0` | `1.0` | pass |

### W2 role diagnostics

Each cell is `validation output relative L2 (runtime axes)`. `HH` means input
and output full-H; `-H` output only; `H-` input only; `--` no full-H. Folded
residual, V/O, RoPE-pair, permutation, and diagonal maps are described by the
arm rather than counted as runtime axes.

| Arm | Q | K | V | O | Gate | Up | Down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | `.1320 HH` | `.1325 HH` | `.2706 HH` | `.2179 HH` | `.2953 HH` | `.3035 HH` | `.3078 HH` |
| A1 | `.2196 -H` | `.2053 -H` | `.2936 --` | `.2646 --` | `.2577 -H` | `.2652 -H` | `.3137 H-` |
| A3 | `.3709 --` | `.2587 --` | `.2936 --` | `.2646 --` | `.2577 -H` | `.2590 -H` | `.3137 H-` |
| A4 | `.3709 --` | `.2587 --` | `.2936 --` | `.2646 --` | `.2553 --` | `.2736 --` | `.1371 H-` |
| A6 | `.3709 --` | `.2587 --` | `.2936 --` | `.2646 --` | `.2553 --` | `.2736 --` | `.4198 --` |
| A22 | `.2196 -H` | `.2053 -H` | `.2936 --` | `.2646 --` | `.2553 --` | `.2736 --` | `.1371 H-` |
| A23 | `.2196 -H` | `.2053 -H` | `.2936 --` | `.2646 --` | `.2553 --` | `.2736 --` | `.4198 --` |
| A25 | `.1320 HH` | `.1325 HH` | `.2841 H-` | `.2514 -H` | `.2953 HH` | `.3030 HH` | `.3078 HH` |
| A27 | `.1320 HH` | `.1325 HH` | `.2706 HH` | `.2179 HH` | `.2995 H-` | `.3031 H-` | `.1672 HH` |
| A28 | `.1320 HH` | `.1325 HH` | `.2841 H-` | `.2514 -H` | `.2995 H-` | `.3031 H-` | `.1670 HH` |

## MLX runtime

The runtime benchmark uses all seven real Llama projection shapes and standard
P32 W2. Each value is the sum of 30 independently synchronized module medians
after 10 warmups; the parenthesized value is the corresponding sum of module
p95s. This is a projected QuantLinear block cost, not a full-model tokens/s
claim. Independent inner timings vary with run order, so selection uses the
complete QuantLinear boundary.

| Arm | Online H/block | M1 ms | M2 ms | M4 ms | M8 ms | M1 standalone H ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 14 | `1.6595 (2.0018)` | `1.7886 (2.1425)` | `1.9996 (2.3070)` | `2.8182 (3.2126)` | `1.6869` |
| A25 | 12 | `1.6405 (2.0312)` | `1.7678 (2.1272)` | `2.0009 (2.2963)` | `2.8323 (3.1407)` | `1.4677` |
| A1 | 5 | `1.3680` | `1.7364` | `1.9655` | `2.7670` | `0.5459` |
| A22 | 3 | `1.3524 (1.7856)` | `1.7225 (2.0591)` | `1.9515 (2.4317)` | `2.7860 (3.0595)` | `0.3282` |
| A6 | 0 | `1.3507 (1.7577)` | `1.7057 (1.9747)` | `1.9161 (2.2379)` | `2.8408 (3.2081)` | `0` |

A25 removes 13.0% of the separately timed Hadamard cost at M1, but the complete
QuantLinear gain is only 1.14% and its M1 p95 is 1.47% worse. M4 is flat and M8
slightly regresses. More aggressive arms are faster but failed propagated
quality. A full transformed-basis MLX Llama container is still required before
reporting layer time, time/token, or tokens/s; inventing those values from
independent modules would be misleading.

## Storage

Standard P32 carries the nominal transition payload, one selector bit per
32-weight segment, one alternative-bank byte per module, and FP32 SU/SV in
the current MLX runtime. Across the seven full Llama layer shapes, W2 measures
`2.05442` effective BPW. All implemented folding arms retain those module-local
SU/SV degrees of freedom and therefore have the same QVQ EBPW. The graph maps
tested here are baked into weights and add only a checkpoint schema/config
marker, not per-token tensors. Non-A0 serialization is deliberately disabled
until that explicit marker and the non-QuantLinear transformed tensors are
integrated into checkpoint save/load.

## A0-to-AX history and pruning

| Arm | Disposition |
| --- | --- |
| A0 | Run at all rates; best-quality control. |
| A1 | Run at all rates; exact and fast, but worse Q/K and aggregate local KL. |
| A2 | Planned, not mislabeled as implemented: learned folded-basis fitting is absent and fails closed. Pruned after the fixed folded family missed the quality gate. |
| A3 | Run at all rates; exact RoPE maps, rejected for Q/K post-quant error. |
| A4 | Run at all rates; exact one-H graph, rejected because it inherits A3 Q/K loss. |
| A5 | Pruned before structured-down fitting because its A3 Q/K parent is dominated. |
| A6 | Run at all rates; exact zero-H graph, rejected for Q/K and down error. |
| A7 | Rejected mathematically: changing the residual basis per layer requires an online conversion across the identity residual edge. |
| A8/A9 | Planner arms retained, but learned-basis rewrite fails closed; no fixed transform is reported as learned. |
| A10 | Subsumed by A1's globally shared residual basis; shorter sharing adds conversions without a new exact fold. |
| A11 | Rejected mathematically: role-specific residual bases disagree at residual addition unless converted online. |
| A12-A14 | Block-H down candidates remain design-only; their A4 parent is dominated and the generic runtime does not pretend a block-H is identity. |
| A15-A17 | Pairwise/permutation down candidates remain design-only for the same staged-pruning reason. |
| A18 | Algebraically equivalent to the tested A6 exact SwiGLU/zero-down candidate; rejected. |
| A19 | Joint bank-family search not promoted because no transform-removal parent beat the quality control. |
| A20 | Run at priority rates; rotation-only RoPE pair is better than A3 but still loses to online Q/K H. |
| A21 | Run at priority rates; identity Q/K output basis is exact but still loses to online Q/K H. |
| A22 | Generated from evidence: retain Q/K and down H, fold SwiGLU. Promoted. |
| A23 | Generated from A22: remove down H. Rejected by the down role. |
| A24 | SwiGLU permutation/scaling only. Rejected because diagonal scaling worsened down quantization. |
| A25 | V/O-only fold. Passes one-layer propagation and is the sole acceptable quality Pareto point; runtime gain is within noise. |
| A26 | A24 plus V/O. Rejected with A24's down regression. |
| A27 | Generated from A24 evidence: permutation-only SwiGLU fold. Rejected after propagated layer error rose 3.15x. |
| A28 | A27 plus V/O fold. Locally dominated by A27 on quality. |
| A29 | Generated after A27 propagation: identity gate/up output bases, no SwiGLU metadata. Within local repeatability and promoted. |
| A30 | A29 plus V/O. Locally dominated by A29. |

## Recommendation and production work

Keep A0 as the production default. A25 is the productionization experiment and
the fastest acceptable quality arm, but its 1.14% M1 median improvement is not
yet material relative to p95/run-to-run variation. A25's exact remaining
transforms are both sides of Q and K; V input only; O output only; and both
sides of gate, up, and down. V output and O input are the two fully folded
head-local maps. Q/K remain because the legal pair-local RoPE family cannot
reproduce the recovery of a full output Hadamard. Gate/up remain because every
identity, permutation, and permutation/scaling replacement increased
propagated layer error by about 3x. Down remains because removing its input H
caused a large post-quant regression. The global residual fold also remains
unpromoted because its A1 family missed the quality gate.

Productionization requires: a checkpoint format marker; saving the rewritten
embedding, norm, LM-head and dense graph state; planner descriptors passed to
each QuantLinear; persisted input/output-H flags; MLX Llama residual-basis
integration; and end-to-end tokens/s validation. Generic QVQ kernels receive
only transform descriptors/flags and contain no Q/K/V/O/gate/up/down name
checks.

Raw artifacts:

- `artifacts/qvq_rotation_stage1_m4max.json`
- `artifacts/qvq_rotation_ax_stage1_m4max.json`
- `artifacts/qvq_rotation_ax2_stage1_m4max.json`
- `artifacts/qvq_rotation_ax3_stage1_m4max.json`
- `artifacts/qvq_rotation_a27_full_stage1_m4max.json`
- `artifacts/qvq_rotation_a25_full_stage1_m4max.json`
- `artifacts/qvq_rotation_ax4_stage1_m4max.json`
- `artifacts/qvq_rotation_runtime_m4max.json`
- `artifacts/qvq_rotation_layer0_w2_m4max.json`
- `artifacts/qvq_rotation_layer0_a27_w2_m4max.json`
- `artifacts/qvq_rotation_layer0_a29_w2_m4max.json`
- `artifacts/qvq_rotation_layer0_a25_w2_m4max.json`
- `artifacts/qvq_p32_mlx_m4max_smoke.json`
- `artifacts/qvq_p32_mlx_m4max.json`

## Verification

The focused MLX, P32, LR-rejection, transform-planner, and folded-axis matrix
passes `409/409` tests. The broader QVQ/P32 run passes 871 tests with 137
skips; its 42 failures all invoke x86-64-only native CPU kernels on this arm64
M4 Max and fail at the existing `qvq_cpu_supported()` architecture guard.
No MLX, P32, planner, or folded-axis test fails. Ruff and `git diff --check`
also pass for the changed surface.
