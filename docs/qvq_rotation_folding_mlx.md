# QVQ offline rotation-folding search on Apple M4 Max and Ampere

Date: 2026-08-31. Model: `ModelCloud/Llama3.2-1B-Instruct` (16 decoder
layers, hidden size 2048, MLP size 8192, 32 query heads, 8 KV heads, head
dimension 64). Quantizer: standard P32 PGC16-v1 with fixed binary bank
selectors. Runtime measurements use a 48 GB Apple M4 Max in high-power mode.
The complete 16-layer W2 quality promotion uses an SM80 CUDA host reported as
`NVIDIA PG506-230` with 96 GB of memory. That host also runs the complete
packed-model CUDA quality and decode benchmark through production
`QVQLinear -> qvq_cuda_gemv` V2B2-P32 planar dispatch.

This report distinguishes exact dense graph rewrites from post-quantization
accuracy. A transform is called folded only when the unquantized model passes
the numerical dense-equivalence checks below. Passing dense parity does not
imply that the transformed basis quantizes equally well.

## Current conclusion

A0 remains the production control. A25 is the only transform-removal arm that
passed the propagated W2 gate: it folds the head-local V output basis into V
and the inverse O input basis into O, reducing 14 to 12 online Hadamards per
block. The first 16-row validation stream favors A25 by 1.49% KL, but two
additional disjoint streams show that this is not a stable win. Across all
5,630 held-out tokens, packed CUDA KL is `0.91659` for A25 versus `0.91758`
for A0 (0.108% lower), while logits relative L2 is 0.357% higher and Top-1,
Top-5, and Top-10 are lower by 0.80, 0.62, and 0.85 percentage points. The
paired 95% bootstrap intervals for KL and Top-1 both cross zero.

The end-to-end packed CUDA model shows a real median decode signal: A25 is
1.90% faster at batch 1 and about 4.1--4.4% faster at batches 2--8. Batch-1
throughput moves from 26.57 to 27.08 token/s. Prefill is flat, and batch-4/8
p95 decode regresses because of late-cycle outliers. Therefore A25 remains the
mathematical/runtime Pareto candidate, but A0 stays the recommended production
architecture until quality is confirmed across fitting seeds/evaluation sets
and the long-tail latency behavior is resolved.

The zero-Hadamard A6 candidate is not acceptable. At W2 its seven-role mean
local output relative L2 is `0.30523`, versus `0.23710` for A0, and its summed
local KL is over 12 times A0. Removing the down transform alone also causes a
large down-projection regression (`0.1371` to `0.4198` local output relative
L2 when comparing A22 with A23).

The cross-arm table retains the common one-layer propagation metric because
only A0 and A25 were promoted to the complete 16-layer run.

| Arm | Description | Online H/block | Other online | Folded sides/block | W2 EBPW | Layer-0 KL | Top-1 | Top-5 | Top-10 | MLX M1 ms | CUDA M1 tok/s | Dense parity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A0 | two-sided RHT | 14 | 0 | 0 | `2.05442` | `0.029735` | `.71795` | `1.0` | `1.0` | `1.6595` | `26.57` | exact control | production |
| A1 | residual + V/O folds | 5 | 0 | 9 | `2.05442` | — | — | — | — | `1.3680` | — | rel `1.82e-6` | local reject |
| A3 | A1 + RoPE Q/K | 3 | 0 | 11 | `2.05442` | — | — | — | — | `1.3722` | — | rel `1.87e-6` | reject |
| A4 | A3 + SwiGLU fold | 1 | 0 | 13 | `2.05442` | — | — | — | — | `1.3566` | — | rel `1.82e-6` | reject |
| A6 | zero H | 0 | 0 | 13 | `2.05442` | — | — | — | — | `1.3507` | — | rel `1.82e-6` | reject |
| A22 | three-H aggressive | 3 | 0 | 11 | `2.05442` | `0.057908` | `.84615` | `1.0` | `1.0` | `1.3524` | — | rel `1.75e-6` | reject |
| A25 | V/O only | 12 | 0 | 2 | `2.05442` | `0.030801` | `.87179` | `1.0` | `1.0` | `1.6405` | `27.08` | rel `1.15e-6` | Pareto/pass |
| A27 | permutation SwiGLU | 12 | 0 | 2 | `2.05442` | `0.034170` | `.84615` | `1.0` | `1.0` | `1.6453` | — | rel `1.13e-6` | reject |
| A29 | identity gate/up output | 12 | 0 | 0 | `2.05442` | `0.034276` | `.94872` | `1.0` | `1.0` | `1.6453` | — | exact zero delta | reject |

CUDA throughput is measured from complete packed models. MLX still lacks a
Llama container that persists these graph bases, so its entries remain
real-shape QuantLinear sums rather than an end-to-end claim.

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

### Full-model W2 promotion run

A progressive full-model W2 A0-versus-A25 runner was added after review of the
small one-layer validation population. It uses 16 cached WikiText-2 train rows
(1,678 tokens) for calibration and 16 disjoint validation rows (1,869 tokens),
with a 128-token cap. The test split remains unread. Within every decoder layer
it recaptures dependencies in `Q/K/V -> O -> gate/up -> down` order, and every
later layer sees all previously quantized layers.

The original M4 Max run was stopped at the user's request because full P32
quantization was too slow on that host. A0 completed five of 16 layers (35
projections, 2,523.2 seconds of fitting); A25 was not started. These partial
values are trajectory diagnostics only and must not be treated as an A0/A25
comparison or a full-model result:

| Quantized through layer | Final KL | Logits relative L2 | Top-1 | Top-5 | Top-10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.088612` | `0.17908` | `.87212` | `.99037` | `.99304` |
| 1 | `0.150978` | `0.23146` | `.83093` | `.97967` | `.98876` |
| 2 | `0.201181` | `0.26658` | `.79882` | `.97164` | `.98823` |
| 3 | `0.266535` | `0.29328` | `.75281` | `.96148` | `.98020` |
| 4 | `0.330728` | `0.31756` | `.72980` | `.95131` | `.97699` |

The interrupted artifact is explicitly marked `interrupted_by_user`; its A0
payload is marked `interrupted_after_layer_4`.

The matched experiment was completed from layer 0 on the SM80 CUDA host. Both
arms quantize all 112 target projections and use identical calibration and
validation samples. `Fit seconds` is quantizer fitting time, not inference
latency; evaluation installs reconstructed quantized weights into the dense
model and does not exercise packed P32 inference kernels.

| Arm | Online H/block | Projections | EBPW | Final KL | Logits rel-L2 | Top-1 | Top-5 | Top-10 | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 14 | 112 | `2.054419` | `0.900642` | `0.491661` | `.55217` | `.81648` | `.88604` | `405.7` |
| A25 | 12 | 112 | `2.054419` | `0.887262` | `0.489304` | `.53986` | `.82076` | `.88604` | `404.2` |

A25 versus A0 is `-1.49%` final KL, `-0.48%` logits relative L2, `-1.23`
Top-1 percentage points, `+0.43` Top-5 points, and no Top-10 change. Dense
parity before quantization passes at `1.12e-6` logits relative L2,
`9.25e-5` max absolute logits delta, and 100% Top-1/5/10 identity.

The full progressive trajectory shows that the result is depth-dependent: A25
is worse through most of layers 0--6, becomes competitive at layer 7, and has
lower KL from layer 8 through the final layer.

| Through layer | A0 KL | A25 KL | A25 KL delta | A0 Top-1 | A25 Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `.085411` | `.090876` | `+6.40%` | `.87854` | `.88497` |
| 1 | `.156804` | `.154114` | `-1.72%` | `.82076` | `.82718` |
| 2 | `.205767` | `.212951` | `+3.49%` | `.78545` | `.78545` |
| 3 | `.263433` | `.281484` | `+6.85%` | `.76833` | `.74532` |
| 4 | `.302290` | `.342736` | `+13.38%` | `.74104` | `.71696` |
| 5 | `.353558` | `.388892` | `+9.99%` | `.71910` | `.69342` |
| 6 | `.403128` | `.434637` | `+7.82%` | `.68860` | `.67309` |
| 7 | `.470211` | `.475894` | `+1.21%` | `.66132` | `.65971` |
| 8 | `.532519` | `.520953` | `-2.17%` | `.63296` | `.64098` |
| 9 | `.598485` | `.590595` | `-1.32%` | `.62012` | `.61958` |
| 10 | `.663142` | `.649837` | `-2.01%` | `.60246` | `.60139` |
| 11 | `.727606` | `.707891` | `-2.71%` | `.58320` | `.58694` |
| 12 | `.775093` | `.754261` | `-2.69%` | `.56394` | `.57999` |
| 13 | `.822580` | `.797732` | `-3.02%` | `.56340` | `.56501` |
| 14 | `.861346` | `.837031` | `-2.82%` | `.55003` | `.55859` |
| 15 | `.900642` | `.887262` | `-1.49%` | `.55217` | `.53986` |

### Packed CUDA multi-stream promotion

The follow-up keeps each arm's canonical packed tensors in memory and replaces
all 112 dense projections with production `QVQLinear` modules. It evaluates
three disjoint 16-row WikiText-2 validation streams (5,630 tokens total) in
FP16 through the native CUDA P32 path. Stream 0 is intentionally identical to
the preceding experiment and exactly reproduces its reconstructed-weight
metrics; streams 1 and 2 test whether that result generalizes. The same dense
FP32 logits are the oracle for both arms, and the test split remains unread.

| Arm | Reconstructed KL | Packed KL | Packed logits rel-L2 | Packed Top-1 | Top-5 | Top-10 | Packed minus reconstructed KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | `.917179` | `.917579` | `.493024` | `.55684` | `.82433` | `.88934` | `+.000400` |
| A25 | `.916400` | `.916587` | `.494783` | `.54885` | `.81812` | `.88082` | `+.000187` |

The small packed-versus-reconstructed deltas validate the in-memory packed
installation and quantify the expected FP16/native-kernel rounding. At actual
packed execution, A25 versus A0 is `-0.108%` KL, `+0.357%` logits relative L2,
`-0.80` Top-1 points, `-0.62` Top-5 points, and `-0.85` Top-10 points.

| Stream | Tokens | A0 packed KL | A25 packed KL | A25 KL delta | A0 Top-1 | A25 Top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,869 | `.900927` | `.887484` | `-1.49%` | `.55110` | `.54040` |
| 1 | 1,898 | `.957266` | `.983824` | `+2.77%` | `.55954` | `.54953` |
| 2 | 1,863 | `.893854` | `.877286` | `-1.85%` | `.55985` | `.55663` |

A 2,000-resample paired bootstrap over held-out texts gives an A25-minus-A0
KL median of `-0.001236` with 95% interval `[-0.028447, +0.026979]`. The
Top-1 median delta is `-0.007951` with interval
`[-0.021095, +0.004783]`. Neither interval excludes zero, so the expanded
quality evidence supports parity/uncertainty rather than an A25 accuracy win.

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

## Packed CUDA full-model runtime

The CUDA measurement uses the two complete in-memory packed models from the
multi-stream promotion, not reconstructed dense weights. Every target
projection dispatches through production `QVQLinear -> qvq_cuda_gemv`
V2B2-P32 planar code on the exclusive SM80 GPU. This is distinct from the
separate direct P32 Ampere window microbenchmark and does not attribute that
kernel's results to `QVQLinear`.

For each arm and batch size, a 128-token prompt is prefetched and then decoded
one cached token per row at a time. Three timing cycles alternate arm order;
each cycle uses five decode warmups and 30 synchronized measurements. The
table reports pooled wall-clock medians and p95s over 90 decode samples.
CUDA-event medians agree within 0.04 ms.

| Batch | A0 median ms | A25 median ms | A25 median delta | A0 token/s | A25 token/s | A0 p95 ms | A25 p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `37.640` | `36.927` | `-1.90%` | `26.57` | `27.08` | `38.816` | `37.342` |
| 2 | `38.366` | `36.755` | `-4.20%` | `52.13` | `54.41` | `39.700` | `39.904` |
| 4 | `39.398` | `37.671` | `-4.38%` | `101.53` | `106.18` | `40.971` | `45.212` |
| 8 | `38.429` | `36.839` | `-4.14%` | `208.18` | `217.16` | `40.492` | `46.945` |

Prefill medians are effectively unchanged: A0/A25 are 164.852/164.748 ms at
batch 1 and 1,194.410/1,193.751 ms at batch 8. The decode median signal is
consistent with removing 32 full Hadamards across the 16-layer model. It is
not yet a clean tail-latency win: A25 p95 improves at batch 1, is flat at
batch 2, and regresses at batches 4 and 8 because the final A25 timing cycle
contains outliers. Raw event and wall samples for every cycle are retained in
the artifact.

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
| A0 | Run at all rates; production quality control. |
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
| A25 | V/O-only fold. Passes one-layer and complete 16-layer W2 propagation. Packed multi-stream KL is statistically tied; containment metrics trend worse. Full-model CUDA median decode improves 1.9--4.4%, with mixed p95. |
| A26 | A24 plus V/O. Rejected with A24's down regression. |
| A27 | Generated from A24 evidence: permutation-only SwiGLU fold. Rejected after propagated layer error rose 3.15x. |
| A28 | A27 plus V/O fold. Locally dominated by A27 on quality. |
| A29 | Generated after A27 propagation: identity gate/up output bases, no SwiGLU metadata. Within local repeatability and promoted. |
| A30 | A29 plus V/O. Locally dominated by A29. |

## Recommendation and production work

Keep A0 as the production default. The complete 16-layer W2 run strengthens
A25 as the productionization experiment, and the packed full-model CUDA run
proves that the planned axes execute end to end without a reconstructed dense
weight cache. The three-stream aggregate does not establish an accuracy win:
KL is statistically tied, logits relative L2 is slightly worse, and all three
containment metrics trend lower. Median CUDA decode improves 1.90% at batch 1
and 4.1--4.4% at batches 2--8, but larger-batch p95 is unstable. A25 is still
not a default change until independent fit seeds/evaluation sets confirm
quality and repeated idle-host timings resolve the tail behavior. A25's exact
remaining transforms are both sides of Q and K; V input only; O output only;
and both sides of gate, up, and down. V output and O input are the two fully
folded head-local maps. Q/K remain because the legal pair-local RoPE family
cannot reproduce the recovery of a full output Hadamard. Gate/up remain
because every identity, permutation, and permutation/scaling replacement
increased propagated layer error by about 3x. Down remains because removing
its input H caused a large post-quant regression. The global residual fold
also remains unpromoted because its A1 family missed the quality gate.

Productionization requires: a checkpoint format marker; saving the rewritten
embedding, norm, LM-head and dense graph state; planner descriptors passed to
each QuantLinear; persisted input/output-H flags; and repeated full-model
quality/runtime validation. CUDA execution is validated in memory here; MLX
still requires Llama graph integration before it can make an equivalent
end-to-end claim. Generic QVQ kernels receive only transform descriptors/flags
and contain no Q/K/V/O/gate/up/down name checks.

Raw artifacts:

The packed CUDA experiment is reproduced by
`scripts/benchmark_qvq_rotation_full_model_cuda.py`; it fails closed unless
CUDA SM80 is available and no foreign compute process owns the selected GPU.

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
- `artifacts/qvq_rotation_full16_a0_a25_w2_m4max.json` (interrupted after A0 layer 4)
- `artifacts/qvq_rotation_full16_a0_a25_w2_a100_sm80.json` (complete A0/A25)
- `artifacts/qvq_rotation_full16_a0_a25_w2_packed_cuda_sm80.json` (three-stream packed CUDA quality/runtime)
- `artifacts/qvq_p32_mlx_m4max_smoke.json`
- `artifacts/qvq_p32_mlx_m4max.json`

## Verification

The focused MLX, P32, LR-rejection, transform-planner, and folded-axis matrix
passes `409/409` tests. The broader 1,459-case QVQ/P32 matrix now passes 1,280
tests with 179 skips and zero failures on this arm64 M4 Max. The 42 native CPU
cases previously reported as failures are explicitly skipped because the
extension's `qvq_cpu_supported()` contract is x86-64-only. Ruff and
`git diff --check` also pass for the changed surface. On the CUDA host, the P32
fitting smoke test passes; the selected generic QVQ, P32/LR, planner,
folded-axis, and exact-P32 Ampere matrix passed 867 tests with 131 platform
skips before the latest-main merge. The expanded post-merge CUDA/P32/Ampere/
planner/folded-axis matrix passes 1,423 tests with 13 skips and zero failures.
Its one initially exposed test-contract failure was corrected by explicitly
requesting the Top-N metrics that the test compares, then the exact full
selection was rerun. No GitHub Actions result is claimed; hardware results are
local to the M4 Max and SM80 hosts described above.
