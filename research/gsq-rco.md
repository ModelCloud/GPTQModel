# GSQ and RCO: QVQ research entry

## Primary sources

- User-provided [Qwen3.8-27B GSQ-RCO model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF).
- [GSQ paper v2, abstract](https://arxiv.org/abs/2604.18556v2),
  [author implementation](https://github.com/IST-DASLab/GSQ).
- [RCO paper v2, abstract](https://arxiv.org/abs/2605.00649v2),
  [author implementation](https://github.com/IST-DASLab/RCO).

Reviewed 2026-09-09. GSQ learns scalar assignments and group scales using a
Gumbel-Softmax relaxation. RCO optimizes assignments under a cost budget using
manifold-aware steps and a discrete feasibility stage. These are distinct
algorithms; the linked model combines them. Its published results are not QVQ
measurements. No author code was copied into the experiment below.

## QVQ adaptation and implementation status

The scalar fitter now has an optional asymmetric cross-moment objective for
the upcoming GPTAQ adapter. Let feature-by-token current inputs be X, native
inputs be Xn, E = Wq-W and D = (Xn-X)X^T. The candidate-dependent part of
`||Wq X - W (X + alpha*(Xn-X))||²` is
`tr(E H E^T) - 2 alpha <E, W D>`, with H = XX^T. H and D must share the same
sample normalization. Keeping the linear term avoids a Hessian inverse and
works for rank-deficient calibration. Scores omit the constant native residual
and can be negative; they must not be labeled absolute reconstruction NMSE.
Independent double-precision paired-activation tests check loss differences and
gradients, and a hard-grid fixture checks export selection against the explicit
native target. These are algebra/correctness tests, not real-model evidence.
GPTAQ now preserves its original-column H and D before its ordinary quantizer
consumes them, then supplies both to scalar GSQ after quantization. Public config
round-trip and activation-order on/off checks compare the resulting objective
against explicit native/current activations. Real F6/seed7 block-1 QKV validation
now passes packed reload, Torch GPU gates and canonical model propagation with
actual paired upstream inputs; both GSQ variants retain baseline exactly. See
the scalar lifecycle report for scope and raw evidence. Complete-model exports
remain pending. FOEM now also preserves its beta-dependent latent-weight updates
as the initializer, followed by GSQ reconstruction fitting (with the cross term
when alpha is nonzero). Beta is not invented as a separate final fitting target.
The real alpha0/beta0.2 block-1 run passes the measured reload/Torch GPU/F6
propagation checks and retains baseline exactly; other coefficients and
complete-model exports remain pending. FOEM-only config serialization now
preserves its coefficients, fixing their previous omission when GPTAQ was absent.

The broader [scalar lifecycle work and compatibility inventory](../docs/experiments/gsq-scalar-lifecycle.md)
tracks GPTQ, AWQ, RTN and remaining method adapters separately from QVQ results.
Real F6/seed7 full-QKV W4 checks now retain the GPTQ baseline for both scalar
GSQ variants. Learned-scale RTN reduces its weight objective, with mixed
downstream effects: KLD falls 3.243%, logit MSE rises 10.166%, and top-1
agreement falls 0.5968 percentage points. The linked protocol includes packed
reload, Torch GPU gates, document intervals and raw reports. This illustrates
why fitting loss, probability divergence and ranked-token agreement must be
reported separately; it does not justify enabling GSQ by default.

Current public integration: [optional GSQ lifecycle](../docs/experiments/gsq-qvq-lifecycle.md),
covering P32 W1–W3.5 and ordinary non-banked V2/L16 W4–W8, including half-bit
rates. The default is disabled. The initial P32 research and historical evidence
below explain the design and remain distinct from the Fisher-objective lifecycle.

Base: freshly fetched ModelCloud/QvQ `origin/main`
`a292a880432afce133e58219549f8d7483a3bf41`.

P32 stores a circular transition history. Its 16-bit states overlap, so choosing
each decoded weight or each state independently does not necessarily produce a
representable payload. See [the P32 contract](p32.md).

`gptqmodel/quantization/qvq_gsq.py` implements a **GSQ-inspired bounded candidate
relaxation**, not scalar GSQ or a complete reproduction of its training method.
Each tile has categorical logits over complete valid circular histories. A
Gumbel sample and geometric temperature schedule produce a differentiable mixture
of their decoded weights. Adam minimizes normalized calibration output MSE.
Hard argmax choices are checked after every step; the best hard calibration
checkpoint, including the unchanged baseline, is retained. Held-out inputs never
participate in optimization or checkpoint selection.

The candidate tensor is `[candidate, tile, words]`; each hard output picks one
complete tile. Banks, codebook and SU/SV remain fixed. There are no new serialized
scales, logits, state trailers or inference kernels. Existing window decoding is
the export authority. The initial helper was research-only; the optional
[QVQConfig lifecycle](../docs/experiments/gsq-qvq-lifecycle.md) now fits the prepared
YAQA Fisher metric before packing.
It materializes decoded candidate tiles, so it is suitable for bounded experiments,
not yet memory-efficient full-model quantization. Joint scale learning remains TODO.

### Math audit and disabled-by-default control

Audited against [GSQ Algorithm 1](https://arxiv.org/html/2604.18556v2#S3)
and the author's pinned revision
[`03fc16484c369e3127225615d5e03e8d3a6043e3`](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/quantization/gumbel_quantizer_2bit.py#L45).
For tile logits `l`, independent uniforms `u`, and temperature `tau`, this
experiment computes `g = -log(-log(u))`, `p = softmax((l + g) / tau)`, and
`W_soft[t] = sum_c p[t,c] * decode(candidate[c,t])`. The noise sign and
temperature placement are correct. The analytic probability Jacobian is
`(diag(p) - p p^T) / tau`; independent double-precision finite differences and
the scalar formula are checked at temperatures 0.1, 1 and 2.

This fixes the paper's logit multiplier kappa at 1. The author implementation
uses trainable scalar group scales and Lion with scheduled temperature/logit
scale; this experiment uses fixed P32 scales/banks, Adam, and geometric
temperature decay. Uniform clamping to `[1e-6, 1-1e-6]` truncates the extreme
noise tails; the author's endpoint stabilization differs. These are explicit
experimental choices, not a faithful scalar-GSQ reproduction.

The fitting loss is `||X(W_soft - W_teacher)||_F^2 / ||X W_teacher||_F^2`
(with a tiny positive denominator floor). This is activation reconstruction,
not the full two-sided YAQA/Fisher objective. With the checked constant-absolute
SV contract, its inner-basis value equals deployed-output NMSE by orthogonality.
Lowering it does not guarantee lower final-model KLD. A soft mixture is generally
not representable: export uses noise-free logit argmax and selects the best
hard calibration checkpoint, including baseline. Complete tile selection keeps
the circular history legal. No held-out rows enter that selection.

`refine_p32_candidates(..., enabled=False)` now defaults to disabled. It returns
an independent byte-identical copy of candidate zero, zero choices, `None`
losses and empty history without decoding, reading calibration, creating an
optimizer or sampling. Use `enabled=True` to fit. Both validation scripts require
`--gsq` to enable fitting; the full-layer runner binds that flag into preparation
provenance and rejects a mismatch at execution. Without it, the `gsq` report arm
is an explicitly disabled baseline copy; the independent deterministic control
still runs. Historical reports predate the flag and ran fitting enabled.

This describes the initial **research** control. The subsequent
[QVQConfig integration](../docs/experiments/gsq-qvq-lifecycle.md) adds a separate
optional lifecycle path with a two-sided Fisher objective.
YAQA remains the initializer. Its real W2.5 run retains every baseline tile and demonstrates no quality gain. The enabled probability refactor is checked for exact
FP32 equivalence to the previously measured expression.

Audit validation: 14 CPU tests pass. The enabled real Llama slice reproduces
its archived payload and both calibration losses exactly. Disabled calls on
the complete real W2.5 Q/K/V exports preserve all bytes and RNG state; see
[audit report](../artifacts/gsq-p32/math-audit-default-off/report.json).
The extracted probability calculation and new control branches are fully
exercised. Coverage for the entire experimental helper is 80% combined
line/branch coverage; pre-existing validation/error and progress branches remain
uncovered. This audit did not rerun full-model inference or GPU kernels; the
earlier model-quality reports remain the propagated evidence.

RCO is research-only and unimplemented. A QVQ implementation must count actual
serialized costs, respect backend-supported rates, and enforce feasibility after
hard assignment; an expected soft budget is insufficient. Keep this independent
of GSQ until each arm has a measured control.

### Lifecycle insertion (initial design; now implemented optionally)

The intended lifecycle is: prepare the existing calibration/Hessian inputs;
apply F6's RHT and rate-specific YAQA initialization; finalize the selected
baseline scales and banks; optionally refine valid P32 assignments using the
calibration objective; harden and validate the proposal; pack and save through
the existing format; verify reload and independent propagated quality.

In source terms, `QVQProcessor` calls `quantize_qvq_linear`. The latter selects
`quantized_inner`, `states`, `selected_bank_ids`, `selected_bank_alt_id`, and
SU/SV before its `pack_trellis` phase. A future integration must update this
entire consistent state together before final packing, after any stage that
would overwrite its selected scales or assignments. A window-only experimental
proposal must be converted back through the existing planar/state APIs; replacing
only a dense reconstructed weight would not change the saved checkpoint.

The experiment currently runs after loading an already quantized snapshot.
It tests the refinement idea with the same deployed representation, not a new
end-to-end quantization lifecycle. Its rollout must remain opt-in until full
module/model evidence and generic load/inference checks pass. Calibration-only
hard-loss rollback is a local fitting guard, not a production quality gate.
Any recovery factors or runtime caches bound to changed weights must be rebuilt.

## Initial validation

The later [complete F6/S7 QKV verification](../docs/experiments/gsq-p32-f6-seed7-full-qkv.md)
uses the original calibration corpus, complete real projections, and full-model
KLD/MSE/Top-N propagation. It supersedes the slice as decision evidence: the
full-model KLD improvement is only about 0.177%, with noise-consistent GSQ-inspired
Top-N changes. Neither method is promoted.

The subsequent [matched W2.5 YAQA comparison](../docs/experiments/gsq-p32-f6-seed7-w25-qkv.md)
freshly quantizes complete Q/K/V projections from real full-model Fisher
calibration. GSQ-inspired refinement slightly worsens final KLD and Top-1 on
the locked set despite tiny local fitting gains; it is not promoted.

`scripts/validate_qvq_gsq.py` uses the published local F6 snapshot and dense
Llama 3.2 1B weights. It captures first-layer q_proj inputs directly from real
token embeddings and RMSNorm, transforms them into the deployed inner basis,
then screens the first 32 input by 32 output coordinates (four P32 tiles).
It uses eight distinct chat-formatted documents truncated to 64 tokens: four
for refinement and four for evaluation. Disjointness is established for this
refinement; overlap with historical snapshot calibration has not been audited.

The 33 candidates per tile comprise the baseline and 32 seeded one-bit payload
mutations. They are not samples from an independently quantized full model.
The matched control searches the same candidates with three hard coordinate
descent sweeps. This control separates candidate availability from any advantage
of the relaxation. All arithmetic uses the CPU FP32 reference.

Raw reports and exact input/candidate tensors live in
`artifacts/gsq-p32/seed7-controlled/`; this partial experiment is intentionally
outside the complete-model snapshot directory. The report records row/token
identity, source hashes, seed, precision, byte counts, and paired document
bootstrap intervals. Correctness fixtures in `tests/test_qvq_gsq.py` cover all
six supported rates. Synthetic fixtures establish algebra/format correctness
only, never a model-quality result.

This is a slice reconstruction screen. It omits the rest of the input matrix,
full-layer output transforms, propagated logits, task scores and GPU execution.
A positive local result calls for larger real-model experiments, not promotion.
Joint scale learning, full-model lifecycle validation and RCO remain tracked in
[the TODO list](../docs/qvq_todos.md#gsq--rco-research--2026-09-09).

### Observed seed-7 results

| Arm | Mean held-out normalized output MSE | Payload bytes |
|---|---:|---:|
| Existing W2 P32 slice | 0.470590 | 256 |
| GSQ-inspired candidate relaxation | 0.425040 | 256 |
| Hard coordinate-search control | 0.418735 | 256 |

The relaxation reduces the local held-out metric by 9.68% versus the baseline.
The paired four-document bootstrap interval for its absolute difference is
[-0.04941, -0.03897]: **clear positive within this narrow slice**. The deterministic
control is better on all four documents: the observed benefit cannot be attributed
to Gumbel-Softmax specifically. This is evidence to improve and expand the
experiment, not to promote the relaxation or to reject scalar GSQ generally.
Full-model metrics remain unmeasured. No timing/speed claim is made.

Seven CPU correctness tests passed; Ruff and whitespace checks passed. Hard
exports round-trip exactly at W1, W1.5, W2, W2.5, W3 and W3.5; the real slice also
reloaded with exact reconstructed weights. GPU tests were not run.

### Metric meanings and missing measurements

The measured metric is normalized output MSE:
`mean((X W_candidate - X W_dense)^2) / mean((X W_dense)^2)`.
The table averages this ratio over the four held-out documents. Lower is better;
zero means identical slice outputs. A 9.68% reduction in this error is not a
9.68% increase in task accuracy. Raw MSE was not separately reported.

Final-logit KLD compares dense and quantized next-token probability distributions:
`sum_v p_dense(v) * log(p_dense(v) / p_quantized(v))`, averaged over valid token
positions. Lower is better; zero means matching distributions. It requires full
model propagation and was **not measured** here.

Top-1 agreement is the fraction of positions where the two models choose the
same highest-probability token. Top-N set overlap can be defined as
`|topN_dense intersect topN_quantized| / N`, averaged over token positions;
higher is better. Another metric sometimes called Top-N is whether the dense
top-1 token is anywhere in the quantized top-N set, so evaluation reports must
name the convention. Top-1/5/10 were **not measured** in this slice experiment.
Neither agreement metric by itself measures task correctness.

## Reproduction

```bash
PYTHONPATH=. /root/venv-py3.14t/bin/python -m scripts.validate_qvq_gsq \
  --gsq \
  --dense /monster/data/model/Llama-3.2-1B-Instruct \
  --snapshot /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32 \
  --data dataset/calibration_mix_128k_qwen3_0.6b/calibration.parquet \
  --output artifacts/gsq-p32/seed7-controlled
```

The output path must be new; existing evidence is never overwritten.

## Review follow-up: restricted search and correctness (2026-09-09)

Review through `26edfa3` correctly distinguishes this implementation from the
paper: QVQ searches a frozen pool of whole-tile payloads, with fixed scales,
Adam and a geometric temperature schedule. Each mutation is a single baseline
bit flip; more optimization steps do not accumulate multiple flips within a
tile. At W2.5, the default pool draws 32 mutations from 640 positions and may
contain duplicates. These restrictions must accompany recovery claims.

The scalar and QQQ objectives now use unnormalized loss when projected teacher
energy is at most FP32 epsilon, instead of dividing by FP32 tiny. The asymmetric
linear term is retained, including when the teacher lies in a metric nullspace.
QVQ rejects enabled scale learning in both `quantize_qvq_linear` and
`refine_trellis_fisher`, matching its public configuration restriction.
Focused CPU checks report 130 passed (5.51 seconds): scalar, QVQ configuration
and QQQ contract suites. This includes zero/tiny teacher targets and a
zero-projected-energy asymmetric case that must improve rather than early-return.

Required follow-up validation remains open:

- Completed: zero-weight RTN/GPTQ through their quantizer hooks, config round
  trip, packing, strict packed-state reload and Torch forward. Four CPU cases
  pass (3.31 seconds), covering fixed and learned scales; decoded weights and
  outputs remain exactly zero. These are regression fixtures, not quality evidence.
- Force a non-baseline QVQ tile choice through quantization, packing and reload.
- Compare GSQ with deterministic search over exactly the same candidate pool on
  matched real-model calibration and disjoint held-out data.
- Evaluate an evolving/multi-bit candidate search separately; current unchanged
  lifecycle payloads establish baseline retention, not effective recovery.

This is a GSQ-inspired experiment, not a reproduction of the paper's joint-scale,
Lion, scheduled-logit and staged reconstruction optimization method.
