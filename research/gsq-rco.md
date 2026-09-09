# GSQ and RCO: P32 research entry

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

## Proposed QVQ adaptation and implementation status

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
the export authority. This helper has no production config or lifecycle dispatch.
It materializes decoded candidate tiles, so it is suitable for bounded experiments,
not yet memory-efficient full-model quantization. Joint scale learning remains TODO.

RCO is research-only and unimplemented. A QVQ implementation must count actual
serialized costs, respect backend-supported rates, and enforce feasibility after
hard assignment; an expected soft budget is insufficient. Keep this independent
of GSQ until each arm has a measured control.

### Proposed lifecycle insertion

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
  --dense /monster/data/model/Llama-3.2-1B-Instruct \
  --snapshot /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32 \
  --data dataset/calibration_mix_128k_qwen3_0.6b/calibration.parquet \
  --output artifacts/gsq-p32/seed7-controlled
```

The output path must be new; existing evidence is never overwritten.
