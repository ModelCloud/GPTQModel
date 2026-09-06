# PTQ recovery research

This directory records scientific findings that affect QVQ quantization, recovery,
and inference. Read the relevant note before changing the corresponding contract.
It is a research index, not a claim that every referenced method is implemented,
validated on QVQ, or enabled in production.

## Reading map

| Topic | Note | Role |
|---|---|---|
| NVFP4 W4A4 and FP8 KV calibration | [NVFP4 hybrid PTQ](nvfp4-hybrid-ptq.md) | Numeric representation, fusion correctness, calibration |
| EoRA and output-residual fitting | [EoRA recovery](eora.md) | Additive low-rank compensation |
| QTIP | [Trellis quantization](qtip.md) | Weight quantizer and parallel-decodable representation |
| YAQA (requested as “VAQA”) | [Model-preserving rounding](yaqa.md) | Full-model-sensitive weight rounding |
| QVQ V2B2-P32 | [P32 and lossless windows](p32.md) | Repository format and runtime contract |
| Google DeepMind Recirculation | [Recirculation](recirculation.md) | Inference-time state intervention; PTQ benefit unproven |

“VAQA” is interpreted here as **YAQA**, consistent with the repository's
`qvq_yaqa.py` and the linked paper. No separate VAQA publication is asserted.
P32 is documented as a QVQ implementation, not an independently identified paper.

## How the pieces relate

Quantizer choice, rounding objective, numeric scales, storage layout, additive
recovery, and recurrence are different axes. A lossless layout conversion cannot
recover quantization error. A fitted low-rank correction cannot replace a required
NVFP4 scale. An inference intervention can improve a task while moving away from
the original teacher; report those outcomes separately.

For proposed W4A4 recovery, distinguish:

- Calibration: estimating quantization parameters from representative inputs.
- Reconstruction fitting: solving for weights or factors against a specified target.
- QAT/distillation: optimizing with a quantized forward path and a training objective.
- Runtime quantization: producing input-dependent codes and block scales.

Calling all four “training scales” loses the distinction needed for implementation.

## Evidence convention

Each note separates **source findings**, **repository evidence**, and **QVQ
implications or proposed experiments**. Paper numbers are author-reported unless
a linked repository report explicitly reproduces them. Cite the paper version and
section, pin implementation evidence to a commit, and record scope and limitations.
Do not copy entire papers or treat an abstract's headline as a universal guarantee.

The initial QVQ source audit is pinned to
[`4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac`](https://github.com/ModelCloud/QvQ/tree/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac).
Verify current code before acting on an implementation statement. Source review
for these notes did not run GPU or model-quality experiments.

## Experiment records

Keep run-specific results in the existing `docs/experiments/` and `artifacts/`
locations; link the evidence here when a durable finding is established. Record
model/checkpoint revision, exact operator and activation domain, W/A/KV precision,
scale convention, correction rank/dtype, calibration/evaluation split, hardware,
backend, and source revision.

Use matched-input kernel references for correctness and disjoint real-model
evaluation for quality. Preserve the acceptance rules in [AGENTS.md](../AGENTS.md).
Measure conversion, transforms, correction, and addition in full-operator timing;
report prefill and decode separately. Include scale, bank, padding, and correction
storage in effective BPW. A note or successful reference fit does not promote a
runtime default.

Agent entry point:
[PTQ recovery research skill](../.agents/skills/qvq-ptq-recovery-research/SKILL.md).
