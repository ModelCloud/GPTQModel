---
name: qvq-ptq-recovery-research
description: Consult and maintain QVQ scientific notes for supported quantization methods, PTQ recovery, calibration and rounding enhancements, low-precision formats, P32 representations, or recirculation experiments.
---

# QVQ PTQ recovery research

Start with [research/README.md](../../../research/README.md), then read the notes
that change the current decision. Resolve these links relative to this file;
`research/` is at the repository root.

## Route the scientific question

- Supported methods, formats, dispatch and enhancement coverage:
  [method inventory](../../../research/supported-methods.md). Follow its method
  notes for GPTQ, AWQ, QQQ, ParoQuant, EXL3, RTN, FP8, bitsandbytes, GGUF and MXFP4.
- GPTAQ/FOEM, GAR, SLQ, rotation, SwiGLU, alignment, spectral recovery or pruning:
  use the inventory's enhancement map and
  [QVQ implementation findings](../../../research/qvq-enhancements.md).
- Activation/KV calibration, NVFP4 representation or fused scales:
  [NVFP4 hybrid PTQ](../../../research/nvfp4-hybrid-ptq.md).
- Low-rank compensation, rank selection or deployed-output fitting:
  [EoRA](../../../research/eora.md).
- Trellis coding and incoherence processing:
  [QTIP](../../../research/qtip.md).
- Full-model-sensitive rounding, Fisher sketches or “VAQA” terminology:
  [YAQA](../../../research/yaqa.md); do not invent a separate VAQA method.
- P32 banks, layout conversion or direct state decoding:
  [P32](../../../research/p32.md).
- Deep-to-shallow feedback, replay or recurrent cache state:
  [Recirculation](../../../research/recirculation.md).

## Apply the finding to the actual operator

Distinguish a METHOD enum, a configuration, a helper, a checkpoint format and a
runtime backend. Check actual caller support and exclusions; do not promote a
helper into a supported end-to-end method or infer A4 from four-bit weight storage.

Distinguish paper claims, repository implementation, measured results and proposed
extensions. Follow pinned source links for historical claims; inspect current code
before asserting present support. A paper's scale calibration is not gradient
training, and EoRA's covariance transform is not an NVFP4 scale.

For recovery, identify the teacher, deployed operator, target residual, activation
coordinate system and precision, factor dtype/rank, and scale/payload binding.
Check existing standard EoRA and `scripts/p32_twenty/native_recovery.py` before
proposing another fitter. Include actual A4 rounding when that is the target path.

Preserve the distinction between lossless representation changes and lossy
conversion or behavior changes. P32 window repacking does not repair quantization;
recirculation changes state evolution and has no established QVQ recovery benefit
merely because dense-model results improve.

Use [AGENTS.md](../../../AGENTS.md) and the existing implementation/evaluation
skills for their applicable gates. Keep kernel parity, propagated model quality,
task scores, storage and timing evidence separate. Research notes do not override
accuracy rules or promote runtime defaults.

## Maintain durable evidence

When a relevant scientific result changes a PTQ recovery decision, update its
note or add one focused Markdown note and link it from the index. Include primary
paper/version/section links, implementation revision, measured scope, limitations,
and explicitly labeled QVQ implications. Keep run-specific artifacts in existing
experiment locations and link them.

Do not present an unverified citation, copied headline or proposed experiment as
an established result. If an identifier is ambiguous, state the interpretation
and evidence. Preserve historical protocol labels and distinguish calibration
inputs from disjoint evaluation data.
