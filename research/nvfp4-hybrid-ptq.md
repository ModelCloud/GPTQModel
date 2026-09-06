# NVFP4 hybrid PTQ: scales, fusion, and recovery

## Sources

- Kozyrev and Maiboroda, *Why Gated DeltaNet Survives 4-Bit Quantization:
  NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM*,
  [arXiv:2609.04098v1](https://arxiv.org/html/2609.04098v1).
- Authors' [Minima checkpoint and recipe](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4).
- Companion [EoRA note](eora.md) and [EoRA paper](https://arxiv.org/abs/2410.21271).

## Source findings

The paper uses calibration-only PTQ, not learned activation scales or QAT.
NVFP4 represents weights and activations with E2M1 values, E4M3 scales per
16-element block, and an FP32 tensor scale (§§2–3).

Its GDN explanation combines localized outlier damage, gate nonlinearities that
attenuate projection error, and delta-rule overwrites that erase state error.
This is architecture-specific evidence, not permission to quantize arbitrary
recurrent states (§5). Reported GDN state error plateaus near 12.6% over 32K tokens;
repairing fused scales reduces the GEMM probe's relative error to approximately 0.002.

Per-module global weight scales became incorrect when serving fused projections
under one global scale. The repair compensates block scales when globals are
harmonized; E4M3 re-rounding introduces residual error (§6).

FP8 KV calibration is separate: 32 static K/V scale tensors for 16 attention
layers improve 32K perplexity from 10.84 to 10.50, recovering 83% of the
uncalibrated-cache penalty. GDN layers have no conventional KV cache (§7).

Evidence covers one model family/size, 32K perplexity and 64K retrieval.
Scale-augmented task scores were inherited, not rerun. A QAT comparator was not
benchmarked. Broken scales misleadingly improved long-context perplexity.
Text-only serving and correct chat templates were necessary controls (§§6, 8–9).
[Paper](https://arxiv.org/html/2609.04098v1)

The model card specifies 496 backbone linears at W4A4; embeddings, head,
convolution, norms and gate parameters remain BF16. It identifies
`NVFP4` plus static per-tensor `kv_cache_scheme`, a 128 × 32K calibration set,
and fused groups `in_proj_qkv+z` and `in_proj_b+a`.
It explicitly excludes QAT and distillation.
[Checkpoint recipe](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4)

## QVQ design implications

These are engineering deductions and proposed validation, not additional paper results.

### Freeze the representation contract before fitting recovery

Keep distinct metadata for weight globals/block scales, activation
globals/block scales, and cache K/V scales. Record scale versus reciprocal-scale
conventions explicitly. Activation codes and local scales depend on runtime
inputs; a calibrated global does not replace runtime block quantization.

A useful dequantization convention is

$$
\widehat v_b = g\,s_b\,q_b.
$$

Under this convention, replacing a projection's global scale `g_i` with
`g_shared` requires `s'_b = s_b * g_i / g_shared`.
This algebra preserves values before scale-format rounding. If the API stores
inverse scales, derive the corresponding conversion rather than copying this
ratio. Check clipping, underflow, zeros, and post-rounding reconstruction.

Calibrate shared-input projections for the intended fused activation operand.
Weight harmonization and sharing activation quantization are separate decisions.
Probe each projection independently and the fused result using identical inputs;
a plausible model score cannot certify the scale contract.

### Fit the actual deployed residual

Use the existing [output-residual recovery machinery](eora.md#existing-qvq-output-residual-research)
as the starting point. Capture the real NVFP4 path, including transforms,
activation rounding, packed weight conversion, scales and epilogue. Fit factors
using the same activation domain and precision consumed by the correction branch.

If a trellis-decoded weight is subsequently converted to NVFP4, that conversion
is another lossy step; a W4 storage label alone does not establish native NVFP4
GEMM compatibility. Lossless [P32 window repacking](p32.md) does not perform this conversion.

### Minimal discriminating comparison

Compare the same checkpoint/input protocol under W4A16, W4A4 with calibrated
scales, W4A4 with weight-residual correction, and W4A4 with deployed-output
correction. Then isolate BF16 KV, unit-scale FP8 KV, and calibrated FP8 KV.
Evaluate correction off/on after export and reload.

Keep kernel drift, teacher agreement, task quality, long-context behavior,
effective BPW, prefill and decode cost separate. A rank-8 hypothesis is not
evidence of sufficient rank, and calibration-only success here is not evidence
that QAT can never help QVQ.
