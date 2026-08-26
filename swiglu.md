# SwiGLU W2 quantization experiment

Date: 2026-08-26

This report records the first matched Smooth-SwiGLU versus no-Smooth control on
the cached Llama 3.2 1B Instruct model. The purpose is to test whether the
offline `up_proj`/`down_proj` reparameterization improves the actual nonlinear
MLP output rather than only improving a local linear reconstruction proxy.

## Matched setup

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Hardware: Apple M4 Max; MLX GPU inference; MPS quantization
- Quantizer: QVQ, W2, `format=qvq`, `rounding=block_ldlq`
- Scope: first decoder layer only; all seven attention/MLP projections in that layer
- Calibration: 8 identical rows from `dataset/calibration_mix_128k_qwen3_0.6b/calibration.parquet`
- Calibration token count: 2,722 non-padding tokens
- Smooth run: group size 16, maximum Smooth-SwiGLU scale-search calibration 64 tokens
- Control: `smooth_swiglu=None`
- Rate: 2.0 BPW for both checkpoints
- Evaluation: identical dense MLX reference, prompt, token IDs, and output shape
- Prompt: `Explain in one sentence why unit tests are useful.`
- Logit tensor shape: `[1, 12, 128256]`

Smooth-SwiGLU rescales only the dense `up_proj` rows and matching `down_proj`
columns before QVQ quantization. `gate_proj` is not rescaled. The dense model
function remains unchanged before quantization.

## Logit comparison

Relative L2 is defined as:

```text
||logits_quantized - logits_dense||_2 / ||logits_dense||_2
```

| Metric | No Smooth-SwiGLU | Smooth-SwiGLU | Absolute change | Smooth change |
|---|---:|---:|---:|---:|
| Relative L2 | 0.1312788093 | **0.1197898642** | -0.0114889451 | **8.7516% lower** |
| RMSE | 0.3996324725 | **0.3636707660** | -0.0359617065 | **8.9987% lower** |
| Maximum absolute error | 3.8955078125 | **3.2822265625** | -0.61328125 | **15.7433% lower** |
| Cosine similarity | 0.9916922450 | **0.9930137396** | +0.0013214946 | higher |

This is positive evidence that Smooth-SwiGLU reduces propagated first-layer
logit error at the same nominal BPW and runtime graph.

## Important guardrail result

The single-prompt top-1 result is mixed:

| Model | Dense top-1 token | Quantized top-1 token | Match |
|---|---:|---:|---:|
| No Smooth-SwiGLU | 8113 | 8113 | yes |
| Smooth-SwiGLU | 8113 | 2435 | no |

Therefore this experiment demonstrates a logit-distribution improvement, not
yet a task-quality improvement. The top-1 mismatch is a guardrail and must be
checked across multiple prompts and layers before Smooth-SwiGLU becomes a
default policy.

## Runtime and artifact checks

The no-Smooth control completed successfully through QVQ save, public MLX
reload, and MLX forward:

| Measurement | No Smooth-SwiGLU |
|---|---:|
| Quantization time | 64.6773 s |
| Save time | 1.1504 s |
| Dense MLX load | 0.9472 s |
| Public MLX reload | 4.3912 s |
| MLX forward | 0.1028 s |
| Checkpoint size | 2272.25 MB |

The Smooth run also completed save/reload/forward and produced the same
estimated 2.0 BPW checkpoint class. Smooth-SwiGLU adds no inference operation,
tensor, or stored scale requirement: the transformed weights are quantized and
stored directly.

## Interpretation and next gate

The result supports retaining Smooth-SwiGLU as an offline QVQ search
candidate. It does not justify claiming universal improvement because this is
one prompt and one quantized decoder layer. The next confirmation should use
multiple disjoint prompts and selected early/late layers, then a full-model
teacher-forced logit and generation comparison. Report both propagated logit
metrics and decision metrics; do not select Smooth-SwiGLU on local module loss
alone.
