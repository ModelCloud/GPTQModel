<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Native MLX quantization kernel plan

Scope: quantization on Apple silicon. Each pull request adds one independently
reviewable kernel or algorithm stage, its public entry point, and a deterministic
Torch-oracle A/B test. Keep the existing Torch paths and checkpoint layouts as
the compatibility reference. Inference kernels are a separate effort.

## Acceptance gate for every pull request

1. Document the exact tensor layout, supported shapes and dtypes, and any
   unsupported options. Reject unsupported cases explicitly.
2. Compare packed codes and other discrete outputs exactly with an independent
   Torch oracle. Compare floating quantization outputs with `rtol <= 1e-6` and
   `atol <= 1e-6`; compare scale-dependent matrices by normalized error
   `<= 1e-6`. Include zero blocks and numerical boundary cases.
3. Test on Apple silicon, then benchmark against the same operation on `main`
   with identical inputs, warmup, synchronization, and a reported median.
4. Verify the result can be saved and consumed by its intended runtime before
   calling a method complete. Include the test command and measured errors in
   the PR description.

## Pull request sequence

| Order | Kernel or method stage | Compatibility target |
| --- | --- | --- |
| 1 | GGUF `Q4_0` block packing | Existing GGUF 18-byte block layout |
| 2 | GGUF `Q8_0` block packing | Existing GGUF 34-byte block layout |
| 3 | GGUF `Q1_0` and `Q1_0_g128` sign packing | Shared 128-value block layout |
| 4 | GGUF `Q2_0` block packing | Existing 64-value block layout |
| 5 | GGUF `Q4_K` block packing | All `q4_k*` aliases |
| 6 | GGUF `Q5_K` block packing | All `q5_k*` aliases |
| 7 | GGUF `Q6_K` block packing | Existing `q6_k` layout |
| 8 | GGUF `TQ1_0` block packing | Existing ternary layout |
| 9 | GGUF `TQ2_0` block packing | Existing ternary layout |
| 10 | GGUF `MXFP4` block packing | Existing microscaling layout |
| 11 | FOEM correction | `alpha` and `beta` options over GPTQ |
| 12 | ParoQuant pair rotation | Forward and inverse transformed-domain math |
| 13 | ParoQuant group quantization | Learned scale and zero-point packing |
| 14 | QQQ quantization | Existing QQQ packed weights and scales |
| 15 | FP8 quantization | Existing FP8 scale and payload format |
| 16 | EXL3 quantization | Existing EXL3 codebook and packed layout |
| 17 | BitsAndBytes quantization | Existing BitsAndBytes format and configuration |

GPTQ and AWQ already have MLX quantization paths. RTN is a GPTQ-family
weight-only option; add its native path with the GPTQ follow-up that preserves
its existing output layout. GPTAQ, EoRA, and GAR are modifiers or postprocessing
stages, so schedule their MLX kernels after the relevant base method passes the
same oracle and checkpoint gate. `NVFP4` is currently a GGUF dequantization-only
format in this repository; it belongs to the later inference effort.
