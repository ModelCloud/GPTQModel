<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GGUF C++ native matmul: activation quantization, bound, and path rule

`GGUF_CPP` runs `ggml_mul_mat` on the packed weights. That kernel has no
full-precision activation path: the CPU backend converts the activations to
Q8_0/Q8_1 (legacy layouts) or Q8_K (K-quant layouts) before the integer dot
product, and the CUDA backend additionally accumulates in low precision. The
native result therefore differs from an FP32 dequantized-weight oracle by
~5e-3..1.5e-2 even when the activations are already FP32 (issue #4219).

`GGUF_TORCH` is the exact path: it dequantizes the weights and multiplies in
FP32, so it matches the oracle to FP32 rounding. `GGUF_CPP` is the fast path and
is 40-97x faster per layer on CPU (measured below), which is why it stays in the
AUTO candidate list.

## Where the error comes from

The loss is entirely in the activation quantization, not in the packed weights,
the dequantization layout, or an accumulation bug in this module. The evidence
is that the native result equals an **independent** Q8 round trip of the
activations (ggml's `quantize_row_q8_0_ref`/`quantize_row_q8_K_ref` followed by
`dequantize_row_*`, then an FP64 matmul with exactly dequantized weights):

| Layout family | CPU residual vs Q8 simulation |
| --- | --- |
| Q1_0, Q1_0_g128, Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, MXFP4, NVFP4 | <= 6.8e-8 |
| Q4_1, Q5_1 (vec_dot also consumes a quantized activation sum) | <= 1.5e-4 |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, TQ1_0, TQ2_0, IQ1_*, IQ2_*, IQ3_*, IQ4_XS | <= 1.4e-7 |

For every layout the *other* activation type is worse by at least 1e-3, which
pins the mapping (legacy layouts use Q8_0/Q8_1, K-quant and IQ layouts use
Q8_K). `tests/kernels/test_gguf_cpp_activation_quantization.py` asserts both
facts, so the error cannot silently move to a different cause.

Q8_K stores one FP32 scale per 256-element super-block while Q8_0 stores one
FP16 scale per 32 elements. That is why K-quant layouts land near 7e-3 and the
legacy layouts near 5e-3: it is ggml's block layout, not a defect here.

## Measured error and the achievable bound

FP64 dequantized-weight oracle, 64x256 layer, `torch.manual_seed(7)` weights and
`torch.manual_seed(11)` activations, `M` in {1, 8, 32}, all 16 quantized layouts
(`Q1_0`, `Q2_0`, `Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_0`, `Q5_1`, `Q5_K`,
`Q6_K`, `Q8_0`, `IQ4_NL`, `IQ2_XS`, `MXFP4`, `NVFP4`):

| Activations | CPU | CUDA |
| --- | --- | --- |
| FP32 | 4.9e-3 .. 7.9e-3 | 4.9e-3 .. 1.1e-2 |
| FP16 | 4.9e-3 .. 7.5e-3 | 5.9e-4 .. 9.3e-4 |
| BF16 | 4.9e-3 .. 7.1e-3 | 4.9e-3 .. 9.7e-3 |

Worst case observed across shapes: 1.5e-2 (Q4_1, 48x256, M=2, FP32 activations,
CUDA). `GGML_NATIVE_MATMUL_ERROR_BOUND = 2e-2` in
`gptqmodel/nn_modules/qlinear/gguf_cpp.py` is the documented bound.

The achievable bound is one Q8 code step per element: symmetric 8-bit
quantization of a block with maximum `amax` has `d = amax/127` and
`|x - x_q| <= d/2`, i.e. 1/254 = 3.9e-3 of the block maximum. A dot product
whose terms do not cancel stays within roughly twice that (5-8e-3 measured); the
CUDA FP32/BF16 paths add ggml-cuda's low-precision accumulation on top
(1.1e-2 measured, 1.5e-2 worst shape), and the CUDA FP16 path is 6-10x better
because it takes a different kernel (5.9e-4..9.3e-4). None of this can be
tightened from this repository: `ggml_mul_mat` accepts only Q8 activations, so
any further reduction means leaving the native kernel for `GGUF_TORCH`.

## Which path a forward takes

The choice is a pure function of static configuration. It is never decided by
timing the two implementations, by activation values, or by call order, so a
checkpoint takes the same path in every process.

| Backend | Product | Bias epilogue |
| --- | --- | --- |
| CPU | native ggml matmul | FP32 bias added to the native result, rounded once |
| CUDA | native ggml matmul | biased layers are recomputed exactly in FP32 (#3789) |
| CPU/CUDA, FP16/BF16 native output | native ggml matmul | biased layers are recomputed exactly in FP32 |

The biased CUDA recompute is not an accident of `bias is not None`: a native
product can be unusable for a bias even when ggml returns an FP32 tensor, and a
low-precision output cannot represent a large product before the bias cancels
it. Measured: Q4_0 on CUDA with FP32 activations of magnitude 1e4 returns NaN
in an FP32 output tensor for a finite result, so adding an FP32 bias afterwards
cannot recover the value. Biased layers therefore take the exact path; unbiased
layers take the native path, whose loss is the bound above. Both branches are
covered by `tests/kernels/test_gguf_cpp_bias_epilogue.py` and
`tests/kernels/test_gguf_cpp_activation_quantization.py`.

`GGUF_TORCH` remains the exact engine. On CPU the native engine is 40-97x faster
per layer, so it stays the AUTO preference; a caller that needs the exact result
selects `BACKEND.GGUF_TORCH` explicitly. See `docs/gguf-kernel-selection.md`.

## Validation

- `tests/kernels/test_gguf_cpp_activation_quantization.py`:
  `test_cpu_native_result_equals_independent_q8_activation_simulation`
  (all layouts), `test_cpu_native_error_stays_within_the_documented_bound`,
  `test_cuda_native_error_stays_within_the_documented_bound`,
  `test_biased_cuda_layer_is_recomputed_exactly`,
  `test_exact_epilogue_rule_is_a_static_function_of_bias_presence`,
  `test_native_path_is_not_chosen_by_timing`,
  `test_native_output_is_bitwise_repeatable`,
  `test_zero_bias_does_not_change_the_cpu_matmul_path`.
- `tests/kernels/test_gguf_cpp_path_selection.py` pins the capability-based CPU
  choice (added for #4219 in #4266).
- `tests/kernels/test_gguf_cpp.py` and
  `tests/kernels/test_gguf_cpp_bias_epilogue.py` keep the end-to-end kernel and
  bias contracts.

Measured on an RTX 4090 with PyTorch 2.15.0+cu132, llama-cpp-python 0.3.36
(ggml 0.25.3), 4 threads.
