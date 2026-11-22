<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GGUF C++ native matmul: activation quantization, measured error, and path rule

`GGUF_CPP` runs `ggml_mul_mat` on the packed weights. That kernel has no
full-precision activation path: the CPU backend converts the activations to
Q8_0/Q8_1 (legacy layouts) or Q8_K (K-quant layouts) before the integer dot
product, and the CUDA backend additionally accumulates in low precision. The
native result therefore differs from an FP32 dequantized-weight oracle even when
the activations are already FP32 (issue #4219).

How large that difference is depends on the activation distribution. It is
~5e-3..1.5e-2 for Gaussian activations, but **outlier channels exceed 2e-2**
because ggml quantizes activations in blocks with a shared scale (#4316). There
is no input-independent error bound for the native kernel.

`GGUF_TORCH` is the exact path: it dequantizes the weights and multiplies in
FP32, so it matches the oracle to FP32 rounding. `GGUF_CPP` is the fast path and
is 40-97x faster per layer on CPU (measured below), which is why it stays in the
AUTO candidate list. A CPU `GGUFCppKernel` can also be constructed with
`exact_matmul=True` to take the exact path for that layer (see
[Forcing the exact path](#forcing-the-exact-path)).

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
legacy layouts near 5e-3 on Gaussian inputs: it is ggml's block layout, not a
defect here. The same block layout is why outliers are worse on K-quant
layouts: one large channel raises the scale for all 256 values in its block.

## Measured error: Gaussian vs outlier channels

### Gaussian activations

FP64 dequantized-weight oracle, 64x256 layer, `torch.manual_seed(7)` weights and
`torch.manual_seed(11)` activations, `M` in {1, 8, 32}, all 16 quantized layouts
(`Q1_0`, `Q2_0`, `Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_0`, `Q5_1`, `Q5_K`,
`Q6_K`, `Q8_0`, `IQ4_NL`, `IQ2_XS`, `MXFP4`, `NVFP4`):

| Activations | CPU | CUDA |
| --- | --- | --- |
| FP32 | 4.9e-3 .. 7.9e-3 | 4.9e-3 .. 1.1e-2 |
| FP16 | 4.9e-3 .. 7.5e-3 | 5.9e-4 .. 9.3e-4 |
| BF16 | 4.9e-3 .. 7.1e-3 | 4.9e-3 .. 9.7e-3 |

Worst Gaussian case observed across shapes: 1.5e-2 (Q4_1, 48x256, M=2, FP32
activations, CUDA); on CPU a Gaussian Q6_K FP32 case reaches 1.8e-2.
`GGML_NATIVE_MATMUL_ERROR_BOUND = 2e-2` in
`gptqmodel/nn_modules/qlinear/gguf_cpp.py` is the envelope the native path is
validated against for this distribution. It is **not** a bound for every input.

### Outlier channels

The #4316 repro uses a Q4_K_M layer (K=4096, N=128, M=64) and scales 16 of the
4096 activation channels by 100, which is representative of LLM hidden states:

```python
import torch
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel

K, N = 4096, 128
g = torch.Generator().manual_seed(0)
cpp, exact = [
    c(bits="q4_k_m", group_size=-1, sym=True, desc_act=False,
      in_features=K, out_features=N, bias=False)
    for c in (GGUFCppKernel, GGUFTorchLinear)
]
raw = torch.randint(0, 256, cpp.qweight.shape, generator=g, dtype=torch.uint8)
raw.view(N, -1, 144)[..., 0:4] = torch.tensor(
    [1e-3, 1e-3], dtype=torch.float16
).view(torch.uint8)
for m in (cpp, exact):
    m.qweight.copy_(raw)
    m.eval()

x = torch.randn(64, K, generator=g)
xo = x.clone()
xo[:, torch.randperm(K, generator=g)[:16]] *= 100.0
for name, xx in (("gaussian", x), ("outlier", xo)):
    with torch.inference_mode():
        ref, out = exact(xx).double(), cpp(xx).double()
    print(
        name,
        "rel %.4f" % ((out - ref).abs().max() / ref.abs().max()).item(),
        "worst row %.4f"
        % ((out - ref).abs().amax(1) / ref.abs().amax(1)).max().item(),
    )
```

Measured with this repro:

| Activations | Relative error | Worst row | Within 2e-2? |
| --- | --- | --- | --- |
| Gaussian | 0.0087 | 0.0138 | yes |
| 16/4096 channels x100 | 0.033 .. 0.038 | 0.076 .. 0.084 | **no** |

A wider sweep (Q4_K/Q6_K, K in {4096, 11008}, FP32/BF16/FP16) measures
0.030 .. 0.041 relative on CPU and up to 0.0167 on CUDA FP32/BF16. The worst row
in the #4316 report is **8.4e-2**, 4.2x the documented 2e-2. The magnitude is
input-dependent: it grows with the ratio between the largest channel in a block
and the rest of that block, and it is not possible to state a finite bound from
this repository.

### Why the per-element step does not bound the result

The quantization step for a block with maximum `amax` is `d = amax/127` for the
8-bit codes, so each element is within `d/2` (1/254 = 3.9e-3 of the block
maximum). That is a bound on the *activation* error relative to the block
maximum, not on the dot product. When one channel dominates a block, every other
element in that block is quantized coarsely relative to its own magnitude, and
the dot product's relative error depends on how the output row's terms cancel.
The Gaussian measurements stay at 5-8e-3 because their block maxima are similar;
the outlier repro reaches 3-8e-2 for exactly this reason. None of this can be
tightened from this repository: `ggml_mul_mat` accepts only Q8 activations, so
any further reduction means leaving the native kernel for `GGUF_TORCH`.

## Which path a forward takes

The default choice is a pure function of static configuration. It is never
decided by timing the two implementations, by activation values, or by call
order, so a checkpoint takes the same path in every process (#4219). An
activation-dependent switch was rejected because it would make the numerics
depend on the input distribution in a way that cannot be reproduced from the
config, and it would pay the outlier-detection cost on every forward.

| Configuration | Product | Bias epilogue |
| --- | --- | --- |
| CPU, default (`exact_matmul=None`, `GGUFCPPConfig(native_matmul=None)`) | native ggml matmul | FP32 bias added to the native result, rounded once |
| CPU, `exact_matmul=True` or `GGUFCPPConfig(native_matmul=False)` | exact torch matmul | FP32 bias added to the exact product |
| CPU, `exact_matmul=False` | native ggml matmul | FP32 bias added to the native result, rounded once |
| CUDA | native ggml matmul | biased layers are recomputed exactly in FP32 (#3789) |
| CPU/CUDA, FP16/BF16 native output | native ggml matmul | biased layers are recomputed exactly in FP32 |

The biased CUDA recompute is not an accident of `bias is not None`: a native
product can be unusable for a bias even when ggml returns an FP32 tensor, and a
low-precision output cannot represent a large product before the bias cancels
it. Measured: Q4_0 on CUDA with FP32 activations of magnitude 1e4 returns NaN
in an FP32 output tensor for a finite result, so adding an FP32 bias afterwards
cannot recover the value. Biased layers therefore take the exact path; unbiased
layers take the native path, whose loss is the measured envelope above. Both
branches are covered by `tests/kernels/test_gguf_cpp_bias_epilogue.py` and
`tests/kernels/test_gguf_cpp_activation_quantization.py`.

`GGUF_TORCH` remains the exact engine. On CPU the native engine is 40-97x faster
per layer, so it stays the AUTO preference; a caller that needs the exact result
selects `BACKEND.GGUF_TORCH` explicitly or uses `exact_matmul=True`. See
`docs/gguf-kernel-selection.md`.

## Forcing the exact path

Three selectors reach the exact (`GGUF_TORCH`-equivalent) CPU numerics, in
decreasing scope:

1. **Per layer (API):** construct the CPU kernel with
   `GGUFCppKernel(..., exact_matmul=True)`. The keyword is accepted by
   `GGUFCppKernel.__init__` and wins over the backend setting; the default
   `exact_matmul=None` keeps the existing native choice, so callers that pass
   nothing see no behaviour change.
2. **Per load (typed backend):** load with
   `backend=GGUFCPPConfig(native_matmul=False)`. Each `GGUFCppKernel` built by
   that load snapshots the setting (`KernelConfig.gguf_cpp_native_matmul`). The
   retired `GPTQMODEL_GGUF_CPP_NATIVE_MATMUL` environment variable is not read;
   setting it only logs a warning.
3. **Engine (config):** load with `BACKEND.GGUF_TORCH`, which never uses the
   native kernel at all.

## Validation

- `tests/test_gguf_cpp_matmul_accuracy_4316.py` (added for #4316):
  `test_gaussian_activations_stay_within_the_documented_envelope`,
  `test_outlier_channels_exceed_the_documented_envelope_on_the_native_path`,
  `test_exact_matmul_selector_takes_the_exact_path`,
  `test_default_path_is_unchanged`.
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
(ggml 0.25.3), 4 threads. The #4316 outlier numbers were reproduced on the host
CPU with llama-cpp-python's CPU backend.
