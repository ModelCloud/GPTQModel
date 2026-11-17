# ParoQuant

## Rotation identity lifecycle

`ParoLinear` can skip activation rotation when both rotation buffers
describe the identity transform. The decision is derived from `theta` and
`channel_scales` when `clear_weight_cache()` invalidates runtime state. The
same method runs after `post_init()`, a state-dict load, and a module move.
Call it after manually changing rotation buffers, including tensors
created under `torch.inference_mode()` whose version cannot be tracked.

```mermaid
flowchart TD
    P["AwqTorchLinear.post_init() or train()"] --> A
    Q["BaseQuantLinear._load_from_state_dict()"] --> A
    R["BaseQuantLinear._apply() for module moves"] --> A
    S["Explicit cache clear after buffer mutation"] --> A
    A["ParoLinear.clear_weight_cache()"] --> B["AwqTorchLinear.clear_weight_cache()"]
    B --> C["_clear_rotation_runtime_cache()"]
    C --> D{"theta or channel_scales on meta?"}
    D -- "Yes" --> E["Set _rotation_identity to false"]
    E --> X["Meta storage cannot run forward"]
    X --> H["Materialized load or move calls clear_weight_cache() again"]
    H --> A
    D -- "No" --> F["is_identity_rotation(theta, channel_scales)"]
    F --> G["Store _rotation_identity"]
    G --> I["ParoLinear.forward() calls _rotate_inputs()"]
    I --> J{"_rotation_identity?"}
    J -- "Yes" --> K["Use input unchanged"]
    J -- "No" --> L["Apply ParoQuant rotation"]
    K --> M["AWQ packed matmul"]
    L --> M
    classDef blocked fill:#fce8e6,stroke:#a33,color:#111
    class E,X blocked
```

| Entry point | Previous mechanism | Current mechanism |
| --- | --- | --- |
| `post_init()` | `paroquant.py:162-166` recalculated `_rotation_identity` after the AWQ cache clear | Inherited AWQ `post_init()` calls `ParoLinear.clear_weight_cache()`, which recalculates it |
| Explicit cache clear | `paroquant.py:168-171` dropped runtime tensors but left the old identity decision | `clear_weight_cache()` drops those tensors and recalculates from current rotation buffers |
| State-dict load or module move | `qlinear/__init__.py:319-337` called the same clear hook, so the old decision survived changed buffers | The clear hook refreshes the decision; meta storage conservatively disables the shortcut until materialized |

The decision is a boolean, not a dense weight cache. The packed AWQ weights
remain the only persistent weight representation. The CPU dense fallback still
dequantizes a temporary weight for its current forward.

Introduced in [PR #3744](https://github.com/ModelCloud/GPTQModel/pull/3744).

## Activation Checkpointing

`ParoConfig.opt_gradient_checkpointing` controls activation checkpointing during ParoQuant's train-style optimization stages.

- `opt_scope="layer"` defaults to `opt_gradient_checkpointing=True`
- `opt_scope="module"` defaults to `opt_gradient_checkpointing=False`
- `opt_scope="compute_block"` defaults to `opt_gradient_checkpointing=False`

Current internal benchmarks have only shown a clear resource-usage benefit for `layer` scope. `module` and `compute_block` support the toggle, but they are not enabled by default because we have not yet measured a consistent memory win there.

## ROCm quantization determinism

ParoQuant derives per-module seeds from `opt_seed`, so a fixed seed is expected
to produce the same checkpoint. CUDA runs are bitwise-identical after
[PR #4499](https://github.com/ModelCloud/GPTQModel/pull/4499) made the fused
rotation gradient independent of the launch plan.

ROCm needed a separate fix. ParoQuant accumulates no Hessian, so it used to
keep hipBLASLt, which selects GEMM kernels per process. Two Qwen2.5-0.5B runs
with the same seed and calibration produced PPL `21.26` and `21.44`. The
quantization run plan now treats `ProcessorStep.PARO` like the Hessian builders
in `_needs_deterministic_hessian` (`gptqmodel/models/base.py`), so
`quant_blas_backend(deterministic=True)` pins rocBLAS for the whole run. Two
runs with rocBLAS pinned produced bitwise-identical checkpoints (issue #4460).

The trade-off is accepted: rocBLAS is deterministic but its fp16/bf16 GEMMs are
slower than hipBLASLt (8-14x on gfx1201 for the shapes in
[ROCm default environment](rocm-default-environment.md)). Deterministic
checkpoints are the product contract, so ParoQuant pays that cost. The
documented reproducibility opt-out remains
`KernelConfig(rocm_quant_blas="hipblaslt")`.

The fused-rotation autotune timing knobs are internal C++ constants now; they
are no longer read from `GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_*` environment
variables. `KernelConfig.paroquant_rotate_autotune`,
`KernelConfig.paroquant_rotate_cta_m` and `KernelConfig.paroquant_rotate_row_pad`
are the only controls.

| Test | Invariant |
| --- | --- |
| `test_paroquant_selects_the_deterministic_pin_on_mocked_rocm` | A `ProcessorStep.PARO` run plan selects the rocBLAS pin on a mocked ROCm platform |
| `test_paroquant_has_no_environment_variable_reads` | The ParoQuant extension sources contain no `GPTQMODEL_PAROQUANT_*` environment reads |

## Triton bias epilogue

ParoQuant Triton decode and prefill kernels accumulate quantized products in
FP32 when a bias is present, add the bias in that accumulator, and cast once to
the activation dtype. This preserves a finite FP16 result when the unbiased
intermediate exceeds the FP16 range. The dense plan uses
`matmul_fp32_reduce` for the same NVIDIA CUDA contract. A differentiable
rotated input or trainable bias uses the dense plan so autograd can return
its gradient. An unbiased fused call
keeps its existing accumulation setting, while a biased call uses FP32
accumulation even when `KernelConfig(fp32_accum=False)`. On NVIDIA CUDA, the dense plan
keeps the stored bias precision until the FP32 epilogue instead of casting it
to the activation dtype first.

```mermaid
flowchart TD
    A["ParoQuantTritonLinear.forward"] --> B["_select_plan"]
    B --> C{"_needs_dense_autograd(rotated)?"}
    C -- "Yes" --> D["Return dense plan"]
    C -- "No" --> E["Choose default or measured plan"]
    D --> F["_run_plan"]
    E --> F
    F --> G{"Selected plan"}
    G -- "Dense" --> H["_forward_triton_dense via matmul_fp32_reduce"]
    G -- "Decode or prefill fused" --> I{"Bias present?"}
    I -- "No" --> J["Existing GEMM accumulation and output store"]
    I -- "Yes" --> K{"Bias is floating [N] on input device?"}
    K -- "No" --> L["Raise ValueError before launch"]
    K -- "Yes" --> M["FP32 GEMM accumulator plus bias"]
    M --> N["Cast once to activation dtype"]
    classDef failure fill:#fce8e6,stroke:#a33,color:#111
    class L failure
```

| Path | Previous mechanism | Current mechanism |
| --- | --- | --- |
| Fused decode and prefill | `gemm.py:308` cast the GEMM result to FP16/BF16; `paroquant_triton.py:143-152` added bias afterward | The Triton kernel adds bias to its FP32 accumulator before its output store |
| Dense fallback | `paroquant_triton.py:136-138` rounded the matmul before bias and cast the stored bias to the activation dtype | `matmul_fp32_reduce` adds the stored bias in FP32 before NVIDIA CUDA output rounding |
| Differentiable input or bias | Fused output had no input gradient; separate post-kernel addition provided only a bias gradient | `_select_plan` chooses the differentiable dense wide path; `_run_plan` also guards direct calls |

On an NVIDIA GB10 (SM 12.1), a packed all-ones weight with 128 FP16 inputs of
`516` and a bias of `-1024` returned `inf` from both prior fused paths; an
independent FP32 Torch oracle and the corrected paths return `65024`. With
BF16 inputs of 127 ones plus `1.25`, and a bias of `-0.5`, the prior paths
returned `127.5`; the oracle and corrected paths return `128`. Tests cover
both dtypes, decode and prefill, the dense fallback, the unchanged unbiased
call, a noncontiguous bias, exact output against an independent packed-code
Torch oracle, malformed-bias rejection, mixed bias and activation dtypes, and
exact input and bias gradients. The CUDA wide reduction does not change
ROCm dense-path arithmetic; the fused ROCm bias path was not measured here.

For resident FP16 inputs with K=N=1024, five warmups and the median of five
runs of twenty calls with CUDA synchronization gave these per-forward results.
The prior behavior was reproduced by a bias-free kernel call followed by a
separate addition. Peak allocator delta includes the output allocations.

| Rows | Prior post-kernel bias | Fused bias |
| --- | --- | --- |
| 1 | 0.063 ms; 4 KiB peak | 0.055 ms; 2 KiB peak |
| 16 | 0.064 ms; 64 KiB peak | 0.056 ms; 32 KiB peak |

The Triton bias epilogue was introduced by [PR #3657](https://github.com/ModelCloud/GPTQModel/pull/3657).

## Native dispatch gate

`ParoLinear` advertises `group_size=16` because its CPU, ATen and dense paths
support it. The native AWQ CUDA GEMM does not:
`validate_awq_gemm_args` in
`gptqmodel_ext/awq/quantization/gemm_cuda_gen.cu` requires uint4b codes, a group
size divisible by 32, an output width divisible by the 64-column CTA tile, and
an output width divisible by that group size. Dispatch therefore asks the
kernel's own shape contract before calling it, instead of letting the kernel
raise from inside a forward.

```mermaid
flowchart TD
    A["ParoLinear.forward"] --> B["_forward_cuda_awq_kernel(rotated)"]
    B --> C["_cuda_awq_kernel_supported(compute_dtype)"]
    C --> D{"HIP build or non-CUDA packed weight?"}
    D -- Yes --> R["return None: portable path"]
    D -- No --> E{"group_size % 32 == 0?"}
    E -- No --> R
    E -- Yes --> F{"out_features % 64 == 0 and out_features % group_size == 0?"}
    F -- No --> R
    F -- Yes --> G{"BF16 activation on SM below 8.0?"}
    G -- Yes --> R
    G -- No --> H["Native AWQ GEMM"]
    R --> I["_forward_cpu_avx512 / _forward_cpu_aten / _forward_dense"]
    style R fill:#fee,stroke:#c00
```

The group size is derived from the packed tensors the kernel actually reads
(`in_features // scales.size(0)`), not from the config value: `group_size=-1`
packs one scale row per input channel, which the kernel turns into
`in_features / scaling_factors.size(0)`.

### Previous behavior

| Mechanism | Previous behavior | New behavior |
| --- | --- | --- |
| CUDA dispatch | Called the native GEMM for every CUDA tensor | Gated on the native validator's shape and device contract |
| `group_size=16` (advertised by `SUPPORTS_GROUP_SIZE`) | `RuntimeError: Group size should be a multiple of 32` | Dense fallback |
| Narrow output width, e.g. 8 | `RuntimeError: OC is not multiple of cta_N = 64` | Dense fallback |
| ROCm | `RuntimeError: The native AWQ JIT extension is CUDA-only and cannot be built by hipcc` | Portable path |
| FP16/BF16 activation beside the reconstructed FP32 grid | `RuntimeError: input dtypes must be the same` | Operands promoted to FP32, one cast at the end |

The gate is a capability query, not a configuration switch: there is no
environment override and no new public option.

### Dense fallback dtype contract

`linear_wide_bias` uses `torch.mm(..., out_dtype=torch.float32)` only when both
operands already share a low-precision dtype, which keeps them narrow and the
reduction wide. A mismatched pair (FP16 activation, FP32 reconstructed weight)
is promoted to at least FP32 before the matmul, so the fallback never narrows a
weight or an activation below FP32; the previous `.float()` arithmetic is
preserved bit for bit for same-dtype operands.

### Validation

Measured with torch 2.16.0.dev20261102+cu130 on an RTX 4090 (SM 8.9) and
torch 2.16.0.dev20261025+rocm10.1 on an R9700 (gfx1201). The oracle is the
independent FP32 packed-grid dequantization in
`tests/kernels/test_paroquant.py::_fp32_awq_oracle`.

| Configuration | Native gate | Result |
| --- | --- | --- |
| `in=16, out=8, group_size=16` | rejected (group size 16) | dense fallback, FP16 `65024` matches the oracle exactly |
| `in=16, out=8, group_size=-1` | rejected (output width 8) | dense fallback |
| `in=128, out=64, group_size=16` | rejected (group size 16) | dense fallback |
| `in=32, out=64, group_size=32` | admitted | native kernel, unchanged |
| `in=128, out=128, group_size=-1` | admitted (derived group size 128) | native kernel, unchanged |
| `in=256, out=128, group_size=128` | admitted | native kernel, unchanged |
| ROCm, `in=32, out=64, group_size=32` | rejected (HIP build) | dense fallback, matches the oracle |

Repeated fallback forwards keep only the packed `int32` buffers; the test
asserts that no `(in_features, out_features)` tensor is retained by the module.

## Output width alignment

An output width is packable only in multiples of eight. `pack_awq` writes eight
INT4 codes per `int32`, and ParoQuant reuses that packed layout for every
inference path (CPU ATen, the dense fallback, the AWQ CUDA GEMM and Triton), so
the alignment belongs to the format rather than to one kernel. Both the
selection-time check and the constructor read the same module constant
`PAROQUANT_OUT_FEATURES_ALIGN`:

```mermaid
flowchart TD
    A["ParoLinear.validate()<br/>inherited BaseQuantLinear._validate"] --> B["BaseQuantLinear._validate_shape_constraints()"]
    B --> C{"out_features divisible by SUPPORTS_OUT_FEATURES_DIVISIBLE_BY, which is [8]?"}
    C -- "No, for example 9" --> R1["False + NotImplementedError<br/>must be divisible by [8]"]
    C -- "Yes" --> D["ParoLinear.__init__"]
    D --> E{"out_features % PAROQUANT_OUT_FEATURES_ALIGN == 0?"}
    E -- "No, direct construction" --> R2["ValueError before super().__init__ allocates buffers"]
    E -- "Yes" --> F["AwqTorchLinear.pack(source, scales, zeros)"]
    F --> G["pack_awq(intweight, zeros, bits=4)"]
    G --> H{"iweights.shape[-1] % 8 == 0 and izeros.shape[-1] % 8 == 0?"}
    H -- "No" --> R3["ValueError from pack_awq<br/>defense in depth, unreachable through the public path"]
    H -- "Yes" --> I["qweight / qzeros int32 buffers"]
    classDef blocked fill:#fce8e6,stroke:#a33,color:#111
    class R1,R2,R3 blocked
```

`SUPPORTS_OUT_FEATURES_DIVISIBLE_BY` and the constructor guard both read the
single module constant `PAROQUANT_OUT_FEATURES_ALIGN`, so the declaration the
backend selector consumes, the constructor, and `pack_awq` cannot drift apart.
`ParoMlxQuantLinear` already declared `[8]` for the same layout; this change
makes the CUDA/CPU class agree with it.

### Previous behavior

| Mechanism | Previous behavior | New behavior |
| --- | --- | --- |
| Declaration consumed by selection | `SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]` at `gptqmodel/nn_modules/qlinear/paroquant.py:104` (pre-fix) | `[PAROQUANT_OUT_FEATURES_ALIGN]` = `[8]` at `paroquant.py:117` |
| Selection-time validation | `BaseQuantLinear._validate_shape_constraints` at `gptqmodel/nn_modules/qlinear/__init__.py:731-733` computed `out_features % 1`, so 9 passed | The same code computes `out_features % 8`, so 9 fails with `NotImplementedError` |
| Direct construction | `ParoLinear.__init__` (`paroquant.py:115-138` pre-fix) had no width check; 9 failed at `AwqTorchLinear.pack` (`torch_awq.py:141`) | `ParoLinear.__init__` raises `ValueError` at `paroquant.py:155-161`, before `super().__init__` |
| Auto-padding | `SUPPORTS_AUTO_PADDING = False` at `paroquant.py:102` (pre-fix), so selection never padded | Unchanged `False` at `paroquant.py:110`; no unpad exists, so the width is constrained |
| `pack_awq` on an unaligned width | `ValueError: INT4 AWQ packing requires ... divisible by 8` at `packing_utils.py:42-43` | Unchanged; now defense in depth, unreachable through the public path |

This is fail-closed rather than padded: `SUPPORTS_AUTO_PADDING` is `False` and
no unpad step exists in any ParoQuant forward, so padding the output width would
require changing packing and all four inference paths together. Declaring the
real alignment keeps every accepted width round-trippable.

### Validation

| Test | Invariant |
| --- | --- |
| `test_paroquant_output_width_declaration_matches_the_packer` | Declaration is `[8]`; `validate` rejects 9 and accepts 8, 16, 64 |
| `test_paroquant_constructor_rejects_an_unpackable_output_width` | Direct construction of width 9 raises before packing; width 8 still constructs |
| `test_paroquant_pack_and_forward_round_trips_at_the_packer_alignment` | Width 8 packs, forwards, and matches `_fp32_awq_oracle` exactly |

## Transformers loading

ParoQuant checkpoints are loaded with `GPTQModel.load`. They are not supported
by `transformers.AutoModelForCausalLM.from_pretrained`, because ParoQuant has
its own packed layout and kernel (`ParoLinear`), not a GPTQ layout.

Before issue #4074, `gptqmodel/__init__.py` registered
`quant_method="paroquant"` as an alias for transformers' `GPTQConfig` and
`GptqHfQuantizer`. That path renamed the method to `gptq`, so optimum kernel
selection then rejected the checkpoint with:

```text
ValueError: Unsupported format: `FORMAT.PAROQUANT` for quantization method `METHOD.GPTQ`
```

The alias was removed. `transformers` now fails cleanly with
`Unknown quantization type, got paroquant`, which names the actual problem
instead of a FORMAT/METHOD mismatch, and `GPTQModel.load` continues to load
ParoQuant checkpoints with `ParoLinear`.

## Fused rotation backward

When the rotation angles train, `_ParoQuantRotateTensorFunc.backward` runs
`rotate_pair_grad` (`gptqmodel_ext/paroquant/rotation.cu`). The op sweeps the
rotations in reverse. It rotates the saved output and the incoming gradient
back by `-theta[r]` in shared memory, and records each rotation's angle
gradient from the swept pair. A second launch, `rotate_grad_reduce_kernel`,
folds the per-row partials.

```mermaid
flowchart TD
    A["load x_grp = y, g_grp = g_out (CTA_M rows, one group)"] --> B["__syncthreads()"]
    B --> C["r = KROT-1 .. 0"]
    C --> D["apply_one(x_grp, -theta[r]); apply_one(g_grp, -theta[r])<br/>thread owns pair (i, j) of rotation r"]
    D --> E["read back own columns i, j for each 4-row granule<br/>partial = sum g_i x_j - g_j x_i"]
    E --> F["__syncthreads() after the readback"]
    F --> C
    F --> G["store x_grp, g_grp"]
    G --> H["rotate_grad_reduce_kernel: fold granules in ascending order"]
    D -. "rotation r-1 rewrites columns another thread still reads" .-> X["write-after-read race (#4461), closed by the barrier at F"]
    classDef blocked fill:#fce8e6,stroke:#a33,color:#111
    class X blocked
```

After the swept step, `x` is the rotation's input and `g = R^T g_out` is the
gradient with respect to that input. For `R = [[c, s], [-s, c]]`,
`dR/dtheta = R J` with `J = [[0, 1], [-1, 0]]`. The angle gradient is
therefore `g_out^T R J x = g^T J x = sum(g_i x_j - g_j x_i)`, with no
sin/cos factor.

| Contract | Before | Now |
| --- | --- | --- |
| Shared-memory ordering | One barrier between `apply_one` and the readback (`rotation.cu:417` at 4bc0413). Nothing ordered the readback of rotation `r` before the `apply_one` writes of rotation `r-1`, so the gradient was sporadically wrong (up to 4.2 on ROCm and 1.7 on CUDA at 2048 features). | The barrier comes after the readback (`rotation.cu:471`). It orders both the write-after-read and the read-after-write between rotations. A thread reads back only columns it wrote itself, so the readback needs no barrier of its own. |
| Launch-plan invariance | One partial per `CTA_M`-row tile (`rotation.cu:436`, `:516`). The autotune picks `CTA_M` by timing, so two cold runs could sum in different orders. Llama-3.2-1B picked 8 vs 16 for its 2048x2048 weight shape, and the checkpoints diverged from layer 0. | One partial per fixed 4-row granule (`kRotatePairGradRowGranule`), folded in ascending order. The result is bit-identical for every `cta_m`/`row_pad`. `cta_m=4` results did not change. |
| Angle gradient | `A cos(theta) - B sin(theta)` on the input-side gradient (`rotation.cu:436`, `paroquant.py:1143`). This is exact only at `theta == 0`: the error against fp64 autograd was 0.7 / 7.1 / 18.5 for \|theta\| up to 0.05 / 0.25 / 1.0. | `A = sum(g_i x_j - g_j x_i)` in `rotate_pair_grad`, `grad_theta` and the torch fallback. These now match fp64 autograd and the unfused optimizer path. |

The `grad_theta` op is reached only when `rotate_pair_grad` is unavailable.
It still folds its batch chunks with `atomicAdd`, so it is not bitwise
reproducible when the batch has more than 32 rows.

### Validation

| Test | Invariant |
| --- | --- |
| `test_paroquant_pair_grad_sweep_is_race_free` | 20 launches at 2048 features are bit-identical for every `cta_m`/`row_pad`, and the gradient matches fp64 autograd |
| `test_paroquant_pair_grad_is_launch_plan_invariant` | x, gradient and angle gradient are bit-identical across all 6 launch plans (1, 1000 and 1001 rows) |
| `test_paroquant_fused_rotation_theta_grad_matches_autograd` | `rotate_pair_grad`, `grad_theta` and the torch fallback, with and without channel scales, match fp64 autograd for \|theta\| up to 0.8 |
| `test_paroquant_pair_grad_reduce_folds_tiles_in_ascending_order` | The granule fold equals adding the per-4-row results one at a time |
