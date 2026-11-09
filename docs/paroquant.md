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

## Triton bias epilogue

ParoQuant Triton decode and prefill kernels accumulate quantized products in
FP32 when a bias is present, add the bias in that accumulator, and cast once to
the activation dtype. This preserves a finite FP16 result when the unbiased
intermediate exceeds the FP16 range. The dense plan uses
`matmul_fp32_reduce` for the same NVIDIA CUDA contract. A differentiable
rotated input or trainable bias uses the dense plan so autograd can return
its gradient. An unbiased fused call
keeps its existing accumulation setting, while a biased call uses FP32
accumulation even when `GPTQMODEL_FP32_ACCUM=0`. On NVIDIA CUDA, the dense plan
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
