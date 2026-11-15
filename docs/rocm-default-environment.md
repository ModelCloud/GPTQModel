<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ROCm default process environment

`import gptqmodel` installs two process-wide defaults on ROCm that change the
numbers a quantization run produces. Both are installed before the first BLAS
call and both defer to a value the caller already set:

| Variable | ROCm default | Why |
|---|---|---|
| `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:256,garbage_collection_threshold:0.7` | `expandable_segments:True` plus `torch.cuda.empty_cache()` corrupts ordinary Torch results on ROCm 10.1 / gfx1201 |
| `ROCBLAS_USE_HIPBLASLT` | `0` | rocBLAS forwards GEMMs to hipBLASLt, which selects a different kernel per process for the same shape; identical quantization runs then save different checkpoints |

CUDA keeps the historical allocator default
(`expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`)
and never has `ROCBLAS_USE_HIPBLASLT` set by GPT-QModel.

## The flow

```mermaid
flowchart TD
    A["import gptqmodel"] --> B["_torch_safety.ensure_safe_torch"]
    B --> C["_alloc_conf.apply_default_alloc_conf"]
    C --> D{"running_on_rocm"}
    D -- "no" --> E{"caller set PYTORCH_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF"}
    D -- "yes" --> F{"caller set PYTORCH_ALLOC_CONF, PYTORCH_CUDA_ALLOC_CONF or PYTORCH_HIP_ALLOC_CONF"}
    E -- "yes" --> G["preserve the caller value"]
    E -- "no" --> H["install DEFAULT_ALLOC_CONF with expandable_segments"]
    F -- "yes" --> G
    F -- "no" --> I["install ROCM_DEFAULT_ALLOC_CONF without expandable_segments"]
    H --> J["_rocm_env.apply_default_rocm_blas_env"]
    I --> J
    G --> J
    J --> K{"running_on_rocm"}
    K -- "no" --> L["leave ROCBLAS_USE_HIPBLASLT untouched"]
    K -- "yes" --> M{"caller set ROCBLAS_USE_HIPBLASLT"}
    M -- "yes" --> N["preserve the caller value"]
    M -- "no" --> O["set ROCBLAS_USE_HIPBLASLT=0"]

    classDef fail fill:#ffe0e0,stroke:#c62828,color:#000
    class G,N fail
```

`apply_default_alloc_conf()` is also re-run from `gptqmodel/models/auto.py` so
importing that module directly still gets the default. The ROCm BLAS default is
applied once, from `gptqmodel/__init__.py`, before `gptqmodel.utils` is
imported.

## Before and after

| Mechanism | Before | After |
|---|---|---|
| Allocator default | `DEFAULT_ALLOC_CONF` installed unconditionally (`gptqmodel/_alloc_conf.py:26-30`, applied at `gptqmodel/__init__.py:29`) | `default_alloc_conf()` picks `ROCM_DEFAULT_ALLOC_CONF` when `torch.version.hip` is set (`gptqmodel/_alloc_conf.py`) |
| ROCm allocator detection | Only `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` counted as caller configuration | ROCm also honours `PYTORCH_HIP_ALLOC_CONF`, which Torch reads before `PYTORCH_ALLOC_CONF` |
| rocBLAS selection | rocBLAS forwarded its GEMMs to hipBLASLt with no GPT-QModel default | `_rocm_env.apply_default_rocm_blas_env()` exports `ROCBLAS_USE_HIPBLASLT=0` unless the caller set it |
| CUDA behaviour | `expandable_segments:True` default | unchanged; no ROCm-only variable is written |

## Measured on gfx1201 (R9700)

Hardware: AMD Radeon AI PRO R9700, `gfx1201`, torch 2.16.0.dev+rocm10.1
(HIP 7.16.26385).

| Check | Before | After |
|---|---|---|
| 2097152x16 matmul with `empty_cache()` between 3 reps (issue #4072) | 0 / 196608 / 294912 bad rows across runs | 0 bad rows |
| Qwen2.5-0.5B-Instruct GPTQ 4-bit g128, 32 samples, two processes (issue #4075) | 334/795 saved tensors differ | 0/795 differ |

## Failure and ambiguity paths

| Situation | Behaviour |
|---|---|
| Caller sets any allocator variable, including `PYTORCH_HIP_ALLOC_CONF` | GPT-QModel writes nothing; the caller's value is used verbatim |
| Caller sets `ROCBLAS_USE_HIPBLASLT` (including `1` or an empty string) | GPT-QModel writes nothing; rocBLAS sees the caller's value |
| Non-ROCm host | CUDA allocator default is installed; `ROCBLAS_USE_HIPBLASLT` is never written |
| BLAS call before `import gptqmodel` | rocBLAS has already fixed its kernel choice; the variable cannot change that process. Import GPT-QModel before the first BLAS call. |

## See also

- [AMD RDNA4 support](rocm-rdna4.md)
