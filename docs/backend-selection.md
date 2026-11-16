<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Backend selection and configuration

GPT-QModel accepts one execution argument: `backend=BACKEND.MARLIN` selects
Marlin with defaults; `backend=MarlinConfig(fp32=False)` selects Marlin with
custom settings. Existing enum members, strings and legacy aliases retain
their selection semantics. Backend compatibility checks still apply to the
checkpoint's method, format, device, dtype and shape.

```python
from gptqmodel import GPTQModel
from gptqmodel.backend import BACKEND, MarlinConfig, RuntimeConfig
from gptqmodel.quantization import QuantizeConfig

model = GPTQModel.load("org/model", backend=BACKEND.MARLIN)
model = GPTQModel.load("org/model", backend=MarlinConfig(fp32=False))

# Shared run controls live separately from backend-specific execution options.
qcfg = QuantizeConfig(bits=4, runtime=RuntimeConfig(max_packers=2))
model = GPTQModel.load("org/unquantized-model", quantize_config=qcfg,
                      backend=MarlinConfig(fp32=False))
# model.quantize(calibration_dataset) retains that typed backend selection.
```

`GPTQModel.from_pretrained`, `GPTQModel.from_quantized`, and `model.quantize`
also accept typed backend configurations. A quantization-time `backend=`
overrides the selection retained by the loader. Use `backend=BACKEND.AUTO`
without a retained typed selection to run the existing automatic chooser.

## Types and options

`BackendConfig` is abstract. Concrete configurations inherit it, declare a
fixed `BACKEND` identity, and validate their options at construction. They are
frozen dataclasses; their backend cannot be supplied as a constructor option.
`MarlinConfig()` has the same execution defaults as `BACKEND.MARLIN`.

Import configurations from `gptqmodel.backend` or `gptqmodel`.

| Configuration | Selects | Options (unprefixed within the configuration) |
| --- | --- | --- |
| `MarlinConfig` | `MARLIN` | `fp32` |
| `TorchConfig` | `GPTQ_TORCH` | `streaming`, `stream_tile`, `triton_dequant`, `rocm_gemv`, `rocm_gemm` |
| `AwqTorchConfig` | `AWQ_TORCH` | Default selection |
| `AwqGemmConfig` | `AWQ_GEMM` | `fp32_accum`, `fused_splitk_reduce` |
| `AwqTritonConfig` | `AWQ_GEMM_TRITON` | `fp32_accum`, `schedule` |
| `SwordfishConfig` | `SWORDFISH` | `dense_m`, `deterministic` |
| `GGUFTorchConfig` | `GGUF_TORCH` | Fused CPU/CUDA row and matrix-size gates, chunk sizes and autotuning settings |
| `GGUFCPPConfig` | `GGUF_CPP_CPU` | GGUF options plus `native_matmul`, `threads` |
| `GGUFCUDAConfig` | `GGUF_CPP_CUDA` | GGUF options |
| `ParoQuantConfig` | `PAROQUANT` | `fp32_accum`, `fused_splitk_reduce`, runtime/rotation dtype caching, `cpu_avx512` |
| `ParoQuantTritonConfig` | `PAROQUANT_TRITON` | ParoQuant options plus autotuning and `decode_max_rows` |
| `BitsAndBytesConfig` | `BITSANDBYTES` | `cpu_packed`, `cpu_int8_dense` |
| `TTBlockFloatConfig` | `TT_BLOCKFLOAT` | `cublas` |

Backends without typed tuning options continue to use `BACKEND` selectors.
Shared quantizer, worker and host controls use `RuntimeConfig`, accepted by
the loaders as `runtime=` or attached to `QuantizeConfig.runtime`. The
[legacy field reference](kernel-config.md#fields) gives the defaults and
units; typed execution options omit the backend prefix (for example
`TorchConfig(stream_tile=256)` maps to `KernelConfig.torch_stream_tile`).
Shared quantizer options keep their names, such as `gptq_fused_block_update`.

## Resolution and ownership

```mermaid
flowchart TD
    A["GPTQModel.load / from_pretrained / from_quantized<br/>BaseQModel.quantize"] --> B["resolve_backend_config(backend, kernel, runtime)"]
    A -- "quantization config present" --> R["resolve_quantize_backend<br/>retain typed choice and generated policy"]
    R --> B
    B --> C{"BackendConfig instance?"}
    C -- yes --> D{"Legacy kernel supplied?"}
    D -- yes --> E["ValueError: ambiguous policy"]
    D -- no --> F["backend.backend + backend.kernel_options()<br/>merge RuntimeConfig.kernel_options()"]
    C -- no --> G["normalize_backend(enum or string)<br/>legacy kernel or RuntimeConfig adapter"]
    B -- "runtime + legacy kernel" --> E
    F --> H["BACKEND + validated KernelConfig"]
    G --> H
    H --> I["Existing backend compatibility / selection"]
    I -- "unsupported method / device / shape" --> J["Existing selection error"]
    I --> K["QuantizeConfig.kernel<br/>make_quant / quant_linear_init_kwargs"]
    K --> L["BaseQuantLinear.__init__<br/>snapshot_kernel_config"]
    L --> M["module.kernel<br/>constructor settings and native op arguments"]
    H --> N["configure_kernel<br/>shared quantizer / host fallback"]
    classDef failure fill:#ffe0e0,stroke:#b00,color:#600
    class E,J failure
```

`resolve_kernel_config` accepts a `KernelConfig` directly, then prefers an
owner's `kernel`, then its `runtime`, then the process fallback. Each quantized
linear owns a fresh copy. Loading another model or mutating the legacy policy
after construction does not retune existing modules. Native AWQ, Swordfish,
block-float and GGUF CPU calls receive the relevant module settings explicitly.
GGUF's shared native CPU backend sets the selected module's thread count under
its existing execution lock.

Shared quantization/host utilities still use a process-wide fallback guarded
by an `RLock`. Concurrent model quantization with different shared runtime
settings is not isolated by this API. `configure_kernel(None)` resets that
fallback to defaults; it does not modify constructed modules.

## Before and after

Locations in the second column refer to the code before this API change.

| Mechanism | Before (`gptqmodel/` location) | Current behavior |
| --- | --- | --- |
| Execution choice | `models/auto.py:872,1005,1129`: normalize enum/string | `resolve_backend_config` additionally accepts a `BackendConfig` and extracts its fixed identity |
| Execution tuning | `models/auto.py:839`: `kernel=` installs a mixed policy | A typed `backend=` selects and tunes; `runtime=` carries shared controls; legacy `kernel=` remains accepted |
| Module policy | `nn_modules/qlinear/__init__.py:288`: stores only the selector; `utils/model.py:623`: forwards qcfg kwargs | `BaseQuantLinear` snapshots an explicit policy, also forwarded by `make_quant` |
| Runtime fallback | `utils/kernel.py:404-414`: owner policy or current global | Direct policy objects are honored; modules keep snapshots; shared utilities retain the global fallback |

```mermaid
flowchart LR
    A["Before: backend enum/string"] --> B["normalize_backend"]
    C["Before: kernel=KernelConfig"] --> D["configure_kernel / optional owner.kernel"]
    B --> E["module backend selection"]
    D --> F["constructor or forward reads active policy"]
    G["Later load replaces active policy"] --> F
```

## Failure and compatibility contract

```mermaid
flowchart LR
    A["Concrete BackendConfig or RuntimeConfig"] --> B["KernelConfig validation"]
    B -- "valid" --> C["resolve_backend_config"]
    B -- "wrong type / range / unknown field" --> D["TypeError / ValueError"]
    C -- "typed backend + legacy kernel<br/>or runtime + legacy kernel" --> E["ValueError before loading"]
    C -- "valid selector and policy" --> F["existing backend validation"]
    F -- "incompatible checkpoint or hardware" --> D
    classDef failure fill:#ffe0e0,stroke:#b00,color:#600
    class D,E failure
```

Existing `kernel=KernelConfig(...)` and `kernel={...}` remain supported with
enum/string selectors. A typed backend cannot be combined with legacy
`kernel=`, including one on `QuantizeConfig`; use `runtime=` for shared controls.
Configurations and effective policies are runtime-only and are not written to
checkpoint metadata. Checkpoints continue to describe their numerical format;
callers choose execution settings on reload.
