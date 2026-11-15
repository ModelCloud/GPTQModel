# Quantizer calibration flows: MoE native inputs, GGUF quality aliases, Hessian staging

This page describes three calibration decisions that share one rule: a tensor
must only be combined with data that describes the same tokens, and a name must
not promise a layout the packer does not write.

Introduced by PR for issue #4067.

## 1. GPTAQ/FOEM routed-expert inputs

`NativeProcessor` records every module input during one full-precision forward.
`GPTAQ` and `FOEM` (alpha > 0) later compare those native inputs with the inputs
seen during the quantized replay to build `dXXT = (X_native - X_quant)^T X_quant`.
For dense modules the two passes call the module with the same rows, so pairing
the recorded entries by call order is valid.

For routed experts it is not. Attention in the same layer is already quantized
when the experts replay, so the router sends a different token subset to each
expert. The old code popped one native entry per expert call:

```mermaid
flowchart TD
    A["full-precision forward"] --> B["NativeProcessor records expert inputs per sample"]
    C["quantized replay of the same layer"] --> D["attention is already quantized"]
    D --> E["router selects a different token subset"]
    E --> F["GPTAQ/FOEM pop native entry by call order"]
    F --> G{"row counts match"}
    G -- no --> H["RuntimeError in dXXT matmul"]
    G -- yes --> I["rows from different tokens"]
    I --> J["silent dXXT corruption"]
    H --> K["quantization aborts"]
    style H fill:#f8d7da,stroke:#b02a37,color:#842029
    style J fill:#f8d7da,stroke:#b02a37,color:#842029
    style K fill:#f8d7da,stroke:#b02a37,color:#842029
```

An expert that receives no routed rows has `H = None`. `GPTAQ.quantize` and
`FOEM.quantize` then dereferenced `self.H.device` and never reached GPTQ's
data-independent fallback.

The fix marks routed experts from the model's `ModulePlan` in
`ModuleLooper.create_named_modules` (`plan.is_expert(name)`). `GPTAQ` and `FOEM`
read that mark and disable the native delta for routed experts: `dXXT = 0`, so
GPTAQ runs the plain GPTQ update and FOEM keeps its beta term. Shared experts
and dense projections still use the native delta because they receive the same
rows in both passes. A zero-sample module now takes the configured fallback
strategy (RTN by default) in both quantizers.

```mermaid
flowchart TD
    A["create_named_modules"] --> B{"ModulePlan.is_expert(name)"}
    B -- yes --> C["NamedModule state is_routed_expert = True"]
    B -- no --> D["dense or shared expert"]
    C --> E["GPTAQ: use_native_delta = False"]
    C --> F["FOEM: gptaq = False, beta retained"]
    E --> G["H from quantized-path rows only"]
    F --> G
    G --> H{"nsamples == 0"}
    H -- yes --> I["RTN fallback"]
    H -- no --> J["GPTQ-style update with dXXT = 0"]
    D --> K["native delta retained"]
    style I fill:#d1e7dd,stroke:#0f5132,color:#0f5132
    style J fill:#d1e7dd,stroke:#0f5132,color:#0f5132
```

| Situation | Before | After |
| --- | --- | --- |
| Routed expert, row counts differ | `RuntimeError` in `gptaq.py` / `foem.py` | `dXXT = 0`; plain GPTQ update |
| Routed expert, row counts happen to match | silent `dXXT` corruption | `dXXT = 0`; no native pairing |
| Routed expert, zero calibration rows | `AttributeError` on `self.H.device` | RTN fallback |
| Dense projection | native delta | unchanged |
| Shared expert | native delta | unchanged |

The alpha correction is intentionally unavailable for routed experts until the
native rows can be gathered with the quantized-path routing. That follow-up is
not implemented here; routed experts are quantized as plain GPTQ (GPTAQ) or
FOEM without the alpha term.

## 2. GGUF `q*_k_s` / `q*_k_m` / `q*_k_l` quality aliases

llama.cpp's `_s`/`_m`/`_l` suffixes select a per-tensor mixture, for example
Q6_K for `attn_v` and `ffn_down` in the `use_more_bits` layers. GPT-QModel
stores every tensor with the base K type (`Q4_K` for `q4_k_*`, and so on), so
the alias name overstated the checkpoint quality.

```mermaid
flowchart TD
    A["GGUFConfig bits = q4_k_m"] --> B["_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE"]
    B --> C["uniform Q4_K for every tensor"]
    C --> D["saved checkpoint name says Q4_K_M"]
    D --> E["user expects the llama.cpp mixture"]
    style E fill:#f8d7da,stroke:#b02a37,color:#842029
```

The fix keeps the uniform storage (the mixture is not implemented) but emits a
loud warning from `GGUFConfig.__post_init__` and from dynamic per-layer
overrides. The bpw table now reports `q5_k_s` as 5.5 bpw, matching the uniform
Q5_K blocks that are actually written.

| Alias | Stored type | Reported bpw |
| --- | --- | --- |
| `q4_k_s`, `q4_k_m` | `Q4_K` | 4.5 |
| `q5_k_s`, `q5_k_m` | `Q5_K` | 5.5 |
| `q3_k_s`, `q3_k_m`, `q3_k_l` | `Q3_K` | 3.4375 |

## 3. Hessian staging dtype range

`HessianConfig.staging_dtype` may request float16. `preferred_staging_dtype`
returned float16 for bfloat16 inputs, so an activation outside float16's range
became `inf` before the fp32 matmul, and the Hessian became `inf`/`NaN`.

```mermaid
flowchart TD
    A["bf16 activation with a channel at 1e5"] --> B["preferred_staging_dtype returns float16"]
    B --> C["staging copy overflows to inf"]
    C --> D["fp32 Hessian is inf/NaN"]
    style D fill:#f8d7da,stroke:#b02a37,color:#842029
```

The staging dtype is now never narrower in range than the input: a bf16 input
with `staging_dtype=float16` stages as bf16 (or fp32 when the device has no
bf16 matmul). An fp16 input may still stage as fp16.

```mermaid
flowchart TD
    A["preferred_staging_dtype input_dtype, device"] --> B{"staging_dtype == float16"}
    B -- no --> C["existing float32 / bfloat16 path"]
    B -- yes --> D{"input_dtype == bfloat16"}
    D -- yes --> E["bfloat16 if supported, else float32"]
    D -- no --> F["float16"]
    style E fill:#d1e7dd,stroke:#0f5132,color:#0f5132
```

Measured on CPU with a bf16 input whose channel 3 is scaled by `1e5` and
`chunk_size=64`: the old float16 staging produced a non-finite Hessian; the
fixed path matches the float64 oracle within `1e-6` normalized error.
