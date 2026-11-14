# Fused kernel dispatch and bias epilogues

Fused quantized kernels avoid materializing a dense weight, but their dispatch
metadata and output epilogues must follow the live checkpoint state. This page
describes the flow introduced in PR for issue #4028. It covers the GPTQ
small-M uniform-group decision, CPU BF16 fused dispatch, and FP32-output
epilogues for biased ExLlamaV2 kernels.

## Live GPTQ dispatch metadata

`TorchLinear` may replace a per-row `g_idx` gather with one scale and zero row
per K tile when `g_idx` is the uniform mapping `arange(in_features) //
group_size`. The decision is keyed by `tensor_cache_key(g_idx)`, so a state
dictionary load, device or dtype move, deepcopy, or in-place write cannot retain
the old layout flag.

```mermaid
flowchart TD
    A["TorchLinear._forward_eager"] --> B["_gptq_gemm_uniform_group_for_current_g_idx"]
    B --> C{"tensor_cache_key(g_idx) changed"}
    C -- "yes" --> D["compare g_idx with arange // group_size"]
    D --> E["store flag and current key"]
    C -- "no" --> F["reuse cached flag"]
    E --> G["gptq_gemm uniform_group"]
    F --> G
    D -- "g_idx invalid" --> H["validate_g_idx_bounds raises"]
```

The gather path remains the safe fallback. A nonuniform `g_idx` never enters
the fused uniform-group load.

## CPU BF16 and FP16 bias dispatch

BF16 has FP32's exponent range, so a BF16 fused product can safely accept a
bias in a FP32 epilogue before the visible cast. FP16 keeps the dense FP32
reduction path when a bias exists because the pre-bias product can exceed
65504.

```mermaid
flowchart TD
    A["TorchAten, TorchAtenAwq, or TorchFusedAwq forward"] --> B{"bias present"}
    B -- "no" --> C["fused int4 kernel"]
    B -- "yes" --> D{"activation dtype"}
    D -- "BF16" --> E["fused int4 kernel"]
    E --> F["out.float() + bias.float()"]
    F --> G["single cast to BF16"]
    D -- "FP16" --> H["dequantize current tile or weight"]
    H --> I["matmul_fp32_reduce with bias"]
    I --> J["single cast to FP16"]
    C --> K["visible output"]
    G --> K
    J --> K
```

`TorchFusedLinear` follows the same FP16 gate and now uses
`matmul_fp32_reduce` in its dense branch. Its BF16 fused epilogue adds the bias
in FP32 before casting.

## ExLlamaV2 FP32 output

Biased FP16 ExLlamaV2 and AwqExLlamaV2 calls keep FP16 activations on the
quantized kernel and request a transient FP32 output tensor. The extension
writes FP32 partial sums, Python adds the bias in FP32, and the result is cast
once. This avoids the previous activation upcast, which selected the dense
reconstruction path.

```mermaid
flowchart TD
    A["ExllamaV2Linear or AwqExllamaV2Linear forward"] --> B{"FP16 activation with bias"}
    B -- "yes" --> C["quantized kernel, out_dtype float32"]
    C --> D["add bias in FP32"]
    D --> E["cast once to FP16"]
    B -- "no" --> F["quantized kernel in activation dtype"]
    E --> G["adapter.apply with original-dtype x"]
    F --> G
    G --> H["visible output"]
```

The FP32 output is local to the call. No reconstructed or expanded weight is
retained on the module or native handle.

AwqExLlamaV2 passes the original-dtype activation to `Lora.apply` after the
final cast. LoRA weights therefore remain in the caller's dtype and the output
does not become FP32.

## Failure and ambiguity contract

| Situation | Before | After |
| --- | --- | --- |
| `g_idx` changes after `post_init` | cached uniform flag stayed true; M=2..64 fused GEMM used the wrong scale row | keyed lookup recomputes the flag before dispatch |
| FP16 biased `TorchFusedLinear` | fused or dense product rounded to `inf` before bias | dense FP32 reduction plus bias, then one cast |
| BF16 biased CPU fused layer | forced to dense dequantization on every forward | fused kernel plus FP32 bias epilogue |
| FP16 biased ExLlamaV2 | activations upcast to FP32, selecting dense reconstruction | FP16 activation, FP32 output from the quantized kernel |
| AwqExLlamaV2 with LoRA | adapter received upcast FP32 activations and upcast LoRA weights | adapter receives original-dtype activations after the output cast |

## Scope

The GPTQ ExLlamaV2 FP32-output kernel and the previous dense reconstruction use
different reduction orders. They agree within the inference tolerance but are
not guaranteed bitwise identical. AwqExLlamaV2 matched the previous wide path
bitwise in the measured K=N=4096 fixture. The CPU fused BF16 path also uses a
different reduction order from the dense fallback; its output is checked
against an independent FP64 oracle.

BitsAndBytes INT8 uses its native fused `int8_mm_dequant` epilogue again when
FP16 activations have no bias, where its FP16 rounding cannot destroy a later
cancellation. BF16 and biased calls keep the FP32 Python epilogue until a
fused quant-plus-epilogue kernel is available.
