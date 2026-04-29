# GPTQModel Ultra Goals

GPTQModel Ultra is a highly hardware/software optimized edition of upstream GPTQModel. The goal is to keep upstream compatibility while pushing harder on quantization speed, quantization quality, runtime efficiency, and hardware-tuned kernels.

## Goals

| Goal | Description |
|---|---|
| Faster quantization | Reduce end-to-end quantization time through better execution planning, parallelism, memory movement, and processor-specific optimizations. |
| Better quantization results | Improve model quality after quantization through more capable calibration paths, better module coverage, and targeted requantization options. |
| Faster hardware-optimized kernels | Add and refine kernels tuned for modern GPU, CPU, and accelerator backends. |
| Broader module coverage | Extend quantization beyond standard transformer linear layers where it improves real model deployment, including embeddings and output heads. |
| Practical model compatibility | Preserve compatibility with upstream GPTQModel workflows, Hugging Face model loading, saving, and inference paths. |
| Production-oriented save/load | Make optimized and requantized models reload cleanly without losing specialized quantized modules. |
| Hardware-aware execution | Use hardware topology, device placement, and memory strategy to make quantization and inference faster on real systems. |

## Focus Areas

| Focus area | Ultra emphasis |
|---|---|
| Quantization pipeline | Faster calibration, forward replay, Hessian collection, packing, and finalization. |
| Quantization quality | Better handling of difficult modules, embedding/output-head coverage, and targeted requantization. |
| Kernel performance | More aggressive use of optimized kernels for supported hardware targets. |
| Memory behavior | Lower CPU/GPU memory pressure during quantization and save/load operations. |
| Hardware portability | Keep fast paths available across CUDA, CPU, and other supported accelerator backends when feasible. |
| Upstream sync | Continue merging upstream GPTQModel improvements while preserving Ultra-specific optimizations. |

## Non-Goals

| Non-goal | Reason |
|---|---|
| Breaking upstream APIs unnecessarily | Ultra should remain easy to adopt for existing GPTQModel users. |
| Optimization without measurable impact | Ultra changes should target speed, memory, quality, compatibility, or kernel performance. |
| Hardware-specific code that blocks general use | Specialized kernels should coexist with safe fallback paths where practical. |

