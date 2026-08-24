# Curated GPU performance engineering resources (quantization and low precision)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## Quantization methods

- [GPTQ](https://arxiv.org/abs/2210.17323) — One-shot second-order weight quantization.
- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) — W8A8 execution by moving quantization difficulty from activations into weights.
- [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) — Low-bit weight-only inference with salient-weight protection.

## Low-precision formats and execution

- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf) — E4M3 and E5M2 formats.
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — Shared-scale MX formats.
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — FP8 and FP4 transformer execution with scaling controls.
- [Blackwell matrix multiply instructions](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html) — `tcgen05`, tensor memory, and Blackwell MMA programming.

## KV-cache compression

- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) — Fewer key-value heads and a smaller KV cache.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — Multi-head latent attention and compressed KV state.
- [KIVI](https://proceedings.mlr.press/v235/liu24bz.html) — KV quantization with separate treatment for keys and values.
- [CacheGen](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf) — KV compression for transfer.

## Inference context

- [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) — FLOPs, parameter bytes, KV bytes, and communication.
- [Efficiently Scaling Transformer Inference](https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf) — Latency, memory, and parallelism costs.
