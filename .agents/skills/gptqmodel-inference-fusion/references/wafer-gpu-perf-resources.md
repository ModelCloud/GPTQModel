# Curated GPU performance engineering resources (inference engines)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## Minimum mental model for serving

- [How to Scale Your Model: Inference](https://jax-ml.github.io/scaling-book/inference/) — One request from prefill through decode, with batching, KV memory, and parallelism.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) — FLOPs, parameter bytes, KV bytes, and communication.
- [Efficiently Scaling Transformer Inference](https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf) — Latency, memory, and parallelism costs.
- [Etalon](https://arxiv.org/html/2407.07000) — TTFT, TPOT, goodput, and latency SLOs.

## Scheduling and continuous batching

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) — Iteration-level scheduling for autoregressive serving.
- [PagedAttention and vLLM](https://arxiv.org/html/2309.06180) — Paged KV allocation and continuous batching.
- [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) — Chunked prefills that reduce interference with decode.
- [SGLang](https://arxiv.org/html/2312.07104) — Prefix reuse, structured programs, and a serving runtime.
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — Main production engine implementations.

## KV cache systems

- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) — Fewer key-value heads and a smaller KV cache.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — Multi-head latent attention and compressed KV state.
- [KIVI](https://proceedings.mlr.press/v235/liu24bz.html) — KV quantization with separate treatment for keys and values.
- [CacheGen](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf) — KV compression for transfer.
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) — A distributed KV cache and data plane.

## Quantization

- [GPTQ](https://arxiv.org/abs/2210.17323) — One-shot second-order weight quantization.
- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) — W8A8 execution by moving quantization difficulty from activations into weights.
- [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) — Low-bit weight-only inference with salient-weight protection.

## Speculative decoding

- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa](https://arxiv.org/html/2401.10774)
- [EAGLE](https://proceedings.mlr.press/v235/li24bt.html) — Feature-level drafting.

## Structured, long-context, and multimodal inference

- [XGrammar](https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf) — Fast grammar engine for structured generation.
- [Guiding LLMs the Right Way](https://proceedings.mlr.press/v235/beurer-kellner24a.html) — Constrained decoding without changing token distribution.
- [Ring Attention](https://arxiv.org/abs/2310.01889) — Exact distributed attention by circulating KV blocks.
- [MInference 1.0](https://arxiv.org/abs/2407.02490) — Dynamic sparse patterns for long-context prefill.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) — Hardware-aligned sparse attention hierarchy.
- [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html)
