# Curated GPU performance engineering resources (MoE serving)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## MoE serving

- [DeepSeek-V3](https://arxiv.org/html/2412.19437) — Routed experts, shared experts, and the model-system design.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) — Expert dispatch and combine kernels.
- [EPLB](https://github.com/deepseek-ai/EPLB) — Expert placement and replication from measured load.
- [MegaScale-Infer](https://arxiv.org/abs/2504.02263) — Large-scale MoE inference and communication overlap.

## Related inference-system context

- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) — Smaller KV cache.
- [PagedAttention and vLLM](https://arxiv.org/html/2309.06180) — Paged KV allocation and continuous batching.
- [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) — Chunked prefills.
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) — Distributed KV cache and data plane.
- [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/)
