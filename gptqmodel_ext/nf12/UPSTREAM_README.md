# nf12

NF12 is a lossless, fixed-length 12-bit encoding of BF16 weight matrices,
based on "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit Weight
Compression" (Liberman, Fang, Zhang, Dao, Fu, Together AI, 2026). This library
implements this data type efficiently on NVIDIA GPUs, from the Ampere generation
up until Blackwell.

This library's bit and memory layout (tiled) are not compatible with the one
described in the paper. You can read [the spec](docs/SPEC.md) for more info on the format.

## Requirements

- NVIDIA GPU with SM80 or newer (Ampere and later).
- CUDA toolkit 13.0+, a C++ compiler, CMake 3.26+, and Ninja for builds.
- uv for Python environment management and builds.
- Python 3.10+ and CUDA-enabled PyTorch 2.10+ for the Python package.
- Weight shapes: `N` (out_features) a multiple of 16, `K` (in_features) a
  multiple of 64.

## Install

```bash
uv pip install nf12
```

Linux x86-64 and ARM64 wheels include kernels for SM80, 86, 89, 90, 100, 110,
and 120. They bundle the CUDA 13 runtime and require a compatible NVIDIA driver.

From source:

```bash
uv sync --locked                          # all supported architectures
NF12_CUDA_ARCHS=120 uv sync --locked      # one architecture for development
```

To defer compilation until the first native operation:

```bash
NF12_BUILD_NATIVE=OFF uv sync --locked
```

You can build the C library alone with CMake:

```bash
cmake -B build/c -G Ninja -DNF12_BUILD_TORCH=OFF -DNF12_CUDA_ARCHS=120
cmake --build build/c
```

## Python

```python
import torch
import nf12

w = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
nw = nf12.encode(w)
assert torch.equal(nf12.decode(nw).view(torch.int16), w.view(torch.int16))

x = torch.randn(2, 7, 4096, dtype=torch.bfloat16, device="cuda")
y = nf12.linear(x, nw)  # preserves leading dimensions
```

Use `nf12.NF12Linear.from_linear(layer)` to convert a BF16 CUDA `nn.Linear`.
Decoding recovers the original weights. GEMM outputs can differ from cuBLAS
due to accumulation order.

### Mixture of experts

```python
w13 = nf12.encode_moe(gate_up)  # BF16 [E, 2I, K], gate rows first
w2 = nf12.encode_moe(down)      # BF16 [E, K, I]
# x: [T, K], router weights and IDs: [T, topk]
out = nf12.fused_moe(x, w13, w2, topk_weights, topk_ids)
```

### Embedding and tied heads

```python
nw = nf12.encode(model.embed_tokens.weight)  # BF16 [V, H]
x = nf12.embedding(token_ids, nw)           # [*token_ids.shape, H]

embed = nf12.NF12Embedding.from_weight(nw)
lm_head = nf12.NF12Linear.from_weight(nw)    # shares the encoded weights
```

## Benchmarks

<details>
<summary>Show benchmark tables</summary>

### Serving throughput vs BF16

NF12 / BF16 output tokens per second. Each cell shows concurrency **1 / 8 / 32**.
Above 1× favors NF12. HTTP generation, 128 input / 256 output tokens,
68 server runs with alternating BF16/NF12 order. PyTorch 2.14, CUDA 13.0.

| GPU | Qwen3-8B | Granite 1B MoE | Qwen3-30B-A3B |
|---|---|---|---|
| CMP 170HX (SM80) | 1.24× / 1.16× / 1.07× | 0.93× / 1.08× / 1.01× | NF12 only* |
| RTX A6000 (SM86) | 1.30× / 1.24× / 1.15× | 1.04× / 1.16× / 1.18× | — |
| RTX 6000 Ada (SM89) | 1.07× / 1.14× / 1.23× | 1.03× / 1.17× / 1.16× | — |
| H200 (SM90) | 1.02× / 1.01× / 0.86× | Unstable | 1.09× / 1.10× / 0.93× |
| B200 (SM100) | 1.07× / 0.95× / 0.81× | 0.95× / 0.80× / 0.78× | 0.90× / 0.83× / 0.87× |
| Jetson Thor (SM110) | 1.32× / 1.34× / 1.28× | 1.10× / 1.14× / 1.14× | NF12 only* |
| RTX PRO 6000 Blackwell (SM120) | 1.25× / 1.18× / 1.21× | 1.22× / 1.14× / 1.06× | 1.27× / 1.23× / 1.26× |

*Qwen3-30B fit only with NF12 under the tested memory caps: 75% on CMP 170HX,
40% on Thor. BF16 was not attempted there. —: excluded on the 48 GB GPUs.
H200 Granite timings were unstable, so no pooled speedup is reported.

The serving harness used a benchmark-only `nf12.fused_moe` adapter.
The baseline used native BF16 backends, including FlashInfer TRT-LLM on B200.
Prefill results are mixed. Greedy outputs can differ; these are not accuracy results.

### Loaded weight memory

| Model | BF16 GiB | NF12 GiB |
|---|---:|---:|
| Qwen3-8B | 15.27 | 11.49 |
| Granite 1B MoE | 2.50 | 1.89 |
| Qwen3-30B-A3B | 56.88 | 42.80 |

### Single-token MoE kernels

NF12 layer latency in µs (speedup over the previous composed NF12 path, **not BF16**).
Real weights, uniform routing, cold CUDA graphs rotating at least 2 GiB.
Median of two run medians. PyTorch 2.15, CUDA 13.2.

| GPU | Qwen3-30B-A3B | DeepSeek-V2-Lite |
|---|---:|---:|
| H200 | 23.2 (1.37×) | 30.5 (1.24×) |
| B200 | 18.9 (1.48×) | 22.1 (1.49×) |
| RTX PRO 6000 Blackwell | 38.9 (1.20×) | 59.6 (1.10×) |

</details>

## Reference

- [Format and API specification](docs/SPEC.md)
- [C API](include/nf12/nf12.h)
- [Benchmarks](bench/) and [tests](tests/)

## License

Apache-2.0.
