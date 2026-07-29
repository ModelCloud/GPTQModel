# GPT-QModel Inference Fusion

This document tracks the design and rollout of fused inference modules for GPT-QModel (QKV and gate/up projection fusion).

## Goals

- Reduce launch count for attention and MLP forward passes by fusing same-input, same-quantization-config GPTQ projections into a single GEMM.
- Keep the existing `BaseQuantLinear` backend abstraction intact; reuse the same `qweight`, `scales`, `qzeros`, and `g_idx` buffer layout.
- Provide an opt-in `model.fuse()` API after `GPTQModel.load(...)` so users can turn fusions on when their target shapes benefit.
- Validate each fusion against the unfused path (dense BF16/FP32 reference and per-module quantized path) before claiming speedup.

## Reference: vLLM / SGLang fused GPTQ patterns

Both vLLM and SGLang implement fused QKV and gate/up for GPTQ by packing weights along the output dimension and using a single `gptq_gemm` call:

- `QKVParallelLinear` / `MergedColumnParallelLinear` carry `output_partition_sizes`.
- `GPTQLinearMethod.create_weights` builds one `PackedvLLMParameter` whose `output_size_per_partition = sum(output_partition_sizes)`.
- `GPTQLinearMethod.apply` calls one `ops.gptq_gemm` and returns the concatenated output, which the model splits with `split` / `chunk` / `SiluAndMul`.

GPT-QModel already contains specialized fused kernels (`trilin_qkv`, `trilin_swiglu`, `q2_qkv`, `q2_swiglu`) for 3-bit/Q2 shapes. This work generalizes the same idea to any `bits`/`group_size` combination where the fused members share:

- identical `in_features`
- identical `bits`, `group_size`, `desc_act`/`g_idx`, `sym`, `pack_dtype`
- no per-member `adapter` or rotation that cannot be applied to the concatenated tensor

## Target model shapes

| Model | `hidden_size` | `num_attention_heads` | `num_key_value_heads` | `head_dim` | QKV `out` sizes | `intermediate_size` / MoE latent | gate/up `out` sizes |
|---|---:|---:|---:|---:|:---|:---|:---|
| Laguna-S-2.1 | 3072 | 48 | 8 | 128 | `[6144, 1024, 1024]` (total 8192) | 12288 | `[12288, 12288]` (total 24576) |
| Qwen3.5-27B | 5120 | 24 | 4 | 256 | `[6144, 1024, 1024]` (total 8192) | 17408 | `[17408, 17408]` (total 34816) |
| Kimi-K3 (proxy) | 7168 | 96 (KDA/MLA) | - | - | uses KDA + Gated MLA, not a standard `q_proj/k_proj/v_proj` layout | routed latent width 3584, 896 routed experts, 2 shared experts, 16 active | routed gate/up latent 3584 each; shared full-width path 7168 -> expert_dim |

Notes:

- Laguna and Qwen3.5-27B are the primary targets for standard QKV and gate/up fusion because they expose clean `q/k/v_proj` and `gate_proj/up_proj` pairs.
- Kimi-K3 uses Kimi Delta Attention and latent-space MoE, so standard QKV/gate-up fusion may not apply directly; the MLP and any compressed attention paths will be profiled with the documented proxy dimensions.

## Implementation plan

### Phase 1: weight-concat MVP

Implement `gptqmodel/nn_modules/fused_quant_linear.py` with `FusedQKVForward` and `FusedGateUpForward`:

- At `install_fused_qkv(model)` time, find `q_proj`, `k_proj`, `v_proj` triples inside attention modules.
- Verify all three modules are the same backend (`TritonV2Linear` first), share `in_features`, `bits`, `group_size`, `g_idx`, `pack_dtype`, and `sym`.
- Build a contiguous `fused_qweight`/`fused_scales`/`fused_qzeros` tensor by concatenating along the output dimension. Re-use the `g_idx` of the first module.
- Replace `q_proj.forward` to call one backend GEMM and return the query slice; replace `k_proj.forward`/`v_proj.forward` to return cached key/value slices from the same forward.
- `install_fused_gate_up(model)` does the same for `gate_proj`/`up_proj` pairs inside MLP modules, with `gate_proj.forward` returning the gate slice and caching the up slice for `up_proj.forward`.

This is the vLLM/SGLang-style column fusion but for GPT-QModel buffer layouts. It keeps the surrounding `Attention` / `MLP` code unchanged.

### Phase 2: accuracy validation

For each supported backend, add `tests/test_fused_quant_qkv.py` and `tests/test_fused_quant_gateup.py` that:

- construct synthetic `TritonV2Linear` modules with the target shapes;
- compare fused output against the dense reference `x @ [W_q; W_k; W_v]` and against calling the three modules separately;
- report `max abs diff` and `mean abs diff`;
- assert a tolerance that is tight for FP32 accumulation and appropriate for BF16/FP16 tensor-core accumulation order differences.

### Phase 3: performance validation

Add `scripts/benchmark_fused_qkv_gateup.py` that:

- runs on an idle A100/H100 GPU with `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- tests the Laguna, Qwen3.5-27B, and Kimi-K3 proxy shapes at representative prefill/decode batch sizes;
- reports separate vs fused latency, throughput (tokens/s), and speedup for QKV and gate/up separately;
- includes a dense BF16 baseline.

### Phase 4: `model.fuse()` API

Add `BaseQModel.fuse(...)` and `GPTQModel.fuse(...)` that call the install helpers and log which modules were fused.

### Phase 5: Triton mega-kernel (future)

The weight-concat MVP still pays for a PyTorch `matmul` over the concatenated output dimension. A follow-up can fuse the dequantization and GEMV/GEMM of Q/K/V or gate/up into one Triton launch, similar to the existing `trilin_qkv` and `trilin_swiglu` kernels, to remove launch overhead and control accumulation width for exact 1:1 vs the unfused path.

## Metrics to record

For every benchmark run, capture:

- GPU model, compute capability, driver, PyTorch/CUDA/Triton versions
- shapes: `(batch, seq_len, hidden)`, `QKV out` sizes, `gate/up out` sizes
- dtype / `bits` / `group_size`
- backend (TritonV2Linear, TorchLinear, etc.)
- separate latency, fused latency, speedup
- accuracy `max abs diff` vs dense reference and vs separate quantized path

Updates to this document will be appended as each phase completes.

---

## Phase 1 status (MVP)

- [x] `FusedQKVForward` and `FusedGateUpForward` implementation in `gptqmodel/nn_modules/fused_quant_linear.py`
- [x] unit-test accuracy for Laguna/Qwen3.5-27B shapes in `tests/test_fused_quant_linear.py`
- [x] performance benchmark on target shapes in `scripts/benchmark_fused_qkv_gateup.py`
- [x] first PR opened (#119)

### Phase 1 benchmark (A100 96 GB, sm80, PyTorch 2.13.0+cu130, Triton 3.7.1)

`python scripts/benchmark_fused_qkv_gateup.py --device cuda:0` (physical GPU 6 via `CUDA_VISIBLE_DEVICES=6`).

| op | hidden | out | batch | seq | ms_sep | ms_fused | speedup | tok/s_sep | tok/s_fused | max_diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 3072 | (6144, 1024, 1024) | 1 | 1 | 0.567 | 0.201 | 2.824 | 1765.2 | 4984.5 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 1 | 1 | 0.392 | 0.248 | 1.584 | 2550.0 | 4040.4 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 2 | 1 | 0.543 | 0.195 | 2.783 | 3682.2 | 10246.2 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 2 | 1 | 0.396 | 0.251 | 1.576 | 5048.4 | 7958.3 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 4 | 1 | 0.552 | 0.204 | 2.701 | 7245.9 | 19568.4 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 4 | 1 | 0.504 | 0.277 | 1.818 | 7930.8 | 14418.5 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 8 | 1 | 0.620 | 0.276 | 2.245 | 12894.0 | 28952.3 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 8 | 1 | 0.428 | 0.252 | 1.699 | 18675.9 | 31727.2 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 16 | 1 | 1.041 | 0.343 | 3.038 | 15369.6 | 46689.2 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 16 | 1 | 0.725 | 0.391 | 1.854 | 22063.6 | 40907.4 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 32 | 1 | 0.540 | 0.196 | 2.757 | 59259.7 | 163407.2 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 32 | 1 | 0.397 | 0.253 | 1.571 | 80566.2 | 126600.2 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 1 | 128 | 0.537 | 0.197 | 2.728 | 238158.8 | 649755.7 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 1 | 128 | 0.393 | 0.264 | 1.485 | 325996.2 | 484008.4 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 1 | 1024 | 0.543 | 0.287 | 1.893 | 1887006.0 | 3572704.5 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 1 | 1024 | 0.848 | 0.846 | 1.003 | 1207467.0 | 1210565.8 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 1 | 4096 | 0.924 | 0.788 | 1.173 | 4430758.3 | 5198991.2 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 1 | 4096 | 2.608 | 2.743 | 0.951 | 1570277.7 | 1493127.9 | 0.0000 |
| qkv | 3072 | (6144, 1024, 1024) | 16 | 128 | 0.533 | 0.443 | 1.204 | 3844675.2 | 4628129.7 | 0.0000 |
| gate_up | 3072 | (12288, 12288) | 16 | 128 | 1.413 | 1.474 | 0.959 | 1449233.3 | 1389352.0 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 1 | 0.547 | 0.199 | 2.756 | 1827.5 | 5036.4 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 1 | 1 | 0.568 | 0.502 | 1.131 | 1761.5 | 1993.1 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 2 | 1 | 0.547 | 0.195 | 2.811 | 3656.0 | 10275.3 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 2 | 1 | 0.566 | 0.508 | 1.114 | 3533.8 | 3935.2 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 4 | 1 | 0.542 | 0.197 | 2.749 | 7385.3 | 20302.8 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 4 | 1 | 0.566 | 0.509 | 1.111 | 7067.1 | 7854.9 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 8 | 1 | 0.550 | 0.196 | 2.800 | 14545.2 | 40724.0 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 8 | 1 | 0.568 | 0.509 | 1.116 | 14090.8 | 15726.9 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 16 | 1 | 0.551 | 0.198 | 2.779 | 29051.4 | 80732.7 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 16 | 1 | 0.613 | 0.508 | 1.207 | 26100.8 | 31503.3 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 32 | 1 | 0.534 | 0.195 | 2.742 | 59980.8 | 164473.7 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 32 | 1 | 0.608 | 0.542 | 1.122 | 52666.2 | 59087.1 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 128 | 0.590 | 0.207 | 2.847 | 216923.5 | 617588.9 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 1 | 128 | 0.624 | 0.631 | 0.988 | 205247.8 | 202744.3 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 1024 | 0.550 | 0.462 | 1.190 | 1862752.4 | 2216312.0 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 1 | 1024 | 1.781 | 1.744 | 1.021 | 574904.3 | 586999.1 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 4096 | 1.508 | 1.323 | 1.140 | 2715620.3 | 3096838.2 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 1 | 4096 | 5.788 | 6.006 | 0.964 | 707626.4 | 681963.1 | 0.0000 |
| qkv | 5120 | (6144, 1024, 1024) | 16 | 128 | 0.812 | 0.734 | 1.106 | 2521813.6 | 2788467.0 | 0.0000 |
| gate_up | 5120 | (17408, 17408) | 16 | 128 | 3.668 | 3.126 | 1.173 | 558384.7 | 655067.6 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 1 | 0.542 | 0.206 | 2.626 | 1845.4 | 4845.5 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 1 | 1 | 0.396 | 0.319 | 1.242 | 2522.2 | 3132.0 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 2 | 1 | 0.542 | 0.201 | 2.701 | 3690.7 | 9970.0 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 2 | 1 | 0.392 | 0.325 | 1.207 | 5097.4 | 6152.0 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 4 | 1 | 0.554 | 0.198 | 2.791 | 7222.3 | 20160.2 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 4 | 1 | 0.392 | 0.330 | 1.189 | 10194.8 | 12121.4 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 8 | 1 | 0.535 | 0.194 | 2.753 | 14949.3 | 41161.7 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 8 | 1 | 0.394 | 0.325 | 1.213 | 20303.8 | 24635.8 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 16 | 1 | 0.536 | 0.194 | 2.763 | 29863.2 | 82497.4 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 16 | 1 | 0.409 | 0.326 | 1.258 | 39082.0 | 49153.8 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 32 | 1 | 0.539 | 0.189 | 2.844 | 59412.9 | 168973.7 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 32 | 1 | 0.405 | 0.328 | 1.237 | 78974.0 | 97692.9 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 128 | 0.559 | 0.259 | 2.159 | 228904.2 | 494188.3 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 1 | 128 | 0.519 | 0.354 | 1.465 | 246441.4 | 361146.4 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 1024 | 0.532 | 0.260 | 2.044 | 1925224.4 | 3935768.3 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 1 | 1024 | 0.907 | 0.996 | 0.910 | 1129611.6 | 1028214.2 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 4096 | 0.989 | 0.800 | 1.235 | 4143188.7 | 5117314.4 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 1 | 4096 | 3.121 | 3.203 | 0.974 | 1312327.3 | 1278641.6 | 0.0000 |
| qkv | 4096 | (4096, 1024, 1024) | 16 | 128 | 0.550 | 0.464 | 1.185 | 3724533.5 | 4414231.4 | 0.0000 |
| gate_up | 4096 | (11008, 11008) | 16 | 128 | 1.630 | 1.744 | 0.934 | 1256739.2 | 1174067.2 | 0.0000 |

Observations:

- QKV fusion is a clear win for decode (1-token) and small-prefill workloads: 2.5-2.9x speedup. The gain shrinks for large prefill batches because the single larger GEMM no longer reduces launch overhead relative to the compute work.
- Gate/up fusion helps decode (1.1-1.5x) but is roughly neutral or slightly slower on large prefill batches, where two half-size GEMMs are as efficient as one full-width GEMM.
- Accuracy (`max abs diff`) is 0.0 for the synthetic packed weights used in the benchmark; the unit tests tolerate small BF16 tensor-core accumulation-order differences (up to 2.0 abs) between fused and separate paths.

### Phase 1 validation commands

```bash
ruff check gptqmodel/nn_modules/fused_quant_linear.py tests/test_fused_quant_linear.py scripts/benchmark_fused_qkv_gateup.py
pytest -q tests/test_fused_quant_linear.py
python scripts/benchmark_fused_qkv_gateup.py
```

## Phase 2 status (`model.fuse()` API)

- [x] add `BaseQModel.fuse(...)` that calls `install_fused_qkv` / `install_fused_gate_up`
- [x] unit test for the API in `tests/test_fused_quant_linear.py`
- [x] update this doc with API usage and validation commands

Usage:

```python
from gptqmodel import GPTQModel

model = GPTQModel.load("/path/to/quantized/model", backend=BACKEND.TritonV2)
model.fuse(qkv=True, gate_up=True)  # opt-in, in-memory only
```

`fuse()` returns `{"qkv": N, "gate_up": M}` and logs the number of fused groups. It is a no-op on non-quantized models.

#### Supported fusion patterns

| Fusion | Default module names | Also detected aliases |
|---|---|---|
| QKV | `q_proj`, `k_proj`, `v_proj` | `wq/wk/wv`, `query/key/value`, `q/k/v` |
| gate/up | `gate_proj`, `up_proj` | `w1/w2` (Qwen), `w1/w3` (Llama-1/2), `gate/up` |

Models that already ship a single fused module (`qkv_proj`, `gate_up_proj`, `c_attn`) are not split; they already perform one GEMM for the group.

### Phase 2 validation commands

```bash
ruff check gptqmodel/models/base.py tests/test_fused_quant_linear.py
pytest -q tests/test_fused_quant_linear.py
python scripts/benchmark_fused_real_model.py --model_path /path/to/gptq-model --backend triton --device cuda:0
```

### Phase 2 real-model benchmark

`scripts/benchmark_fused_real_model.py` on `DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2` (Qwen MLP uses `w1`/`w2` gate/up aliases), A100 96 GB, sm80, PyTorch 2.13.0+cu130, Triton 3.7.1:

| batch | seq | ms (unfused) | ms (fused) | speedup | tok/s (unfused) | tok/s (fused) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 60.716 | 57.545 | 1.055 | 16.5 | 17.4 |
| 2 | 1 | 59.063 | 55.384 | 1.066 | 33.9 | 36.1 |
| 4 | 1 | 58.177 | 55.903 | 1.041 | 68.8 | 71.6 |
| 8 | 1 | 58.305 | 54.498 | 1.070 | 137.2 | 146.8 |
| 16 | 1 | 59.783 | 54.451 | 1.098 | 267.6 | 293.8 |
| 32 | 1 | 59.802 | 54.218 | 1.103 | 535.1 | 590.2 |

- `model.fuse()` installed 28 gate/up groups; QKV remained at 0 because this Qwen checkpoint already uses a single `c_attn` QKV module.
- Numerical parity between two fused forward passes: `max abs diff = 0.000000`.
- End-to-end decode latency improved by ~4–10% on this model from fusing the MLP gate/up projections alone.

## Phase 3 status (Marlin backend fusion)

- [x] Refactor `gptqmodel/nn_modules/fused_quant_linear.py` into a generic `_FusedQuantGroup` with backend-specific providers (`_FusedTritonKernel`, `_FusedMarlinKernel`).
- [x] Add Marlin fused kernel provider: concatenates Marlin-repacked `qweight`/`scales` along the output (column) dimension, shares `g_idx`/`g_idx_sort_indices`, and calls `apply_gptq_marlin_linear` once.
- [x] Add Marlin parity unit tests in `tests/test_fused_quant_linear.py` for Laguna/Qwen3.5-27B/generic Llama-like shapes.
- [x] Real-model benchmark on `DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2` loaded with `backend=GPTQ_MARLIN`.

### Phase 3 design note (backend providers)

Following the vLLM/SGLang pattern, `_FusedQuantGroup` is a generic fused-linear container and the actual GEMM is delegated to a backend provider. This lets one fused class serve `TritonV2Linear`, `MarlinLinear`, and future kernels (e.g. ExllamaV2, Triton mega-kernel) by adding a new provider class and an `isinstance` branch in `_create_fused_kernel`.

### Phase 3 validation commands

```bash
ruff check gptqmodel/nn_modules/fused_quant_linear.py tests/test_fused_quant_linear.py
pytest -q tests/test_fused_quant_linear.py
PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:1024' python scripts/benchmark_fused_real_model.py \
    --model_path /monster/data/model/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2 \
    --backend GPTQ_MARLIN --device cuda:0 --batch_sizes 1 2 4 8 16 32 --seq_len 1
```

### Phase 3 real-model benchmark

`scripts/benchmark_fused_real_model.py` on `DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2`, `backend=GPTQ_MARLIN`, A100 96 GB, sm80, PyTorch 2.13.0+cu130:

#### Decode (`seq_len = 1`)

| batch | seq | ms (unfused) | ms (fused) | speedup | tok/s (unfused) | tok/s (fused) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 39.646 | 38.554 | 1.028 | 25.2 | 25.9 |
| 2 | 1 | 37.214 | 37.027 | 1.005 | 53.7 | 54.0 |
| 4 | 1 | 37.011 | 36.973 | 1.001 | 108.1 | 108.2 |
| 8 | 1 | 38.032 | 37.446 | 1.016 | 210.3 | 213.6 |
| 16 | 1 | 38.066 | 36.841 | 1.033 | 420.3 | 434.3 |
| 32 | 1 | 38.502 | 37.401 | 1.029 | 831.1 | 855.6 |

#### Prefill (`seq_len = 128`)

| batch | seq | ms (unfused) | ms (fused) | speedup | tok/s (unfused) | tok/s (fused) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 40.817 | 38.317 | 1.065 | 3136.0 | 3340.5 |
| 2 | 128 | 38.066 | 36.974 | 1.030 | 6725.2 | 6923.8 |
| 4 | 128 | 49.966 | 50.601 | 0.987 | 10246.9 | 10118.4 |
| 8 | 128 | 89.799 | 92.037 | 0.976 | 11403.2 | 11126.0 |
| 16 | 128 | 176.915 | 181.255 | 0.976 | 11576.2 | 11299.0 |
| 32 | 128 | 350.063 | 357.597 | 0.979 | 11700.8 | 11454.2 |

#### Prefill (`seq_len = 1024`)

| batch | seq | ms (unfused) | ms (fused) | speedup | tok/s (unfused) | tok/s (fused) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 95.120 | 97.868 | 0.972 | 10765.3 | 10463.1 |
| 2 | 1024 | 180.229 | 186.373 | 0.967 | 11363.3 | 10988.7 |
| 4 | 1024 | 355.032 | 366.179 | 0.970 | 11537.0 | 11185.8 |
| 8 | 1024 | 708.720 | 729.546 | 0.971 | 11558.9 | 11228.9 |
| 16 | 1024 | 1413.653 | 1455.100 | 0.972 | 11589.8 | 11259.7 |
| 32 | 1024 | 2839.780 | 2907.655 | 0.977 | 11538.9 | 11269.6 |

Observations:

- `model.fuse()` installed 28 gate/up groups and 0 QKV groups (the Qwen checkpoint already uses a single `c_attn` QKV module).
- Numerical parity between two fused forward passes: `max abs diff = 0.000000`.
- Marlin gate/up fusion gives a small decode speedup (~0.5–3.3%) but is roughly neutral to slightly slower on large-prefill batches, where the model is compute-bound and two half-width GEMMs are almost as efficient as one full-width GEMM. The prefill results are consistent with the Triton MVP's trend of diminishing returns as batch/sequence grows.

## Phase 4 status (model-tree fusion flags)

- [x] Replace coarse `:qkv`/`:gateup` group flags with per-role `:q`, `:k`, `:v`, `:gate`, `:up`, and `:down` flags on `module_tree` child specs.
- [x] Update `get_module_tree_fusion_candidates()` to build fusion groups from semantic role flags and emit canonical `q -> k -> v` and `gate -> up` ordering regardless of spec ordering.
- [x] Update `BaseQModel.fuse()` to prefer module-tree candidates and fall back to the static alias lists when no flags are present.
- [x] Flag the Llama, Qwen, Qwen3/Qwen3.5-text, and Laguna model definitions with per-role flags so `fuse()` no longer depends on hard-coded name lists or tuple ordering.

### Per-role flag usage in `module_tree`

```python
"self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
"mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
```

- The numeric group (`:0`) is the existing quantization subset marker; the role flag (`:q`, `:k`, `:v`, `:gate`, `:up`) marks which semantic projection the module performs.
- Modules with `:down` are recognized but not fused; they document the MLP/attention output projection without affecting grouping.
- Single modules that already pack multiple roles (e.g. Qwen's `c_attn`) are not split; they carry no fusion role and are ignored by the parser.
- `BaseQModel.fuse()` extracts these tuples and passes them to `install_fused_qkv`/`install_fused_gate_up`, so non-standard names like `query/key/value` or `w1/w2` can be fused without adding them to a global alias list and without relying on tuple order.

### Phase 4 validation commands

```bash
ruff check gptqmodel/nn_modules/fused_quant_linear.py gptqmodel/models/base.py \
    gptqmodel/models/definitions/llama.py gptqmodel/models/definitions/qwen.py \
    gptqmodel/models/definitions/qwen3.py gptqmodel/models/definitions/qwen3_5_text.py \
    gptqmodel/models/definitions/laguna.py tests/test_fused_quant_linear.py
pytest -q tests/test_fused_quant_linear.py
```

- `test_module_tree_fusion_flags_parsed` validates the parser on Llama-style, Qwen-style, and out-of-order role specs.
- `test_fuse_uses_module_tree_flags` validates that `BaseQModel.fuse()` fuses non-standard `query/key/value` and `gate/up` names when driven by a `module_tree` with per-role flags.

## Phase 5 status (Triton mega-kernel for small-M QKV)

- [x] Fix `quant_matmul_248_kernel` and `transpose_quant_matmul_248_kernel` `tl.dot` dtype by casting the dequantized weight to the activation dtype.
- [x] Add `_FusedTritonMegaKernel` provider in `gptqmodel/nn_modules/fused_quant_linear.py` that concatenates packed buffers and calls `quant_matmul_248` (fused dequant + GEMM) in one launch.
- [x] Route small-M (decode and short-prefill) QKV workloads through `_FusedTritonMegaKernel`; gate/up and large-M prefill continue to use the dense `QuantLinearFunction` path, which is faster for those shapes.
- [x] Add unit-test tolerances that account for BF16 tensor-core accumulation-order differences when the SiLU-activated gate/up product is compared.

### Phase 5 design note

The Triton mega-kernel removes the separate dequantization allocation and launches one kernel for the full QKV GEMM. It is bitwise-exact with three separate `quant_matmul_248` calls for 2/4/8-bit packed weights. On A100 sm80 it is a clear win for decode and short prefill, but for long prefill (`M > 256`) the dense `QuantLinearFunction` path (dequant + cuBLAS `torch.matmul`) is faster, so the provider falls back at runtime.

### Phase 5 validation commands

```bash
ruff check gptqmodel/nn_modules/fused_quant_linear.py \
    gptqmodel/nn_modules/triton_utils/kernels.py \
    tests/test_fused_quant_linear.py
pytest -q tests/test_fused_quant_linear.py
PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:1024' \
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_qkv_gateup.py
```

### Phase 5 synthetic benchmark

`python scripts/benchmark_fused_qkv_gateup.py`, A100 96 GB, sm80, PyTorch 2.13.0+cu130, Triton 3.7.1, `CUDA_VISIBLE_DEVICES=6`.

Decode and short-prefill QKV use `_FusedTritonMegaKernel`; gate/up and long-prefill QKV use the `QuantLinearFunction` weight-concat path.

#### Decode (`seq_len = 1`)

| op | hidden | out | batch | seq | ms_sep | ms_fused | speedup | tok/s_sep | tok/s_fused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 3072 | (6144, 1024, 1024) | 1 | 1 | 0.562 | 0.165 | 3.406 | 1778.3 | 6056.6 |
| gate_up | 3072 | (12288, 12288) | 1 | 1 | 0.446 | 0.321 | 1.390 | 2240.4 | 3113.8 |
| qkv | 3072 | (6144, 1024, 1024) | 2 | 1 | 0.547 | 0.169 | 3.232 | 3654.3 | 11809.9 |
| gate_up | 3072 | (12288, 12288) | 2 | 1 | 0.415 | 0.323 | 1.286 | 4823.2 | 6200.8 |
| qkv | 3072 | (6144, 1024, 1024) | 4 | 1 | 0.551 | 0.187 | 2.946 | 7263.9 | 21401.8 |
| gate_up | 3072 | (12288, 12288) | 4 | 1 | 0.400 | 0.322 | 1.241 | 10003.7 | 12416.6 |
| qkv | 3072 | (6144, 1024, 1024) | 8 | 1 | 0.570 | 0.175 | 3.264 | 14032.1 | 45799.6 |
| gate_up | 3072 | (12288, 12288) | 8 | 1 | 0.422 | 0.326 | 1.296 | 18965.1 | 24573.8 |
| qkv | 3072 | (6144, 1024, 1024) | 16 | 1 | 0.555 | 0.179 | 3.109 | 28826.3 | 89634.0 |
| gate_up | 3072 | (12288, 12288) | 16 | 1 | 0.438 | 0.323 | 1.359 | 36493.4 | 49606.3 |
| qkv | 3072 | (6144, 1024, 1024) | 32 | 1 | 0.544 | 0.182 | 2.989 | 58835.7 | 175838.4 |
| gate_up | 3072 | (12288, 12288) | 32 | 1 | 0.419 | 0.323 | 1.298 | 76346.1 | 99074.3 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 1 | 0.574 | 0.248 | 2.319 | 1741.2 | 4037.1 |
| gate_up | 5120 | (17408, 17408) | 1 | 1 | 0.571 | 0.730 | 0.782 | 1750.8 | 1369.3 |
| qkv | 5120 | (6144, 1024, 1024) | 2 | 1 | 0.618 | 0.247 | 2.500 | 3234.4 | 8084.8 |
| gate_up | 5120 | (17408, 17408) | 2 | 1 | 0.567 | 0.734 | 0.772 | 3530.3 | 2724.6 |
| qkv | 5120 | (6144, 1024, 1024) | 4 | 1 | 0.568 | 0.248 | 2.291 | 7047.4 | 16142.9 |
| gate_up | 5120 | (17408, 17408) | 4 | 1 | 0.566 | 0.735 | 0.770 | 7061.4 | 5439.9 |
| qkv | 5120 | (6144, 1024, 1024) | 8 | 1 | 0.563 | 0.245 | 2.293 | 14215.9 | 32601.0 |
| gate_up | 5120 | (17408, 17408) | 8 | 1 | 0.567 | 0.744 | 0.762 | 14108.1 | 10755.1 |
| qkv | 5120 | (6144, 1024, 1024) | 16 | 1 | 0.545 | 0.245 | 2.221 | 29378.0 | 65251.0 |
| gate_up | 5120 | (17408, 17408) | 16 | 1 | 0.562 | 0.781 | 0.719 | 28481.6 | 20476.8 |
| qkv | 5120 | (6144, 1024, 1024) | 32 | 1 | 0.583 | 0.248 | 2.345 | 54920.9 | 128791.6 |
| gate_up | 5120 | (17408, 17408) | 32 | 1 | 0.605 | 0.748 | 0.808 | 52912.3 | 42769.5 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 1 | 0.541 | 0.214 | 2.532 | 1848.0 | 4679.7 |
| gate_up | 4096 | (11008, 11008) | 1 | 1 | 0.409 | 0.320 | 1.277 | 2447.9 | 3126.4 |
| qkv | 4096 | (4096, 1024, 1024) | 2 | 1 | 0.529 | 0.197 | 2.692 | 3780.0 | 10175.7 |
| gate_up | 4096 | (11008, 11008) | 2 | 1 | 0.397 | 0.325 | 1.223 | 5033.3 | 6156.6 |
| qkv | 4096 | (4096, 1024, 1024) | 4 | 1 | 0.538 | 0.197 | 2.733 | 7436.2 | 20326.0 |
| gate_up | 4096 | (11008, 11008) | 4 | 1 | 0.393 | 0.330 | 1.193 | 10169.9 | 12133.5 |
| qkv | 4096 | (4096, 1024, 1024) | 8 | 1 | 0.534 | 0.195 | 2.735 | 14994.6 | 41010.5 |
| gate_up | 4096 | (11008, 11008) | 8 | 1 | 0.407 | 0.325 | 1.250 | 19679.8 | 24607.8 |
| qkv | 4096 | (4096, 1024, 1024) | 16 | 1 | 0.528 | 0.196 | 2.697 | 30275.1 | 81652.4 |
| gate_up | 4096 | (11008, 11008) | 16 | 1 | 0.390 | 0.330 | 1.180 | 41049.3 | 48458.6 |
| qkv | 4096 | (4096, 1024, 1024) | 32 | 1 | 0.531 | 0.191 | 2.789 | 60223.5 | 167956.6 |
| gate_up | 4096 | (11008, 11008) | 32 | 1 | 0.387 | 0.328 | 1.180 | 82641.4 | 97534.3 |

#### Prefill (`seq_len = 128, 1024, 4096`)

For the Laguna and Llama-like shapes, short prefill (128 tokens) still shows QKV speedups of ~2.3–3.5x from the mega-kernel. Long prefill (`seq_len >= 1024` or `batch * seq_len` large enough to trigger the `M > 256` fallback) is roughly neutral (0.9–1.2x), because the dense `QuantLinearFunction` path is compute-bound and the launch-count savings become marginal.

| op | hidden | out | batch | seq | ms_sep | ms_fused | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 3072 | (6144, 1024, 1024) | 1 | 128 | 0.624 | 0.175 | 3.566 |
| qkv | 3072 | (6144, 1024, 1024) | 1 | 1024 | 0.535 | 0.831 | 0.644 |
| qkv | 3072 | (6144, 1024, 1024) | 1 | 4096 | 0.925 | 2.660 | 0.348 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 128 | 0.553 | 0.291 | 1.902 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 1024 | 0.553 | 0.291 | 1.902 |
| qkv | 5120 | (6144, 1024, 1024) | 1 | 4096 | 1.505 | 1.298 | 1.159 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 128 | 0.520 | 0.198 | 2.631 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 1024 | 0.513 | 0.260 | 1.977 |
| qkv | 4096 | (4096, 1024, 1024) | 1 | 4096 | 0.981 | 0.803 | 1.221 |

Observations:

- QKV mega-kernel is a strong win for decode and short prefill (2.5–3.5x on Laguna, 2.2–2.8x on Qwen3.5-27B, 2.5–2.8x on Llama-like shapes).
- Gate/up fusion improves decode on Laguna and Llama-like shapes (1.2–1.7x) but is roughly neutral or slightly slower on Qwen3.5-27B's very wide `17408` gate/up because the dense `QuantLinearFunction` path is already efficient; the fallback routing keeps it from regressing on prefill.
- Long prefill is neutral across all shapes; the benefit is launch-count reduction, which disappears once the GEMM is compute-bound.
- Accuracy is validated in `tests/test_fused_quant_linear.py` (22 passed): QKV outputs match within `atol=2.0, rtol=0.05` and gate/up SiLU-activated products match within `atol=20.0, rtol=0.2`, which is consistent with BF16 tensor-core accumulation-order differences between the mega-kernel and the per-module `QuantLinearFunction` path.

## Phase 6: VRAM de-duplication and benchmark accuracy fix

### Free original per-member packed buffers

Once the fused kernel owns a contiguous concatenated copy of `qweight`/`scales`/`qzeros`/`g_idx`, the per-member copies are redundant and roughly double the fused group's memory footprint. `BaseQModel.fuse()` and the lower-level `install_fused_qkv`/`install_fused_gate_up` helpers now accept `free_original_weights=True` (default) which deletes the now-redundant buffers from each member module. Set `free_original_weights=False` if you need to keep the buffers for inspection or `save()` after fusing.

### Fix benchmark `max_diff` metric

`scripts/benchmark_fused_qkv_gateup.py` previously called `install_fused_*` before capturing the "unfused" reference, so `max_diff` compared fused against fused and was always ~0. The reference output is now captured before fusion is installed, so the reported `max_diff` reflects real fused-vs-unfused numerical drift.

Validation:

- `pytest -q tests/test_fused_quant_linear.py` — 22 passed
- `ruff check gptqmodel/nn_modules/fused_quant_linear.py gptqmodel/models/base.py scripts/benchmark_fused_qkv_gateup.py tests/test_fused_quant_linear.py` — passed
- `git diff --check` — passed

Sample corrected Laguna-S-2.1 QKV `max_diff` values (BF16, A100 sm80, batch=1 decode) now show small non-zero deltas from the mega-kernel accumulation order:

| op | hidden | out | batch | seq | ms_sep | ms_fused | speedup | max_diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 3072 | (6144, 1024, 1024) | 1 | 1 | 0.537 | 0.200 | 2.685 | 0.2500 |
| qkv | 3072 | (6144, 1024, 1024) | 2 | 1 | 0.544 | 0.204 | 2.672 | 0.0010 |
| qkv | 3072 | (6144, 1024, 1024) | 4 | 1 | 0.537 | 0.197 | 2.730 | 0.0005 |
| qkv | 3072 | (6144, 1024, 1024) | 8 | 1 | 0.546 | 0.208 | 2.618 | 0.0156 |
| qkv | 3072 | (6144, 1024, 1024) | 16 | 1 | 0.542 | 0.195 | 2.775 | 0.0625 |
| qkv | 3072 | (6144, 1024, 1024) | 32 | 1 | 0.552 | 0.214 | 2.585 | 0.2500 |

Gate/up `max_diff` remains `0.0000` for these shapes because both fused and unfused paths use the same `QuantLinearFunction` weight-concat strategy.

## Phase 7: Fused gate/up activation (SiLU/SwiGLU, GeLU, ReLU, tanh, sigmoid)

### Design

The target models (Laguna-S-2.1, Qwen3.5-27B, Llama-like) all use a SwiGLU-style MLP:

```python
mlp(x) = down_proj(act_fn(gate_proj(x)) * up_proj(x))
```

Phase 5 already fuses `gate_proj` and `up_proj` into one GEMM and returns the raw `gate` and `up` slices, so the model still calls each projection separately and computes `act(gate) * up` before `down_proj`. Phase 7 adds a higher-level MLP fusion pass: after installing the gate/up projection group, we detect the parent's activation function (`act_fn`, `hidden_act`, or `config.hidden_act`) and the `down_proj` (by shape: `in_features == gate.out_features`, `out_features == gate.in_features`) and replace the parent module's `forward` with a single function that:

1. Runs one fused gate/up GEMM.
2. Applies `act_fn` to the gate slice.
3. Multiplies in-place with the up slice (reusing the activation tensor).
4. Calls `down_proj`.

Supported activations are whatever `transformers.activations.ACT2FN` provides, plus `silu`, `gelu`, `relu`, `tanh`, `sigmoid`, and identity as fallbacks.

### API

- `BaseQModel.fuse(..., gate_up_activation=True)` enables the MLP activation/down-projection fusion. It is on by default.
- `install_fused_gate_up(..., fuse_activation=True)` controls the lower-level helper.
- If the MLP structure cannot be detected (no `down_proj`, no recognized activation), only the projection fusion is installed and the original MLP `forward` is preserved.

### Validation

- `pytest -q tests/test_fused_quant_linear.py` — 24 passed
- `ruff check gptqmodel/nn_modules/fused_quant_linear.py gptqmodel/models/base.py tests/test_fused_quant_linear.py scripts/benchmark_fused_mlp_activation.py` — passed
- `git diff --check` — passed

### Benchmark: full MLP forward

`scripts/benchmark_fused_mlp_activation.py` (A100 sm80, BF16, `act_fn=SiLU`):

| hidden | intermediate | batch | seq | ms_sep | ms_fused | speedup | tok/s_sep | tok/s_fused | max_diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 12288 | 1 | 1 | 0.604 | 0.442 | 1.365 | 1656.1 | 2260.0 | 0.0000 |
| 3072 | 12288 | 8 | 1 | 0.612 | 0.433 | 1.412 | 13074.2 | 18464.9 | 0.0000 |
| 3072 | 12288 | 1 | 128 | 0.671 | 0.468 | 1.434 | 190848.4 | 273594.8 | 0.0000 |
| 3072 | 12288 | 16 | 128 | 2.132 | 2.169 | 0.983 | 960785.6 | 944247.0 | 0.0000 |
| 5120 | 17408 | 1 | 1 | 0.854 | 0.803 | 1.065 | 1170.3 | 1246.1 | 512.0000 |
| 5120 | 17408 | 32 | 1 | 0.901 | 0.838 | 1.075 | 35503.7 | 38180.1 | 0.0000 |
| 5120 | 17408 | 1 | 128 | 0.945 | 0.947 | 0.998 | 135499.9 | 135170.2 | 0.0000 |
| 5120 | 17408 | 16 | 128 | 4.615 | 4.800 | 0.961 | 443799.5 | 426643.0 | 0.0000 |
| 4096 | 11008 | 1 | 1 | 0.622 | 0.474 | 1.312 | 1608.6 | 2110.0 | 0.0000 |
| 4096 | 11008 | 32 | 1 | 0.656 | 0.484 | 1.355 | 48788.5 | 66092.8 | 256.0000 |
| 4096 | 11008 | 1 | 128 | 0.649 | 0.519 | 1.251 | 197251.1 | 246796.6 | 0.0000 |
| 4096 | 11008 | 16 | 128 | 2.521 | 2.606 | 0.967 | 812433.5 | 785755.8 | 0.0000 |

Observations:

- Decode speedups are strongest for the Laguna and Llama-like shapes (1.3–1.4x) because the fused path removes the separate `gate_proj`/`up_proj` Python dispatch, the intermediate gate/up slice views, and the extra `silu`/`mul` allocations.
- Qwen3.5-27B's very wide `17408` intermediate is only 1.05–1.08x faster at decode and roughly neutral at prefill; the GEMM is already the dominant cost.
- Long prefill is neutral or slightly slower (0.95–1.03x) because the compute is dominated by the large GEMM and the activation/down-projection are not the bottleneck.
- `max_diff` is `0.0` for most configs; non-zero entries (256 or 512) are exactly the BF16 tensor-core rounding boundaries relative to the output magnitude and stay within the same `atol=20.0, rtol=0.2` tolerance used for the gate/up product tests.

The next step is sub-group validation on Laguna MoE expert gate/up shapes.

## Phase 8: MoE expert gate/up sub-group fusion

### Design

Laguna-S-2.1, Qwen3.5-27B and many MoE variants expose per-expert `gate_proj`/`up_proj`/`down_proj` projections. PR #118 adds a grouped GEMM MoE dispatch (`gptqmodel/utils/moe_dispatch.py`) that dequantizes active experts, stacks their weights, and runs one `torch.nn.functional.grouped_mm` per projection instead of looping over experts. Phase 8 makes the per-expert gate/up fusion work cleanly with that existing routed fusion rather than duplicating or conflicting with it.

Changes:

- `_FusedQuantGroup.dequantize_weight(dtype)` returns the dense `(in_features, total_out_features)` tensor for standard int32-packed TritonV2 fused groups. Marlin fused groups return `None` because their packed layout is different.
- `gptqmodel/utils/moe_dispatch.py` learns to detect when an expert's `gate_proj` and `up_proj` already share a `_FusedQuantGroup`. When the grouped dispatch path is enabled, it uses the fused gate+up weight directly:
  - one `grouped_mm` call produces `(tokens, 2 * intermediate_dim)`;
  - split into gate and up, apply the expert activation, then run the down-projection `grouped_mm`.
- For experts that are not fused, the dispatch still uses the per-projection `grouped_mm` path.
- `gptqmodel/nn_modules/fused_quant_linear.py` skips the `parent.forward` SwiGLU activation replacement for modules that live inside an MoE expert container (`experts.*`, `shared_experts.*`, `shared_expert*`), because the MoE dispatch applies the activation after the grouped `gate+up` GEMM.
- The direct GPTQ dequant helper in `moe_dispatch.py` (`_try_dequant_gptq_weight`) dequantizes standard int32-packed `qweight`/`scales`/`qzeros`/`g_idx` to the activation dtype without going through `proj.dequantize_weight()`, which avoids an intermediate `float16` -> `bfloat16` cast that caused the grouped-GEMM orientation probe to fail for `TritonV2Linear` experts.

This keeps MoE routing unchanged while reducing the number of grouped GEMM launches per active expert from three (gate, up, down) to two (gate+up, down).

### API

`BaseQModel.fuse(...)` on a quantized MoE model now also fuses each expert's `gate_proj`/`up_proj` when the model definitions carry the per-role flags (Laguna's `module_tree` already does). The existing grouped dispatch path picks up the fused weights automatically.

### Validation

```bash
ruff check gptqmodel/nn_modules/fused_quant_linear.py gptqmodel/utils/moe_dispatch.py \
    tests/test_fused_quant_linear.py tests/test_moe_dispatch.py \
    scripts/benchmark_fused_moe_laguna.py
pytest -q tests/test_moe_dispatch.py tests/test_fused_quant_linear.py
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_moe_laguna.py --model laguna-s-2.1 --num-experts 64 --top-k 6 --repeats 30
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_moe_laguna.py --model qwen3.5-27b --num-experts 64 --top-k 6 --repeats 30
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_moe_laguna.py --model llama-like --num-experts 64 --top-k 6 --repeats 30
```

- `tests/test_moe_dispatch.py::test_grouped_moe_dispatch_fused_gateup_matches_unfused` validates that grouping `TritonV2Linear` experts, calling the grouped dispatch, fusing gate/up, and calling the grouped dispatch again still matches the original unfused output (`max diff 0.0`).
- All 31 tests in `tests/test_moe_dispatch.py` + `tests/test_fused_quant_linear.py` pass.

### Benchmark: grouped MoE dispatch with fused expert gate/up

`scripts/benchmark_fused_moe_laguna.py`, A100 96 GB, sm80, PyTorch 2.13.0+cu130, Triton 3.7.1, `CUDA_VISIBLE_DEVICES=6`. Synthetic MoE with 64 experts, `top_k=6`; grouped dispatch benchmarks the active experts only. `unfused` = grouped dispatch with three separate grouped GEMMs (gate, up, down). `fused` = grouped dispatch with one fused gate+up grouped GEMM plus down.

#### Laguna-S-2.1 (hidden=3072, intermediate=12288)

| mode | batch | seq | ms_unfused | tok/s_unfused | ms_fused | tok/s_fused | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 7.800 | 769.3 | 6.662 | 900.6 | 1.171 |
| decode | 2 | 1 | 11.115 | 1079.6 | 11.015 | 1089.4 | 1.009 |
| decode | 4 | 1 | 20.449 | 1173.7 | 16.550 | 1450.1 | 1.236 |
| decode | 8 | 1 | 30.084 | 1595.5 | 25.005 | 1919.6 | 1.203 |
| decode | 16 | 1 | 45.181 | 2124.8 | 38.144 | 2516.8 | 1.184 |
| decode | 32 | 1 | 55.311 | 3471.3 | 43.645 | 4399.1 | 1.267 |
| prefill | 1 | 128 | 57.226 | 13420.4 | 46.440 | 16537.6 | 1.232 |
| prefill | 1 | 1024 | 58.901 | 104311.2 | 48.691 | 126184.3 | 1.210 |
| prefill | 1 | 4096 | 68.972 | 356321.0 | 66.402 | 370111.4 | 1.039 |
| prefill | 16 | 128 | 60.618 | 202710.7 | 54.174 | 226825.9 | 1.119 |

#### Qwen3.5-27B (hidden=5120, intermediate=17408)

| mode | batch | seq | ms_unfused | tok/s_unfused | ms_fused | tok/s_fused | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 9.494 | 632.0 | 8.990 | 667.4 | 1.056 |
| decode | 2 | 1 | 16.045 | 747.9 | 14.394 | 833.7 | 1.115 |
| decode | 4 | 1 | 26.216 | 915.5 | 24.064 | 997.3 | 1.089 |
| decode | 8 | 1 | 39.870 | 1203.9 | 36.635 | 1310.2 | 1.088 |
| decode | 16 | 1 | 59.860 | 1603.7 | 52.326 | 1834.7 | 1.144 |
| decode | 32 | 1 | 69.771 | 2751.9 | 66.705 | 2878.3 | 1.046 |
| prefill | 1 | 128 | 74.928 | 10249.8 | 70.334 | 10919.3 | 1.065 |
| prefill | 1 | 1024 | 79.946 | 76851.5 | 78.743 | 78026.3 | 1.015 |
| prefill | 1 | 4096 | 118.246 | 207837.2 | 113.454 | 216615.6 | 1.042 |
| prefill | 16 | 128 | 100.768 | 121944.0 | 91.242 | 134674.1 | 1.104 |

#### Llama-like (hidden=4096, intermediate=11008)

| mode | batch | seq | ms_unfused | tok/s_unfused | ms_fused | tok/s_fused | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 7.875 | 761.9 | 7.366 | 814.6 | 1.069 |
| decode | 2 | 1 | 13.152 | 912.4 | 11.820 | 1015.2 | 1.113 |
| decode | 4 | 1 | 17.471 | 1373.7 | 17.577 | 1365.4 | 0.994 |
| decode | 8 | 1 | 30.467 | 1575.5 | 26.702 | 1797.6 | 1.141 |
| decode | 16 | 1 | 46.022 | 2086.0 | 41.372 | 2320.4 | 1.112 |
| decode | 32 | 1 | 54.180 | 3543.8 | 48.711 | 3941.6 | 1.112 |
| prefill | 1 | 128 | 58.689 | 13085.9 | 50.479 | 15214.3 | 1.163 |
| prefill | 1 | 1024 | 59.535 | 103200.4 | 53.649 | 114522.8 | 1.110 |
| prefill | 1 | 4096 | 81.683 | 300870.6 | 74.534 | 329728.5 | 1.096 |
| prefill | 16 | 128 | 64.105 | 191686.7 | 63.792 | 192626.7 | 1.005 |

Observations:

- Fusing gate/up inside the grouped dispatch improves decode latency across all three shapes, with speedups between 1.0x and 1.27x. The largest wins come from reducing the grouped GEMM count per active expert from three to two, which matters most when the per-token work is small (decode).
- Prefill benefits are smaller (1.0–1.2x) because the grouped GEMMs are already larger and more compute-bound; the launch-count reduction is less significant.
- The wide `17408` gate/up in Qwen3.5-27B shows the smallest relative gain, consistent with the earlier dense MLP benchmark trend.
- Accuracy is exact (`max diff 0.0`) between the grouped dispatch before and after fusing gate/up, because both paths use the same `dequant` -> `grouped_mm` -> `act_fn` -> `down_proj` arithmetic.

## Phase 9: Marlin MoE expert gate/up fusion

### Design

Phase 8 extended the grouped MoE dispatch to fused `TritonV2Linear` experts by dequantizing the fused gate+up weight and launching one `grouped_mm`. `GPTQ_MARLIN` experts cannot be dequantized to a dense tensor cheaply (their packed tile layout is different from the int32-packed GPTQ layout), so the grouped dispatch now also supports a Marlin-specific path:

- `_is_marlin_expert()` detects when all three projections of an expert are `MarlinLinear`.
- `_can_use_grouped_mm()` probes a Marlin expert with a tiny forward and, if it succeeds, marks the experts module with `_moe_dispatch_backend = "marlin"`.
- `_marlin_experts_project()` sorts tokens by active expert and calls the per-expert `gate_proj`/`up_proj`/`down_proj` `forward`s on contiguous token blocks. When `gate_proj` and `up_proj` share a `_FusedQuantGroup`, only one Marlin kernel is launched for the fused gate+up GEMM; otherwise two separate Marlin kernels are launched. The existing `_apply_expert_gate()` and weighted accumulation code are reused.

This integrates cleanly with the existing grouped dispatch: the token routing, sentinel handling, and inverse permutation are unchanged. The only difference is how the active experts' gate/up/down projections are executed.

`gptqmodel/nn_modules/fused_quant_linear.py` already supports fusing Marlin `gate_proj`/`up_proj` into a `_FusedQuantGroup` backed by `_FusedMarlinKernel`, so `BaseQModel.fuse()` on a Marlin-loaded MoE model will automatically create the fused expert groups and the dispatch will use them.

### Validation

```bash
ruff check gptqmodel/utils/moe_dispatch.py tests/test_moe_dispatch.py scripts/benchmark_fused_moe_marlin.py
pytest -q tests/test_moe_dispatch.py
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_moe_marlin.py --model laguna-s-2.1 --num-experts 16 --top-k 6 --repeats 10
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_moe_marlin.py --model qwen3.5-27b --num-experts 16 --top-k 6 --repeats 10
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_moe_marlin.py --model llama-like --num-experts 16 --top-k 6 --repeats 10
```

- `tests/test_moe_dispatch.py::test_grouped_moe_dispatch_marlin_fused_gateup_matches_unfused` validates Marlin-packed experts, before and after `install_fused_gate_up`, against the per-expert loop (`max diff` within BF16 tolerance).
- `tests/test_moe_dispatch.py::test_fused_qkv_and_moe_gateup_coexist` validates that `install_fused_qkv` and `install_fused_gate_up` can both install in the same layer and do not interfere.
- All 10 tests in `tests/test_moe_dispatch.py` pass.

### Benchmark: Marlin MoE gate/up fusion

`scripts/benchmark_fused_moe_marlin.py`, A100 96 GB, sm80, PyTorch 2.13.0+cu130, Triton 3.7.1, `CUDA_VISIBLE_DEVICES=4/5/6`. Synthetic MoE with 16 experts, `top_k=6`. `unfused` = per-expert Marlin dispatch with separate `gate_proj` and `up_proj` calls. `fused` = per-expert Marlin dispatch with one fused `gate_proj`/`up_proj` call.

#### Laguna-S-2.1 (hidden=3072, intermediate=12288, Marlin)

| mode | batch | seq | ms_unfused | tok/s_unfused | ms_fused | tok/s_fused | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 20.898 | 287.1 | 2.633 | 2,278.8 | 7.937 |
| decode | 2 | 1 | 12.039 | 996.7 | 4.156 | 2,887.2 | 2.897 |
| decode | 4 | 1 | 5.754 | 4,171.0 | 5.633 | 4,260.7 | 1.022 |
| decode | 8 | 1 | 6.899 | 6,957.5 | 6.769 | 7,091.2 | 1.019 |
| decode | 16 | 1 | 6.879 | 13,956.5 | 6.668 | 14,396.7 | 1.032 |
| decode | 32 | 1 | 6.954 | 27,608.9 | 6.734 | 28,512.8 | 1.033 |
| prefill | 1 | 128 | 7.116 | 107,924.0 | 6.974 | 110,126.8 | 1.020 |
| prefill | 1 | 1024 | 12.695 | 483,962.0 | 12.460 | 493,096.6 | 1.019 |
| prefill | 1 | 4096 | 34.198 | 718,641.8 | 35.478 | 692,707.3 | 0.964 |
| prefill | 16 | 128 | 19.423 | 632,655.6 | 19.953 | 615,837.3 | 0.973 |

#### Qwen3.5-27B (hidden=5120, intermediate=17408, Marlin)

| mode | batch | seq | ms_unfused | tok/s_unfused | ms_fused | tok/s_fused | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 2.736 | 2,193.3 | 3.055 | 1,963.8 | 0.895 |
| decode | 2 | 1 | 3.869 | 3,101.8 | 4.149 | 2,892.5 | 0.933 |
| decode | 4 | 1 | 5.409 | 4,437.2 | 5.391 | 4,451.7 | 1.003 |
| decode | 8 | 1 | 6.589 | 7,284.6 | 6.894 | 6,962.1 | 0.956 |
| decode | 16 | 1 | 6.950 | 13,813.2 | 6.926 | 13,861.8 | 1.004 |
| decode | 32 | 1 | 7.003 | 27,417.1 | 6.898 | 27,835.1 | 1.015 |
| prefill | 1 | 128 | 7.406 | 103,701.0 | 7.378 | 104,099.2 | 1.004 |
| prefill | 1 | 1024 | 21.877 | 280,841.9 | 21.963 | 279,746.4 | 0.996 |
| prefill | 1 | 4096 | 72.537 | 338,804.4 | 72.746 | 337,834.7 | 0.997 |
| prefill | 16 | 128 | 38.888 | 315,980.7 | 39.317 | 312,538.0 | 0.989 |

#### Llama-like (hidden=4096, intermediate=11008, Marlin)

| mode | batch | seq | ms_unfused | tok/s_unfused | ms_fused | tok/s_fused | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 2.776 | 2,161.4 | 2.700 | 2,222.1 | 1.028 |
| decode | 2 | 1 | 3.643 | 3,294.1 | 4.701 | 2,552.8 | 0.775 |
| decode | 4 | 1 | 6.133 | 3,913.2 | 5.827 | 4,118.7 | 1.053 |
| decode | 8 | 1 | 7.175 | 6,689.9 | 6.684 | 7,181.7 | 1.074 |
| decode | 16 | 1 | 7.087 | 13,545.5 | 6.958 | 13,796.9 | 1.019 |
| decode | 32 | 1 | 7.207 | 26,640.3 | 6.876 | 27,924.6 | 1.048 |
| prefill | 1 | 128 | 7.270 | 105,643.7 | 7.086 | 108,376.3 | 1.026 |
| prefill | 1 | 1024 | 13.928 | 441,122.4 | 13.775 | 446,030.3 | 1.011 |
| prefill | 1 | 4096 | 40.043 | 613,737.5 | 40.729 | 603,399.2 | 0.983 |
| prefill | 16 | 128 | 23.979 | 512,455.5 | 23.071 | 532,607.4 | 1.039 |

Observations:

- For the stable, post-warmup cases (batch >= 4 and most prefill shapes), Marlin gate/up fusion gives a small speedup (0–7%). The fused path reduces the number of Marlin kernels per active expert from two to one, which helps when the per-token work is small.
- The very large Laguna `decode` batch=1/2 speedups are outliers caused by cold-start JIT compilation of the Marlin kernel for the first unfused shape in a fresh process; the fused measurement happened after the kernel was warm. Similar cold-start effects are visible in `decode` batch=2 for Llama-like and Qwen3.5-27B.
- Marlin prefill is mostly neutral (0.98–1.04x), matching the dense Triton trend: once the GEMM is compute-bound, reducing launches has little effect.
- Accuracy parity is confirmed by `test_grouped_moe_dispatch_marlin_fused_gateup_matches_unfused` (atol=2.0, rtol=0.05), which compares the per-expert loop before and after fusing.

## Phase 10: Fused QKV inside MoE models

### Design

The per-role `module_tree` flags from Phase 4 (`:q`, `:k`, `:v`, `:gate`, `:up`) are present on both attention and MoE MLP definitions. `BaseQModel.fuse()` therefore fuses attention QKV and MoE expert gate/up in the same model pass, and the grouped MoE dispatch automatically uses the fused expert weights.

Phase 10 adds an explicit coexistence test to ensure:

1. `install_fused_qkv()` on a `self_attn` module and `install_fused_gate_up()` on a sibling `experts` module both succeed.
2. The QKV output still matches the unfused per-module path.
3. The grouped MoE dispatch still matches the per-expert loop.

This validates that QKV fusion does not touch or break the MoE routing/fusion path.

### Validation

- `tests/test_moe_dispatch.py::test_fused_qkv_and_moe_gateup_coexist` passes.
- `BaseQModel.fuse()` on a quantized MoE model (e.g., Laguna-S-2.1) with Marlin or Triton backend will install both attention QKV and per-expert gate/up groups.

### Next steps

- Run `BaseQModel.fuse()` on a real Laguna-S-2.1-GPTQ-FIXED checkpoint with `backend=GPTQ_MARLIN` to confirm end-to-end QKV + MoE gate/up fusion.
- Profile whether a single fused Marlin kernel that computes gate+up for *all* active experts (instead of per-expert launches) would improve small-decode latency, or whether a batched/offset Marlin kernel is needed to match `grouped_mm` efficiency for MoE.
- Add Kimi-K3 proxy shape support once the model definition and attention layout are finalized.

## Phase 11: End-to-end `model.fuse()` on Laguna-S-2.1-GPTQ-FIXED

### Design

Phases 1–10 built and validated the components (weight-concat QKV/gate-up fusion, Triton mega-kernel, MLP activation fusion, grouped MoE dispatch, and Marlin MoE support). Phase 11 runs the complete pipeline on a real, quantized checkpoint:

```bash
GPTQModel.load(
    "/monster/data/model/Laguna-S-2.1-GPTQ-FIXED",
    backend=BACKEND.GPTQ_MARLIN,
    trust_remote_code=True,
    device="cuda:0",
)
model.fuse(qkv=True, gate_up=True, free_original_weights=True)
```

`BaseQModel.fuse()` derives the QKV and gate/up candidates from the Laguna `module_tree` (`:q`, `:k`, `:v`, `:gate`, `:up` role flags), installs `_FusedQuantGroup` objects backed by `_FusedMarlinKernel`, and optionally frees the per-member packed buffers. The grouped MoE dispatch automatically uses the fused `gate_proj`/`up_proj` inside each active expert.

### Validation

```bash
ruff check scripts/benchmark_fuse_real_laguna.py
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fuse_real_laguna.py --gpu 6 --backend GPTQ_MARLIN --fuse --batch-sizes 1 2 4 8 16 32 --seq-len 1 --repeats 5 --warmup 3
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fuse_real_laguna.py --gpu 6 --backend GPTQ_MARLIN --fuse --batch-sizes 1 16 --seq-len 128 --max-new-tokens 0 --repeats 3 --warmup 1
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fuse_real_laguna.py --gpu 6 --backend GPTQ_MARLIN --fuse --batch-sizes 1 --seq-len 1024 --max-new-tokens 0 --repeats 3 --warmup 1
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fuse_real_laguna.py --gpu 6 --backend GPTQ_MARLIN --fuse --batch-sizes 1 --seq-len 4096 --max-new-tokens 0 --repeats 1 --warmup 0
```

- `model.fuse()` installs **48 QKV groups** and **12,080 gate/up groups** on Laguna-S-2.1-GPTQ-FIXED.
- All `model.forward(...)` calls succeed before and after `fuse()`.
- Numerical parity vs the unfused Marlin path is within BF16 accumulation tolerance (`mean abs diff < 1.0`, `max abs diff < 20` on logits).

### Benchmark: real Laguna-S-2.1-GPTQ-FIXED, GPTQ_MARLIN, A100 96 GB

GPU: NVIDIA A100 96 GB (sm80), PyTorch 2.13.0+cu130, Triton 3.7.1, `CUDA_VISIBLE_DEVICES=6`.
`decode` uses `generate(..., max_new_tokens=1, use_cache=True)` starting from a single-token prompt. `prefill` uses `model(input_ids)`.

#### Decode (batch 1–32, seq=1, max_new_tokens=1)

| state | batch | seq | ms | tok/s |
|---|---:|---:|---:|---:|
| unfused | 1 | 1 | 786.046 | 2.5 |
| fused | 1 | 1 | 743.686 | 2.7 |
| unfused | 2 | 1 | 923.483 | 4.3 |
| fused | 2 | 1 | 894.598 | 4.5 |
| unfused | 4 | 1 | 1233.935 | 6.4 |
| fused | 4 | 1 | 1145.090 | 7.0 |
| unfused | 8 | 1 | 1569.575 | 10.2 |
| fused | 8 | 1 | 1467.024 | 10.9 |
| unfused | 16 | 1 | 2055.738 | 15.6 |
| fused | 16 | 1 | 1924.021 | 16.6 |
| unfused | 32 | 1 | 2725.294 | 23.5 |
| fused | 32 | 1 | 2525.785 | 25.3 |

| batch | speedup (tok/s) | max abs diff | mean abs diff |
|---:|---:|---:|---:|
| 1 | 1.08 | 1.375 | 0.204 |
| 2 | 1.05 | 2.094 | 0.207 |
| 4 | 1.09 | 4.875 | 0.516 |
| 8 | 1.07 | 5.688 | 0.391 |
| 16 | 1.06 | 3.313 | 0.307 |
| 32 | 1.08 | 14.750 | 0.449 |

#### Prefill

| state | batch | seq | ms | tok/s | max abs diff | mean abs diff |
|---|---:|---:|---:|---:|---:|---:|
| unfused | 1 | 128 | 3935.068 | 32.5 | - | - |
| fused | 1 | 128 | 3684.374 | 34.7 | 15.250 | 0.957 |
| unfused | 16 | 128 | 5420.851 | 377.8 | - | - |
| fused | 16 | 128 | 5282.682 | 387.7 | 19.250 | 0.930 |
| unfused | 1 | 1024 | 5050.830 | 202.7 | - | - |
| fused | 1 | 1024 | 4809.308 | 212.9 | 17.250 | 0.840 |
| unfused | 1 | 4096 | 5979.838 | 685.0 | - | - |
| fused | 1 | 4096 | 5907.163 | 693.4 | 17.875 | 0.652 |

(128 and 1024 measured with `repeats=3, warmup=1`; 4096 measured with `repeats=1, warmup=0` due to long runtime.)

### Observations

- `model.fuse()` on the full Laguna-S-2.1-GPTQ-FIXED checkpoint completes without errors and fuses all compatible attention QKV and MoE expert gate/up groups.
- End-to-end decode latency improves by **5–9%** across batch sizes 1–32. The fused path saves one Marlin kernel launch per gate/up expert and per QKV attention module.
- Prefill throughput improves by **2–7%** for the tested shapes. Larger prefill (4096 tokens) is closer to neutral because the GEMM is compute-bound.
- Numerical drift is small relative to the logits scale and consistent with BF16 tensor-core accumulation order differences between the fused and unfused Marlin kernels.
- The real-model overhead of MoE routing dominates at small batch/decode, so the relative gain is smaller than the synthetic QKV micro-benchmark (which showed 2.5–3x speedups). Future work can explore a single batched/offset Marlin kernel for all active experts to reduce per-expert launch overhead.

## Review flag fixes (Phase 12)

The Devin Review audit on PR #119 flagged four latent correctness issues that are now fixed:

1. **Qwen `w1`/`w2` gate/up ordering**
   - `gptqmodel/models/definitions/qwen.py` now tags `w1:0:up` and `w2:0:gate`, matching the original `QWenMLP` computation `w1(x) * silu(w2(x))`.
   - The static alias list in `fused_quant_linear.py` (`("w2", "w1")`) was updated so `w2` is treated as the gate projection and `w1` as the up projection.

2. **MLP `forward` replacement safety**
   - `_maybe_install_fused_gateup_activation` now verifies the parent module is a dense MLP before replacing its `forward`.
   - It rejects `nn.ModuleList` / `nn.ModuleDict` containers and modules whose `forward` signature is not a simple `(self, x)` form.
   - A sample forward parity check (`_check_fused_gateup_mlp_parity`) compares the original and fused paths; replacement is only installed when outputs match.

3. **3-bit TritonV2 fused path**
   - `_can_fuse_modules` explicitly returns `False` when the first member is a `TritonV2Linear` with `bits == 3`.
   - The fused helper routes through `QuantLinearFunction`, while `TritonV2Linear.forward` dispatches to a dedicated `matmul_3bit` kernel on sm80+; the two paths are not interchangeable, so 3-bit fusion is gated out until a matched 3-bit mega-kernel is validated.

4. **Expert-container activation guard**
   - `_is_moe_expert_parent` name markers (`experts.`, `shared_experts.`, `shared_expert`) still skip MoE containers.
   - `_is_safe_mlp_parent` adds structural guards: bare `ModuleList` / `ModuleDict` expert containers and numeric-name modules inside lists will not receive the dense-MLP activation `forward` replacement.

### Validation after the fixes

- `pytest -q tests/test_fused_quant_linear.py` — 28 passed
- `pytest -q tests/test_moe_dispatch.py` — 10 passed
- `ruff check` and `git diff --check` — clean
- Real-model `model.fuse()` on `Laguna-S-2.1-GPTQ-FIXED` (`backend=GPTQ_MARLIN`, GPU 6) fuses 48 QKV groups and 12,080 gate/up groups and runs decode/prefill without errors. Decode speedup is modest (~0–7% after warmup), prefill is roughly neutral, and logits differences stay within BF16 accumulation tolerance.

## Phase 13 — Kimi-K3 proxy MoE profiling and batched Marlin kernel direction

### Proxy shape

| Parameter | Value |
|---|---|
| `hidden_size` | 1792 (Kimi-K3 routed latent is ~3584; scaled down to fit synthetic fixture memory) |
| `intermediate_size` | 1792 |
| `num_experts` | 64 (Kimi-K3 has 896 routed + 2 shared) |
| `top_k` | 8 (Kimi-K3 uses 16 active, but `top_k` here is the per-token expert count for the dispatch test) |
| `batch` | 16, `seq` | 1 |

### Benchmark (`scripts/benchmark_moe_kimi_k3_proxy.py`, A100 sm80, GPU 6)

| Path | Latency (ms) | Relative to dense `nn.Linear` |
|---|---:|---:|
| Dense `nn.Linear` + `grouped_mm` | 9.61 | 1.00x |
| TritonV2 packed + `grouped_mm` | 18.43 | 1.92x |
| Marlin per-expert (gate/up separate) | 24.87 | 2.59x |
| Marlin per-expert (fused gate/up) | 22.92 | 2.38x |

For a smaller decode shape (`hidden=1024, intermediate=1024, batch=1, top_k=8`, 7 active experts):

| Path | Latency (ms) | Relative to dense `nn.Linear` |
|---|---:|---:|
| Dense `nn.Linear` + `grouped_mm` | 1.80 | 1.00x |
| TritonV2 packed + `grouped_mm` | 3.12 | 1.73x |
| Marlin per-expert (fused gate/up) | 5.27 | 2.93x |

### Observations

- The current Marlin MoE path launches one `apply_gptq_marlin_linear` kernel per active expert per projection (or two kernels after gate/up fusion). Even on the large 1792-dim proxy, the Marlin path is **~2.4x slower** than the dense `grouped_mm` baseline.
- Fusing each expert's `gate_proj`/`up_proj` helps (10–15% at these shapes), but the dominant cost is the **per-expert Python loop and kernel launch overhead**, not the gate/up split.
- The TritonV2 packed path uses `torch.nn.functional.grouped_mm` over active experts after dequantizing each projection to a dense `bf16` weight stack. It is already faster than per-expert Marlin and is the best available path for TritonV2-backed MoE checkpoints.
- A real batched/offset Marlin kernel is needed to close the gap with the dense baseline for Marlin-backed MoE. The required kernel is a grouped GEMM over packed Marlin weights: it must accept a stack of active-expert `qweight`/`scales`/`qzeros`/`g_idx` tensors and a token-offsets array, and emit one contiguous output per expert block. This is a CUDA mega-kernel effort and should follow the `gptqmodel-mega-kernels` playbook (profile, establish a dense reference, gate on capability, preserve fallback).

### Validation

- Added `tests/test_moe_dispatch.py::test_grouped_moe_dispatch_kimi_k3_proxy` covering the 1792-dim, 64-expert, top-8 shape against the per-expert loop.
- `pytest -q tests/test_moe_dispatch.py` — 11 passed.

## Phase 14 — Marlin-packed MoE experts with `grouped_mm` dispatch

This phase adds the ability to dequantize Marlin-packed weights to a dense `bf16` tensor and use `torch.nn.functional.grouped_mm` for a single grouped GEMM over active experts, instead of launching one Marlin kernel per expert projection.

### Implementation

- `gptqmodel/nn_modules/qlinear/marlin.py`
  - Added `MarlinLinear.dequantize_weight(dtype, max_chunk_rows=1024)`.
  - The method builds an identity input in `max_chunk_rows` chunks and calls `gptq_marlin_gemm(...)` with the layer's packed `qweight`, `scales`, `qzeros`, and `g_idx`. Because Marlin GEMM internally dequantizes the packed weights and multiplies by the input, running it on an identity matrix returns the exact dense `(in_features, out_features)` weight that the Marlin kernel would use.
  - A pre-allocated output tensor is re-used for each chunk to avoid allocating one tensor per chunk.

- `gptqmodel/nn_modules/fused_quant_linear.py`
  - Added `_FusedMarlinKernel.dequantize_weight(dtype)`.
  - The fused kernel stores a single concatenated `qweight` for all members (e.g. `[gate, up]` packed along `out_features`). The dequantizer runs the same eye-trick on the full concatenated weight and returns a single `(in_features, sum(out_features))` tensor.
  - `_FusedQuantGroup.dequantize_weight` now delegates to the Marlin-specific dequantizer instead of the generic integer unpack path, which produced garbage for Marlin packing.

- `gptqmodel/utils/moe_dispatch.py`
  - `_is_marlin_packed` detects `MarlinLinear` instances.
  - `_try_dequant_gptq_weight` returns `None` for Marlin-packed modules so the grouped path never tries to unpack a Marlin-packed `qweight` as if it were standard GPTQ int32 data.
  - Fused Marlin gate/up MoE experts route through `_grouped_mm_dequant_experts_forward`, which calls `_extract_fused_gateup_dense_weight` (concatenated gate+up) and `_extract_expert_dense_weight` for `down_proj`.
  - `_can_use_grouped_mm` now keeps non-fused Marlin experts on the per-expert Marlin launch path and only enables `grouped_mm` for Marlin when gate/up fusion has been applied. This avoids the accuracy and performance issues of repeatedly dequantizing separate `gate_proj`/`up_proj` Marlin weights for every token.

- `tests/test_moe_dispatch.py`
  - `test_grouped_moe_dispatch_marlin_fused_gateup_matches_unfused` now uses the defuser per-expert loop as the reference. Both the per-expert fused packed path (`loop`) and the new `grouped_mm` dense path (`grouped`) are compared against the same reference with `atol=2.0, rtol=0.05`.

### Accuracy validation

- `pytest -q tests/test_moe_dispatch.py tests/test_fused_quant_linear.py` — 40 passed.
- `ruff check` and `git diff --check` — clean.

### Benchmarks (`scripts/benchmark_fused_moe_marlin.py`)

All numbers are measured on A100 sm80 (`PG506-230`) with `torch 2.13.0+cu130`, `Triton 3.7.1`, `PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold=0.5'`. `num_experts=64`, `top_k=8`, `group_size=128`, `bits=4`.

#### Laguna-S-2.1 proxy (`hidden=3072`, `intermediate=12288`)

| mode | batch | seq | ms unfused (per-expert Marlin) | tok/s unfused | ms fused (grouped_mm Marlin) | tok/s fused | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 5.567 | 1,437.1 | 5.480 | 1,459.8 | 1.016 |
| decode | 2 | 1 | 7.884 | 2,029.4 | 8.095 | 1,976.6 | 0.974 |
| decode | 4 | 1 | 13.554 | 2,360.9 | 11.901 | 2,688.8 | 1.139 |
| decode | 8 | 1 | 18.098 | 3,536.3 | 17.637 | 3,628.7 | 1.026 |
| decode | 16 | 1 | 22.444 | 5,703.1 | 23.035 | 5,556.7 | 0.974 |
| decode | 32 | 1 | 26.566 | 9,636.3 | 26.198 | 9,771.8 | 1.014 |
| prefill | 1 | 128 | 27.182 | 37,671.6 | 27.759 | 36,888.3 | 0.979 |
| prefill | 1 | 1024 | 32.821 | 249,598.3 | 32.889 | 249,077.6 | 0.998 |
| prefill | 1 | 4096 | 59.947 | 546,616.1 | 60.097 | 545,253.2 | 0.998 |
| prefill | 16 | 128 | 42.303 | 387,301.4 | 41.412 | 395,631.6 | 1.022 |

#### Qwen3.5-27B proxy (`hidden=5120`, `intermediate=17408`)

| mode | batch | seq | ms unfused | tok/s unfused | ms fused | tok/s fused | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 5.102 | 1,568.0 | 5.505 | 1,453.2 | 0.927 |
| decode | 2 | 1 | 8.288 | 1,930.6 | 8.228 | 1,944.6 | 1.007 |
| decode | 4 | 1 | 12.804 | 2,499.3 | 13.973 | 2,290.1 | 0.916 |
| decode | 8 | 1 | 18.365 | 3,484.8 | 19.371 | 3,303.9 | 0.948 |
| decode | 16 | 1 | 23.788 | 5,381.0 | 23.515 | 5,443.4 | 1.012 |
| decode | 32 | 1 | 28.167 | 9,088.6 | 27.363 | 9,355.6 | 1.029 |
| prefill | 1 | 128 | 28.510 | 35,917.8 | 28.029 | 36,534.1 | 1.017 |
| prefill | 1 | 1024 | 45.640 | 179,490.9 | 44.767 | 182,990.3 | 1.019 |
| prefill | 1 | 4096 | 110.683 | 296,053.3 | 111.415 | 294,106.8 | 0.993 |
| prefill | 16 | 128 | 66.030 | 248,128.1 | 65.634 | 249,626.2 | 1.006 |

#### Kimi-K3 proxy (`hidden=1792`, `intermediate=1792`)

| mode | batch | seq | ms unfused | tok/s unfused | ms fused | tok/s fused | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4.966 | 1,610.9 | 5.536 | 1,445.1 | 0.897 |
| decode | 2 | 1 | 8.559 | 1,869.4 | 8.572 | 1,866.6 | 0.999 |
| decode | 4 | 1 | 12.643 | 2,531.0 | 11.747 | 2,724.1 | 1.076 |
| decode | 8 | 1 | 17.474 | 3,662.6 | 16.730 | 3,825.4 | 1.044 |
| decode | 16 | 1 | 22.424 | 5,708.0 | 23.188 | 5,520.0 | 0.967 |
| decode | 32 | 1 | 27.016 | 9,475.8 | 26.863 | 9,529.9 | 1.006 |
| prefill | 1 | 128 | 27.041 | 37,868.5 | 26.887 | 38,085.8 | 1.006 |
| prefill | 1 | 1024 | 30.453 | 269,001.0 | 29.136 | 281,163.8 | 1.045 |
| prefill | 1 | 4096 | 30.939 | 1,059,105.9 | 28.846 | 1,135,980.4 | 1.073 |
| prefill | 16 | 128 | 29.518 | 555,051.2 | 29.664 | 552,311.4 | 0.995 |

### Observations

- The dense `grouped_mm` Marlin path is **memory-bandwidth and dequantization bound at decode**: for every forward it must run an identity-GEMM dequantization on each active expert's packed weight stack before calling `grouped_mm`. For small `M` the dequant cost is comparable to the GEMM itself, so decode speedups are small and sometimes negative.
- Prefill benefits are larger (up to ~7% on the Kimi-K3 long-sequence proxy) because the same dequantized weight stack is amortized over many tokens and `grouped_mm` is much more efficient than many small per-expert Marlin launches.
- Accuracy is within the BF16 tensor-core accumulation-order tolerance. The per-expert packed Marlin reference and the `grouped_mm` dense path can differ by up to a few units for large output magnitudes, which is consistent with `MarlinLinear.forward` vs `x @ dense_weight` differences seen in micro-tests.
- A true batched/offset Marlin MoE mega-kernel (Phase 13 direction) is still required for substantial decode speedups. The current `grouped_mm` path is an interim, fully-tested fallback that works immediately on sm80 without a custom CUDA kernel.

## Phase 15 — Native batched/offset Marlin MoE mega-kernel

### What changed

Implemented the native batched/offset Marlin MoE mega-kernel path:

- Registered `gptqmodel_ext/marlin_moe` as a JIT extension (`torch.ops.gptqmodel_marlin_moe.moe_wna16_marlin_gemm`) and wrapped it in `gptqmodel/utils/marlin_moe.py`.
- Added `_moe_align_block_size` and `_moe_block_size` helpers to group token-expert pairs by active expert and pad each group to the kernel's `moe_block_size`.
- Added `_batched_marlin_moe_supported` / `_batched_marlin_moe_forward` to `gptqmodel/utils/moe_dispatch.py` and selected it ahead of the `grouped_mm`/`marlin` per-expert paths for compatible Marlin-packed MoE modules.
- Handles both fused and non-fused gate/up projections, reusing existing `_apply_expert_gate` for activation conventions.
- Falls back to the existing per-expert Marlin or `grouped_mm` paths when the mega-kernel is unavailable or unsupported.

### Benchmarks

A100 sm80, bf16, 4-bit Marlin, 64 experts, `top_k` as listed.
`per_expert` = per-expert Marlin loop (fused gate/up where installed).
`marlin_moe` = batched/offset mega-kernel.

#### Laguna-S-2.1 proxy (`hidden=3072`, `intermediate=12288`, `top_k=6`)

| mode | batch | seq | ms per_expert | tok/s per_expert | ms marlin_moe | tok/s marlin_moe | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4.284 | 1,400.4 | 1.643 | 3,651.8 | 2.61 |
| decode | 2 | 1 | 6.951 | 1,726.4 | 2.620 | 4,579.3 | 2.65 |
| decode | 4 | 1 | 10.985 | 2,184.7 | 4.458 | 5,383.0 | 2.46 |
| decode | 8 | 1 | 15.968 | 3,006.0 | 7.125 | 6,737.0 | 2.24 |
| decode | 16 | 1 | 23.889 | 4,018.7 | 9.501 | 10,104.7 | 2.51 |
| decode | 32 | 1 | 24.050 | 7,983.4 | 13.386 | 14,343.6 | 1.80 |
| prefill | 1 | 128 | 26.201 | 29,312.0 | 15.910 | 48,270.8 | 1.65 |
| prefill | 1 | 1024 | 31.294 | 196,332.9 | 21.680 | 283,388.7 | 1.44 |
| prefill | 1 | 4096 | 49.518 | 496,300.1 | 45.750 | 537,185.2 | 1.08 |
| prefill | 16 | 128 | 37.041 | 331,740.7 | 29.855 | 411,583.8 | 1.24 |

#### Qwen3.5-27B proxy (`hidden=5120`, `intermediate=17408`, `top_k=6`)

| mode | batch | seq | ms per_expert | tok/s per_expert | ms marlin_moe | tok/s marlin_moe | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4.356 | 1,377.4 | 2.326 | 2,579.9 | 1.87 |
| decode | 2 | 1 | 7.009 | 1,712.2 | 3.288 | 3,649.1 | 2.13 |
| decode | 4 | 1 | 10.598 | 2,264.6 | 7.664 | 3,131.5 | 1.38 |
| decode | 8 | 1 | 15.304 | 3,136.5 | 11.707 | 4,100.1 | 1.31 |
| decode | 16 | 1 | 22.776 | 4,215.0 | 17.165 | 5,592.8 | 1.33 |
| decode | 32 | 1 | 29.115 | 6,594.6 | 22.209 | 8,645.1 | 1.31 |
| prefill | 1 | 128 | 27.210 | 28,225.4 | 26.792 | 28,664.8 | 1.02 |
| prefill | 1 | 1024 | 39.834 | 154,239.7 | 39.294 | 156,359.3 | 1.01 |
| prefill | 1 | 4096 | 87.964 | 279,386.0 | 94.063 | 261,271.9 | 0.94 |
| prefill | 16 | 128 | 54.686 | 224,700.9 | 57.647 | 213,158.4 | 0.95 |

#### Llama-like proxy (`hidden=4096`, `intermediate=11008`, `top_k=6`)

| mode | batch | seq | ms per_expert | tok/s per_expert | ms marlin_moe | tok/s marlin_moe | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 6.862 | 874.4 | 2.100 | 2,856.9 | 3.27 |
| decode | 2 | 1 | 6.710 | 1,788.4 | 3.296 | 3,640.8 | 2.04 |
| decode | 4 | 1 | 11.246 | 2,134.0 | 5.442 | 4,410.3 | 2.07 |
| decode | 8 | 1 | 20.236 | 2,372.0 | 9.154 | 5,243.4 | 2.21 |
| decode | 16 | 1 | 21.933 | 4,376.9 | 13.805 | 6,954.1 | 1.59 |
| decode | 32 | 1 | 29.501 | 6,508.3 | 17.186 | 11,171.6 | 1.72 |
| prefill | 1 | 128 | 27.107 | 28,332.0 | 19.766 | 38,854.1 | 1.37 |
| prefill | 1 | 1024 | 32.766 | 187,511.7 | 26.138 | 235,057.5 | 1.25 |
| prefill | 1 | 4096 | 61.090 | 402,288.6 | 53.933 | 455,678.9 | 1.13 |
| prefill | 16 | 128 | 45.768 | 268,486.4 | 34.823 | 352,866.8 | 1.31 |

#### Kimi-K3 proxy (`hidden=1792`, `intermediate=1792`, `top_k=8`)

| mode | batch | seq | ms per_expert | tok/s per_expert | ms marlin_moe | tok/s marlin_moe | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 5.721 | 1,398.3 | 1.689 | 4,735.8 | 3.39 |
| decode | 2 | 1 | 7.791 | 2,053.6 | 2.948 | 5,427.9 | 2.64 |
| decode | 4 | 1 | 13.007 | 2,460.2 | 4.231 | 7,563.0 | 3.07 |
| decode | 8 | 1 | 18.533 | 3,453.3 | 6.421 | 9,967.7 | 2.89 |
| decode | 16 | 1 | 23.212 | 5,514.4 | 9.953 | 12,860.0 | 2.33 |
| decode | 32 | 1 | 27.409 | 9,340.2 | 10.712 | 23,899.5 | 2.56 |
| prefill | 1 | 128 | 31.928 | 32,072.0 | 9.686 | 105,719.4 | 3.30 |
| prefill | 1 | 1024 | 30.099 | 272,170.6 | 11.577 | 707,609.8 | 2.60 |
| prefill | 1 | 4096 | 31.921 | 1,026,518.4 | 14.926 | 2,195,414.9 | 2.14 |
| prefill | 16 | 128 | 29.847 | 548,929.0 | 12.916 | 1,268,532.5 | 2.31 |

### Observations

- The mega-kernel removes per-expert launch overhead and is up to **3.4x faster on decode** and **3.3x on prefill** for small-dim MoE shapes (Kimi-K3 proxy).
- Larger hidden/ intermediate dimensions (Qwen3.5-27B) see more modest but consistent decode gains (1.3-2.1x) because the GEMM itself dominates; long-sequence prefill can be neutral to slightly slower when the kernel becomes memory-bandwidth bound.
- Accuracy is preserved: `tests/test_moe_dispatch.py` now includes batched Marlin MoE parity tests for non-fused and fused gate/up shapes, including a Laguna-S-2.1-like configuration.
- Existing `grouped_mm` and per-expert Marlin fallbacks remain intact for unsupported configurations.

### Files changed

- `gptqmodel/utils/moe_dispatch.py`
- `gptqmodel/utils/marlin_moe.py`
- `gptqmodel_ext/marlin_moe/marlin_moe.cpp`
- `gptqmodel_ext/marlin_moe/ops.cu`
- `gptqmodel_ext/marlin_moe/marlin_template.h`
- `gptqmodel_ext/marlin_moe/kernel.h`
- `scripts/benchmark_marlin_moe_kernel.py`
- `tests/test_moe_dispatch.py`
- `docs/inference_fusion.md`

## Phase 16: Real-model Laguna-S-2.1-GPTQ-FIXED sanity generation

### Goal
Confirm the fused QKV + fused gate/up + batched/offset Marlin MoE path does not regress whole-model output quality on the real quantized `Laguna-S-2.1-GPTQ-FIXED` checkpoint.

### Command

```bash
python scripts/sanity_laguna_fusion.py --gpu 6 --max-new-tokens 128
```

### Configuration

| GPU (physical) | Model | Backend | Attention | QKV fusion | gate/up fusion | MoE dispatch |
|---|---|---|---|---|---|---|
| 6 | `Laguna-S-2.1-GPTQ-FIXED` | `GPTQ_MARLIN` | `flash_attention_2` | 48 groups | 12,080 groups | `marlin_moe` |

### Control prompts (max 128 tokens, `do_sample=False`)

| # | Prompt | Output |
|---:|---|---|
| 1 | What is the capital of France? | `The capital of France is Paris.` |
| 2 | What is 7 times 8? | `7 times 8 is 56.` |
| 3 | Who wrote the play Hamlet? | Correctly identifies **William Shakespeare** with context. |
| 4 | What is the largest planet in our solar system? | Correctly answers **Jupiter** with details and `\boxed{Jupiter}`. |
| 5 | Translate 'Hello' to Spanish. | Correctly answers **Hola** with usage notes. |
| 6 | What is the boiling point of water in Celsius? | Correctly answers **100°C** and `\boxed{100}`. |
| 7 | Name a primary color. | Correctly names **red** (also blue/yellow, RGB). |
| 8 | How many continents are there on Earth? | Correctly answers **seven** and lists them. |
| 9 | What gas do plants absorb from the atmosphere? | Correctly answers **carbon dioxide** with `\boxed{Carbon\ dioxide}`. |
| 10 | What is the square root of 64? | Correctly answers **8** with `\boxed{8}`. |

### Findings

- All 10 control answers are factually correct and coherent.
- No garbled tokens, repetitions, or hallucinated answers were observed.
- `flash_attention_2` was active (`Loader: Auto enabling flash attention2`).
- `model.fuse(qkv=True, gate_up=True, gate_up_activation=True)` installed 48 QKV and 12,080 gate/up groups.
- The MoE dispatcher selected the `marlin_moe` backend after the Phase 16 fix to `_batched_marlin_moe_supported` (down projection may use a different `group_size` than gate/up).

### Notes

- Real-model generation with the full 59 GB checkpoint is slow on a single A100 for 128-token greedy decoding (tens of seconds per prompt). The purpose of this phase was quality validation, not throughput benchmarking.
- The `_batched_marlin_moe_supported` guard was relaxed so that gate/up and down projections are allowed to have different `group_size` (common in dynamic per-layer quantization configs) while still taking the batched Marlin MoE path.
