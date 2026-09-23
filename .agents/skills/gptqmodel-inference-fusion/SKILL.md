---
name: gptqmodel-inference-fusion
description: Add or debug fused QKV, gate/up, and activation fusion for quantized inference. Use for model.fuse(), flash_attention_2, backend-specific fused kernels, and docs/model_inference_optimize.md style decode/prefill benchmarking.
---

# GPT-QModel inference fusion

For QVQ optimization, first read [qvq-kernel-accuracy](../qvq-kernel-accuracy/SKILL.md) for accuracy-preserving math
and the locked numerical contract; apply it before selecting lower precision or ranking performance candidates.

Use when optimizing quantized inference through weight concatenation, fused forwards, or `flash_attention_2`,
especially on Laguna, Qwen3.5-27B, and Kimi-K3 proxy shapes.

## When to use

- Implementing or calling `model.fuse(qkv=True, gate_up=True, gate_up_activation=True)`.
- Fusing `q_proj`/`k_proj`/`v_proj` into a single GEMM and slicing outputs.
- Fusing `gate_proj`/`up_proj` into a single GEMM and applying `SiLU(gate) * up`.
- Comparing fused vs unfused prefill/decode throughput batch 1-32.
- Diagnosing `torch.compile` failures, recompile limits, or cache-key issues.

## Key files

- `gptqmodel/models/base.py` (`model.fuse`)
- `gptqmodel/nn_modules/fused_quant_linear.py` (`_FusedQuantGroup`, `install_fused_qkv`, `install_fused_gate_up`)
- `gptqmodel/utils/moe_dispatch.py` (MoE dispatch used after fusion)
- `gptqmodel/nn_modules/triton_utils/kernels.py` (`fused_silu_mul`, backend kernels)
- `docs/inference_fusion.md`, `docs/model_inference_optimize.md`
- `scripts/benchmark_fuse_real_laguna.py`

## Workflow

1. **Verify members are fusible.**
   - Same backend (`TritonV2Linear`, `MarlinLinear`, etc.), same `in_features`, same `bits`, `group_size`,
     `desc_act`, `sym`, `pack_dtype`.
   - No per-member `adapter`, `online_full_had`, or rotation that cannot apply to the concatenated tensor.
   - QKV members must be `q_proj`, `k_proj`, `v_proj` inside one attention module; gate/up must be `gate_proj`,
     `up_proj` inside one MLP.

2. **Concatenate packed weights along output dim.**
   - The backend-specific fused kernel builds contiguous `qweight`/`scales`/`qzeros` from the member modules.
   - Share `g_idx` from the first member.
   - Record member `out_features` partition sizes (or the resulting `slices` list) for output slicing.

3. **Replace member `forward`s to slice the fused output.**
   - Member 0 computes the fused GEMM, caches slices, returns its slice.
   - Sibling members return cached slices and clear the cache after use.
   - For gate/up activation fusion, compute `SiLU(gate_slice) * up_slice` in one kernel if the backend supports it.

4. **Choose attention.**
   - Prefer the fastest numerically accepted attention implementation actually supported on the live A100+ device and runtime. Do not hard-code FlashAttention-2 as universally best: Hopper/Blackwell runtimes may expose newer kernels, and shape/KV-cache regimes change the winner.
   - Keep eager/SDPA fallback for unsupported shapes or devices.

5. **Validate numerics first, speed second.**
   - Compare fused output vs dense reference `x @ [W_0; W_1; W_2]` and vs three separate modules.
   - Use tight tolerance for FP32, relaxed for BF16/FP16 tensor-core accumulation order.
   - Run `pytest -q tests/test_fused_quant_linear.py`.

6. **Benchmark decode/prefill separately.**
   - Batch sizes 1, 2, 4, 8, 16, 32.
   - Report `new tok/s` (decode) and `tok/s` (prefill) separately.
   - Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` and an idle GPU gate.

## Anti-patterns

- Do not fuse across different `group_size`/`bits`/backends.
- Do not fuse per-expert modules inside MoE `experts.N`.
- Do not rely on `torch.compile` to optimize a Pythonic cache keyed on string UUIDs; it recompiles per value.
- Do not report throughput for `seq=1, max_new_tokens=1` as new-token decode without halving.

## See also

- [Curated GPU performance engineering resources](references/wafer-gpu-perf-resources.md) — External reading list for inference engines from wafer-ai's performance engineering index.


## A100+ fusion performance rules

- Fusion is a data-movement and launch-cost optimization, not a goal by itself.
  Use Nsight Systems to prove launch/materialization cost exists and Nsight
  Compute to ensure the fused resource union does not destroy residency.
- Preserve strided fused outputs when downstream kernels can consume them
  efficiently. Copying Q/K/V or gate/up slices to contiguous buffers can erase
  the benefit of the fused producer.
- When multiple fused children share the same input, stage/reuse that input once
  only when the shared/register footprint and synchronization are cheaper than
  re-reading it through cache. Measure the complete operator.
- A fused rank/adapter projection stored as `A[K,R]` may be ideal for a vector
  load that consumes all R values per K, but poor for a warp fixed on rank. Pick
  the execution layout to match warp ownership; an immutable transposed cache can
  be preferable to strided per-lane loads.
- On Ampere use `cp.async` pipelines only when enough compute hides the copy.
  On Hopper use TMA/WGMMA only when the descriptor/pipeline overhead amortizes.
  On Blackwell re-evaluate ownership around TCGen05/TMEM rather than porting the
  Hopper schedule unchanged.
- Do not introduce cross-stream overlap without proving dependencies and adding
  events. Concurrent kernels can contend for Tensor Cores/L2/HBM and regress even
  when the timeline visually overlaps.
