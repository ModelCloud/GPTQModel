---
name: gptqmodel-moe
description: Add, debug, or optimize Mixture-of-Experts support in GPT-QModel. Use for MoE routing, top-k dispatch, expert dispatch backends, grouped GEMM, per-expert loops, and QKV/gate-up fusion inside MoE models.
---

# GPT-QModel MoE support

Use when the work touches MoE routers, expert `ModuleList`s, dispatch alignment, grouped GEMM,
or fused gate/up/QKV in MoE models (e.g., Laguna, Qwen3-MoE, Olmoe, Kimi-K3).

## When to use

- Adding a new MoE model family or MoE layer mapping.
- Changing expert dispatch: per-expert, `grouped_mm`, or `marlin_moe`.
- Optimizing routing/top-k alignment (`_moe_align_block_size`, `_moe_block_size`).
- Fusing QKV or gate/up projections while preserving per-expert weight boundaries.
- Debugging wrong outputs, hangs, or high host-side overhead in MoE inference.

## Key files

- `gptqmodel/utils/moe_dispatch.py` (`_batched_marlin_moe_forward`, `_moe_align_block_size`, grouped dispatch)
- `gptqmodel/utils/marlin_moe.py` (JIT `moe_wna16_marlin_gemm` wrapper)
- `gptqmodel/models/base.py` (`model.fuse`, applies QKV/gate-up fusion to MoE-compatible models)
- `gptqmodel/nn_modules/fused_quant_linear.py` (`_is_moe_individual_expert`, `install_fused_*` helpers)
- `gptqmodel/models/definitions/laguna.py`, `qwen3.py`, `olmoe.py`
- `tests/test_moe_dispatch.py`, `scripts/benchmark_fuse_real_laguna.py`

## Workflow

1. **Understand the MoE structure.**
   - Identify `experts`, `shared_experts`, `gate`, `router`, `topk_ids`, `topk_weights`.
   - Check `module_tree` and `moe_alias_specs` for expert indexing and shared-expert routing.

2. **Choose the dispatch backend from runtime facts.**
   - `GPTQ_MARLIN` + uniform W4G64 in a projection: try `moe_wna16_marlin_gemm`
     (`GPTQMODEL_MARLIN_MOE_BACKEND=marlin_moe`).
   - Heterogeneous `group_size`/shapes: fall back to `per_expert` active loops.
   - Dense GPU path: `torch.nn.functional.grouped_mm` (`GPTQMODEL_MARLIN_MOE_BACKEND=grouped_mm`) or per-expert loops.
   - CPU / non-Marlin path: per-expert loops or the generic `BaseQuantLinear` forward.
   - Never infer hardware capability from a CUDA index.

3. **Gate on packed shape homogeneity.**
   - All experts in a projection must share `qweight`/`scales` shape (`group_size`, padded in/out features)
     for one stacked launch.
   - If gate/up and down have different `group_size`, stack gate/up together and validate down separately.

4. **Avoid host-side sync in the hot path.**
   - Replace `nonzero`/`item`/`any` per expert with vectorized `searchsorted`/`_moe_align_block_size`.
   - Pre-stack active weights once per cluster; reuse alignment metadata across gate/up/down.

5. **Fusion rules.**
   - `model.fuse(qkv=True, gate_up=True)` is safe at the attention/MLP parent level.
   - Do NOT fuse per-expert `gate_proj`/`up_proj` inside `experts.N` (it duplicates packed weights and can OOM).
   - Check `_is_moe_individual_expert` before replacing `forward`.

6. **Validate.**
   - `pytest -q tests/test_moe_dispatch.py`
   - Run sanity generation on Laguna-S-2.1/Qwen3.5-27B/Kimi-K3 proxies.
   - Benchmark decode/prefill batch 1, 2, 4, 8, 16, 32 separately and compare against unfused/dense baseline.

## Anti-patterns

- Do not use `getattr(self, str(i))` or Pythonic per-expert loops in hot dispatch; precompute module lists.
- Do not call `.item()`/`.tolist()`/`nonzero()` on routing tensors inside the forward loop.
- Do not fuse individual experts.
- Do not drop `adapter`, `had_K`, `online_full_had`, or `online_partial_had` transforms when dequantizing expert weights.
