# Quantization Runtime Sharing

This document describes the calibration-time sharing paths used by GPT-QModel to avoid duplicated work inside same-input module groups.

Same-input groups are modules in the same execution subset that consume the exact same activation tensor. Common examples are:

- attention projections: `q_proj`, `k_proj`, `v_proj`
- MLP projections: `gate_proj`, `up_proj`

The sharing lifecycle is driven by `LoopProcessor.prepare_subset()` and `LoopProcessor.cleanup_subset()`. The subset stage calls these generic hooks around forward capture and worker processing, so each quantization method can install only the temporary state it needs.

## GPTQ Same-Input Hessian Sharing

GPTQ builds a Hessian-like `X^T X` statistic from calibration activations. For same-input modules with the same input width and Hessian configuration, this statistic is identical even when the modules have different output rows.

GPT-QModel therefore shares:

- batch-level Hessian accumulation
- merged per-device Hessian materialization
- inverse/Cholesky preparation used during quantization

The implementation is intentionally conservative:

- only plain `GPTQ` tasks participate
- `GPTAQ` and `FOEM` are excluded because they maintain additional method-specific statistics
- embedding modules are excluded because they use token-frequency diagonal state instead of dense Hessian state
- modules must have matching input-column count and matching Hessian settings
- the runtime cache key includes the actual source activation tensor view and batch index

The main implementation points are:

- `gptqmodel/looper/gptq_processor.py`: groups compatible modules in `prepare_subset()`, caches batch accumulation, and owns the shared inverse cache
- `gptqmodel/quantization/gptq.py`: records shared batch ownership, materializes shared Hessians, and reuses inverse/Cholesky results
- `gptqmodel/quantization/config.py`: exposes `enable_shared_hessian_cache`, enabled by default on `GPTQConfig`
- `tests/test_gptq_shared_hessian.py`: validates q/k/v sharing, sample accounting, materialized Hessian pointer reuse, inverse-cache reuse, disabled-cache behavior, and exact enabled/disabled quantization math

## AWQ Same-Input Activation Sharing

AWQ does not use Hessian accumulation. Its repeated work for same-input groups is activation handling during scale search: the input is copied to CPU for later replay and `abs(input).mean(dim=0)` is reduced chunk-by-chunk.

GPT-QModel shares:

- the captured activation CPU copy for identical same-batch inputs
- the chunked per-channel activation mean, `x_mean`

GPT-QModel deliberately does not share:

- weight-dependent AWQ scale search
- clipping search
- final replay tensors passed to `apply_scale()`

The replay tensor boundary matters because AWQ scaling mutates `input_feat_dict` values in place. The capture cache may store one shared tensor, but folded replay tensors are kept independent before scaling so q/k/v or gate/up modules cannot accidentally scale each other's inputs.

The main implementation points are:

- `gptqmodel/looper/awq_processor.py`: manages per-subset activation-copy sharing, per-layer `x_mean` sharing, and alias-safe replay tensor folding
- `gptqmodel/quantization/config.py`: exposes `enable_activation_x_mean_cache`, enabled by default on `AWQConfig`
- `tests/test_awq_shared_activation.py`: validates CPU copy dedupe, CUDA capture-to-CPU dedupe on GPU 6/7 by default, `x_mean` reuse, variable-length no-alias behavior, disabled-cache behavior, and exact enabled/disabled activation math

## AWQ Scale-Search Weight Restore

AWQ evaluates a grid of candidate scales. Each candidate temporarily mutates the inspected linear weights, pseudo-quantizes them in place, scores reconstruction loss, and then restores the original weights before the next ratio.

GPT-QModel keeps a pristine master weight copy for that restore step. By default, `AWQConfig(scale_search_gpu_weight_restore=True)` keeps the master copy on GPU when free-memory headroom is sufficient, avoiding repeated CPU-to-GPU restores across the ratio grid. If headroom is low, or if the toggle is disabled, the processor uses the lower-VRAM CPU master-copy path.

This optimization does not change AWQ search math: it only changes where the pristine restore copy lives.

## Operational Notes

These optimizations are enabled by default and scoped to one subset/layer lifecycle. They are released by `cleanup_subset()` to avoid retaining large calibration tensors after worker processing.

Use process-level quantization config to disable a sharing path for A/B validation or debugging:

- `GPTQConfig(enable_shared_hessian_cache=False)`
- `AWQConfig(enable_activation_x_mean_cache=False)`
- `AWQConfig(scale_search_gpu_weight_restore=False)`

Disabling any toggle restores the corresponding lower-sharing or lower-VRAM path without changing the quantization algorithm.

Use the exposed test counters when benchmarking or debugging:

- `GPTQProcessor.shared_hessian_stats()`
- `AWQProcessor.shared_activation_stats()`

The counters distinguish requests, hits, and misses so a benchmark can verify that the optimization is active in addition to measuring wall time and peak memory.
