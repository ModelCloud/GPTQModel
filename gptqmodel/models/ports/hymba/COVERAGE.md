# Hymba qualification

Environment: Python 3.14.6 free-threaded (`sys._is_gil_enabled() == False`),
Transformers 5.23.0, Torch 2.15.0+cpu. No trained multi-GB tensor payload, NVIDIA
GPU, ROCm GPU, FlashAttention, Mamba SSM or causal-conv extension was required.
Results below qualify the Torch reference implementation on CPU.

## Owned regressions

`python -s -m pytest tests/test_hymba_local_port.py tests/module_tree/test_hymba_local_port.py -q -x`
completed with **69 passed, zero skips in 5.16s**.

| Test node (under the files above) | Verified behavior |
| --- | --- |
| `test_definition_owns_selection_without_remote_trust` | Packaged loader and no remote trust requirement; red on BASE (AutoModel loader) |
| `test_definition_declares_actual_checkpoint_ssm_owners` | Explicit SSM ownership/exclusion; red on BASE (undeclared A/D and scan projections) |
| `test_actual_loader_ignores_malicious_auto_map` | Four API trust/bypass combinations; real config, meta shell and safetensors load; malicious config/model markers stay absent |
| `test_public_gptqmodel_consumer_uses_real_local_model` | Four combinations through actual `GPTQModel.load`, real tokenizer and model; exact logits match original fixture |
| `test_scan_matches_separate_float64_scalar_recurrence_with_nonzero_state` | Three sequence lengths, independent scalar FP64 scan with nonzero initial state, dt bias/softplus, D skip and SiLU gate |
| `test_full_hybrid_computation_matches_independent_float64_oracle` | Nine complete numerical layouts using direct checkpoint tensor equations, manual convolution, FP64 attention/recurrence/MLP; includes fixed and dynamic NTK |
| `test_short_prefill_multitoken_and_decode_continuation_match_full_forward` | Eight layouts; one-token prefill, multi-token continuation, decode, memory boundaries, producer-only KV, nonzero SSM state |
| `test_low_precision_continuation_retains_dtype_and_fp32_oracle` | Separate CPU FP16/BF16 continuation and FP32 comparison |
| `test_generate_greedy_and_beam_cache_reordering` | Actual GenerationMixin greedy output matches full recompute; beam generation and exact KV/conv/SSM reordering |
| `test_partial_failure_invalidates_cache_and_reset_allows_exact_retry` | Layer/head/loss failure after live prefill; poisoned retries rejected; reset/replay matches clean output |
| `test_safetensors_save_reload_resave_equivalence` | Tied/untied weights; no pickle emission; config, tensors, logits and subsequent saves match |
| `test_padding_causality_and_inputs_embeds` | Left padding, suffix causality, IDs vs embeddings, memory variants and padded cache continuation |
| `test_cache_rejects_cross_request_ownership_and_unsupported_rollback` | Cross-model use rejected; reset makes reuse legal; crop and legacy conversion rejected |
| `test_cache_rejects_other_thread_before_any_tensor_access` | Second-thread transaction denied before Torch calls; original request remains usable |
| `test_no_mlp_router_output_does_not_invent_experts_or_break_continuation` | No-MLP variant returns no fictitious routers/auxiliary loss and preserves continuation |
| `test_awq_actual_plain_fallback_consumes_all_declared_projections_without_folding` | Real AWQProcessor quantizes/stages all declared projections; non-target weights unchanged, finite native outputs; four layouts |
| `test_actual_calibration_worker_preserves_native_layer_masks_and_shared_kv` | Real calibration worker matches every native layer for two batches, contiguous and gapped KV groups, global-to-local mask transitions |
| `test_published_1_5b_state_dict_matches_every_checkpoint_key_and_shape` | Exact 611-key published checkpoint inventory, 32 real meta layers, tied lm-head alias audited separately |
| `test_all_layout_weight_owners_survive_actual_defuser_conversion` | Nine real meta layouts; deepest owners, quantization/exclusion tags, no SSM omissions after actual Defuser conversion |
| `test_registry_aliases_moe_hooks_and_looper_consumers` | Actual registered alias, expert roles, lifecycle block/expert resolution and looper grouping |
| `test_explicit_integrator_registry_fixture` | Exact proposed central plan fixture, including ordered exclusions and wildcard experts |

## Red/green evidence

The calibration-worker regression failed before the definition replay hooks:
48/144 elements mismatched (maximum absolute difference 0.1745944) when the
first global layer's `use_swa=False` was incorrectly reused for a local layer.
The two native worker cases pass after per-layer policy and source-frontier
selection. No modeling kernel is mocked in these tests.

The expanded public loader matrix exposed the shared tokenizer-config issue
with API trust and bypass both enabled: checkpoint `evil_config.py` executed
and raised. Canonical parent fix `5423ad629` supplies the selected config to
HF's tokenizer loader; after that prerequisite all four public combinations
pass. That prerequisite is not owned by this port.

The affected suites were also run together:

```bash
python -s -m pytest tests/test_hymba_local_port.py \
  tests/module_tree/test_hymba_local_port.py tests/test_awq_scaling_groups.py \
  tests/test_awq_weight_mean.py tests/test_local_model_loading.py \
  tests/test_local_model_paths.py tests/test_looper_helpers.py -q -x -rs
```

Result: **293 passed, 33 skipped, 8 deprecation warnings in 20.41s**. Of the
skips, 27 require unavailable CUDA devices and six are unrelated unavailable
checkpoint/architecture fixtures (Pangu twice, Instella, K2-Horizon, OPT350,
MuseGlimmer). No Hymba case skipped. Ruff and `git diff --check` pass.

The exact central registered-plan consumer contract passed for all 206 aliases
plus its two admission contracts when the explicit port-local Hymba fixture
was overlaid **in memory only**; no shared fixture was edited. This validates
the proposed integration entry rather than claiming the old central JSON
already matches the new definition.

## Integration

The parent integrator owns central fixtures, `docs/`, README and packaging.
Replace `HymbaQModel/gptq` in
`tests/module_tree/fixtures/registered_module_plans.json` with the exact object
in `module_plan_fixture.json`. `models.hymba` already maps to `HymbaQModel`.
The fusable inventory exclusion remains correct: no separately qualified,
non-expert QKV or gate/up group. If Hymba is added to the central AWQ layer
fixture, include `hymba` in `PLAIN_QUANTIZATION_FAMILIES`.

The runtime package is discovered by `setuptools.find_packages()`. Include
`LICENSE.pdf`, `NOTICE`, and these provenance/coverage documents in shared
source/wheel manifest rules. The port's Python runtime does not read the
checkpoint-shape, config or registry JSON fixtures. Central mechanism docs
can link or incorporate the shipped [flow reference](README.md); central
README must link the mechanism page as required by repository guidance.

## Limits

- Full trained `nvidia/Hymba-1.5B-Instruct` logits, quality, evaluation scores,
  long-context behavior and quantized-checkpoint reload remain unqualified.
  Only its actual config and safetensors header were retrieved.
- The reference recurrence loops over sequence positions and retains full KV.
  It is intended for correctness/portability, with no optimized throughput
  claim for the 1.5B checkpoint.
- No optimized Mamba/causal-conv, Flex/FlashAttention, NVIDIA/ROCm kernel,
  multi-device placement, offload, or free-threaded GPU safety claim is made.
  The cache test qualifies Python locking, not concurrent GPU execution.
- Training/gradients, padding-aware MoE auxiliary loss and dynamic-NTK cache
  continuation across a changing base were not qualified. Dynamic NTK full
  forward is compared to an independent oracle; the published model uses
  fixed NTK.
- Attention probabilities, static/compiled caches, crop/rollback, legacy
  KV-only conversion and speculative rollback have no supported implementation
  and reject explicitly. Use a fresh cache for replay.
- AWQ scale optimization/folding and GPU packing are not qualified. Plain
  quantization and staging execute the real CPU processor without fake kernels.
