<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPT-QModel DeepSeek V2 compatibility

The `deepseek_v2` definition selects this package's config and causal LM directly
with `LocalModelLoader`. Neither caller trust nor the security bypass can replace
them with checkpoint Python. No global Transformers registrations are changed.

## Provenance

- Native Transformers 5.23.0 at
  [`52b080b6416a7897e2af90febeb4dfad22b84d19`](https://github.com/huggingface/transformers/tree/52b080b6416a7897e2af90febeb4dfad22b84d19/src/transformers/models/deepseek_v2):
  the config, decoder/model constructors and model API are subclassed/adapted;
  MLA, rotary math, compressed cache layers, expert execution and generation stay
  native, with package-owned cache publication and derived-buffer lifecycle.
  The installed `modeling_deepseek_v2.py` matched this revision byte for
  byte (SHA256 `8c807b695c6d3835b5035703d6def4a45dfa6296a2b35583064be5f937ee32ec`).
- DeepSeek-Coder-V2-Lite-Instruct at
  [`e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11`](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/tree/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11):
  `configuration_deepseek.py`, `modeling_deepseek.py`, and `config.json` establish
  legacy routing and dense/MoE selection. Inspected as text; never executed.
  Modeling source SHA256:
  `7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c`.
- Both sources use Apache-2.0. Attribution and inherited provenance are preserved
  in [NOTICE](NOTICE) and the Python headers. No checkpoint weights are vendored.

## Architecture and checkpoint contract

Native 5.23 supports the exact DeepSeek V2 MLA architecture. Its documented
configuration, however, types `n_routed_experts` as an integer; its decoder does
not implement legacy periodic `moe_layer_freq`, and its router does not apply
legacy `norm_topk_prob`. This small native-derived implementation preserves:

- Full (`q_lora_rank > 0`) and Lite (`q_lora_rank=None`) Q projections;
  low-rank KV projection, distinct no-position/rotary head widths and compressed
  native KV cache; native RoPE including legacy YaRN metadata.
- A dense MLP when `n_routed_experts=None`. Otherwise a layer is MoE exactly when
  `layer_idx >= first_k_dense_replace` and `layer_idx % moe_layer_freq == 0`.
  Shared experts may be omitted with `n_shared_experts=None`.
- Greedy and group-limited greedy softmax routing. For top-k greater than one,
  `norm_topk_prob=True` normalizes the selected probabilities without routed
  scaling; all other cases multiply by `routed_scaling_factor`, as upstream did.
- Standard dense/checkpoint names and Transformers' fused expert tensor ABI.
  The model's `is_custom_code=False` marker enables the **native serialization
  conversion map** for this ABI. It does not authorize remote code. Safetensors
  round trips and public lazy loading test both fused expert tensors and legacy
  per-expert gate/up/down keys. Defuser exposes per-expert linear modules.

Expert-parallel checkpoints (`ep_size != 1`), non-softmax scoring, unknown top-k
methods, invalid expert/group ranges, nonpositive MoE frequencies and invalid
scaling factors fail explicitly. Load a consolidated all-expert checkpoint.
Other architectural limitations remain those of native Transformers 5.23; this
package does not add new attention kernels or distributed expert execution.

The module plan declares both low-rank norms and the router as nonquantized,
and declares Q/K/V/O and gate/up/down roles. No projection fusion is qualified.
AWQ always uses the existing **plain quantization** fallback: MLA crosses
low-rank normalization and split/expanded attention states, while the MoE router
and shared experts add consumers. No scale-search/folding safety is claimed.

```mermaid
flowchart TD
    A[DeepSeekV2QModel.loader] --> B[LocalModelLoader: packaged config and model]
    B --> C{DeepseekV2Config validation}
    C -->|invalid or ambiguous| D[ValueError before weight loading]
    C -->|valid| E[DeepseekV2DecoderLayer]
    E --> F[Native DeepseekV2Attention: MLA and cache]
    E --> G{experts set, layer past dense prefix, frequency matches?}
    G -->|yes| H[DeepseekV2Moe: legacy top-k normalization]
    G -->|no| I[Native DeepseekV2MLP]
    H --> J[Native fused experts and optional shared experts]
    B --> K[Native checkpoint conversion map]
    K --> L[Defuser: per-expert modules for quantization]
    L --> M[AWQ: empty scaling groups, plain quantization]
```

Previously the definition required remote trust and relied on checkpoint model
selection. It now owns class selection. Before/after implementation references:
the old `DeepSeekV2QModel.require_trust_remote_code` versus its `loader`;
native `DeepseekV2TopkRouter.forward` versus this package's override;
native `DeepseekV2DecoderLayer.__init__` versus the periodic/dense selection here.
Shared security-loader and central flow documentation remain integrator-owned.

## Cache publication and derived RoPE state

`DeepseekV2Model.forward` forks a caller's ordinary native `DynamicCache` before
running the native decoder. The fork copies the cache and layer metadata; native
`DynamicLayer.update` concatenates into new tensors, so the prefix tensors need
no extra clone. Continue with **`outputs.past_key_values`**, including the cache
in tuple outputs. The caller's input remains unchanged on success and on decoder,
head or loss failure. A failed suffix can be retried against the same prefix.
Native DDP cache conversion, crop, batch selection/repetition, reorder, reset and
replay remain native cache operations.

Only non-offloaded native `DynamicCache` with ordinary `DynamicLayer` entries is
qualified. Static, sliding, quantized, offloaded and arbitrary subclass caches
raise `TypeError` before decoder execution. Incomplete or unequal-length decoder
prefixes raise `ValueError`; an empty lazy native cache is accepted. This port
does not claim compiled static-cache or accelerator/offloaded-cache support.

`DeepseekV2RotaryEmbedding._apply` and the model initialization hook regenerate
`inv_freq` and `original_inv_freq` in FP32 from the pinned native RoPE recipes.
They never promote an already rounded FP16/BF16 frequency table. Dynamic RoPE
retains its grown sequence length and native short-position reset behavior; YaRN
retains native scaling. The derived tables remain nonpersistent. Learned tensors
retain normal PyTorch casting and checkpoint behavior without repair or rewrite.

| Before (at `da3ecd85c`) | Shipped replacement |
| --- | --- |
| `modeling_deepseek_v2.py:45-52`: native selected scores scaled before dividing and normalizing; extreme finite factors produced NaNs | `DeepseekV2TopkRouter.forward`: FP32 logits/softmax, native group mask/top-k index selection, then selected-probability normalization without scaling; other branches stay native |
| `modeling_deepseek_v2.py:102-132`: inherited model/CausalLM forward mutated caller cache before head/loss completed | `DeepseekV2Model.forward` and `_fork_cache`: validate and fork metadata, publish continuation only through public outputs; failures leave caller prefix reusable |
| `modeling_deepseek_v2.py:114`: native rotary buffers followed parameter casts and lost FP32 frequency precision | `DeepseekV2RotaryEmbedding._refresh_frequencies`, `_apply`, `_NativeCheckpointLayout._init_weights`: regenerate nonpersistent FP32 tables on the destination device from native config |

```mermaid
flowchart TD
    A[DeepseekV2Model.forward: past_key_values] --> B{_fork_cache: supported native cache?}
    B -->|unsupported type or offloading| X[TypeError before decoder]
    B -->|incomplete or nonuniform prefix| Y[ValueError before decoder]
    B -->|supported| C[Copy cache and layer metadata; retain prefix tensors]
    B -->|None| D[Native forward: optional new DynamicCache]
    C --> D
    D --> E[Native DynamicLayer.update: new concatenated tensors]
    E --> F[Decoder output; native CausalLM head and loss]
    E -->|decoder fails| G[Propagate error; caller prefix unchanged]
    F -->|head or loss fails| G
    F -->|success| H[Public output carries continuation cache]
    H --> I[Caller continues with outputs.past_key_values]
    G --> J[Retry suffix with original caller prefix]
    classDef failure fill:#fee,stroke:#b00
    class X,Y,G failure
```

The normalization formula is pinned to `MoEGate.forward` in the inspected
DeepSeek source above: when `top_k > 1 and norm_topk_prob`, divide selected
probabilities by their sum plus `1e-20` and do not multiply by the scaling factor.
Native 5.23's `DeepseekV2TopkRouter.forward` supplies the group-mask and top-k tie
semantics. Native `DeepseekV2RotaryEmbedding`, `ROPE_INIT_FUNCTIONS` and
`dynamic_rope_update` supply frequency construction, growth and reset semantics.

## Qualification

Run from the repository root:

```sh
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest tests/test_deepseek_v2_local_port.py tests/module_tree/test_deepseek_v2_local_port.py -q --tb=short
```

These use tiny real CPU/meta models, independent softmax/group/top-k routing,
native numerical and cache/generation parity, causal masking, dense and periodic
MoE schedules, optional shared experts, YaRN configuration, actual Defuser,
checkpoint tensor ownership and native conversion mappings, exact lazy-shell
materialization, safetensors round trips, malicious `auto_map` under both trust
values and both bypass states, corrupted-load/corrected-retry transitions, and
actual AWQ plain quantization with router/norm preservation. The independent
model-specific registry case pins all nonquantized paths and fusion exclusion.

Initial port qualification used Python 3.14.6 free-threaded, Transformers 5.23.0 and CPU Torch
2.15.0: **53 passed** in 5.97 seconds. In a disposable checkout, assigning the real native
router forward to the compatibility router made both normalized top-k cases
fail (2 failed / 6 passed); assigning the real native decoder constructor made
the dense/periodic variants fail (3 failed / 1 passed); disabling the native
checkpoint-layout declaration broke all four safetensors conversion cases.
The same tests pass with the shipped implementation. The original definition
also failed all five initial ownership/AWQ tests before the plan/policy fix.

The affected registry/AWQ suite with proposed integrator-owned expectation
updates passed **583 tests, 6 skipped** in 15.01 seconds in that checkout. This
includes `test_mla_awq_quantization_completes`, which runs public plain AWQ
quantization. Without the proposed shared updates, four assertions still expect
the previous DeepSeek plan/folding policy. No shared fixture or AWQ test changes
are part of this port's commit.

The router/cache/RoPE follow-up adds **46 cases** (99 owned cases total). In an
isolated checkout of `da3ecd85ce644778197bca9ec9f8f3968486a143`, the unchanged
new tests produced **32 failed, 14 passed, 43 deselected** in 5.93 seconds:

- Four extreme-scale normalized router cases returned nonfinite weights; the
  two scale-1.0 controls passed. The independent oracle pins both routing modes.
- Sixteen cache cases detected decoder/head/loss partial mutation, input-cache
  mutation on success, or failure to reject unqualified caches. Retry checks
  compare logits and every cache layer against a fresh five-token forward.
- Twelve post-construction RoPE cast cases failed public generation or numerical
  assertions. For default BF16 Lite, identical saved tensors generated suffix
  `[10, 8]` before reload and `[17, 7]` after reload. Twelve direct low-dtype
  construction controls passed. The fixture uses `initializer_range=0.2` so
  attention differences survive low-precision output rounding.

Exact regression selector (run against the old modeling file for red, and the
shipped modeling file for green):

```sh
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest tests/test_deepseek_v2_local_port.py -k 'normalized_routing_ignores or public_cache_failure or native_cache_conversion_fork or empty_native_cache or unsupported_cache_fails or rope_precision_cast' -q --tb=short
```

The affected suite also passed on parent
`569c693de` with just the owned follow-up applied: **744 passed, 11 skipped** in
22.77 seconds using:

```sh
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest tests/test_deepseek_v2_local_port.py tests/module_tree/test_deepseek_v2_local_port.py tests/module_tree/test_registered_module_plans.py tests/test_awq_scaling_groups.py tests/test_awq_mla_scaling.py tests/test_awq_moe_router_scaling.py tests/test_local_model_loading.py tests/test_local_model_paths.py tests/test_checkpoint_local_model.py -q --tb=short
```

All owned cases pass. Skips are nine unrelated AWQ fixtures and two local Llama
recovery cases; none qualify DeepSeek. A separate broad leaf-ownership audit on
parent `023d0406b` found two unrelated `lfm3` / `lfm3_vl` ownership failures
(`conv.conv`, `self_attn.g_proj`), both reproduced on that clean parent. No
shared files are changed by this follow-up. Central documentation remains the
integrator's responsibility.

No trained multi-GB checkpoint, accuracy/perplexity run, GPU, FlashAttention,
distributed experts, or complete packed quantized checkpoint export/inference
was qualified. Actual low-precision accelerator numerics and real trained
DeepSeek-V2/Coder-V2 checkpoint evaluation remain separate validation work.
