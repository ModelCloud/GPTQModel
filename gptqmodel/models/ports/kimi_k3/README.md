# GPT-QModel Kimi K3 / AttnRes local inference

## Ownership and provenance

This package owns the `kimi_k3` multimodal model and the AttnRes layout of
`kimi_linear`. It is adapted from
[`moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06`](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06).
See [NOTICE.md](NOTICE.md) for file attribution, tokenizer hashes and licenses.
The ordinary **non-AttnRes** `kimi_linear` architecture remains the exact native
Transformers 5 `KimiLinearConfig` / `KimiLinearForCausalLM`, not this decoder.

Transformers 5.23 has that native ordinary Kimi-Linear implementation but no
native K3/AttnRes config/task mapping. K3 has a different residual-bank protocol,
split convolution owners and latent `block_sparse_moe.experts.*.w1/w3/w2` weights.
Substituting the native ordinary model would silently change the architecture.

## Raw-config local dispatch

```mermaid
flowchart TD
    A[load_model_config] --> B[LocalModelLoader.load_config]
    B --> C[KimiConfigFactory.from_pretrained: read raw config JSON]
    C --> D{model_type}
    D -->|kimi_k3| E[KimiK3Config: packaged text and vision configs]
    D -->|kimi_linear| F{is_kimi_k3_config on raw markers}
    D -->|other| X[ValueError: unsupported model type]
    F -->|false| G[Exact native Transformers KimiLinearConfig]
    F -->|true| H{Explicit positive attn_res_block_size and no MTP?}
    H -->|no| X
    H -->|yes| I[Packaged KimiLinearConfig]
    E --> J[KimiModelFactory: KimiK3ForConditionalGeneration]
    I --> K[KimiModelFactory: packaged KimiLinearForCausalLM]
    G --> L[Native definition or direct factory: native KimiLinearForCausalLM]
    E --> M[Package-owned TikTokenTokenizer and KimiK3Processor]
    I --> N[KimiTokenizerFactory: package-owned TikTokenTokenizer]
    G --> O[KimiTokenizerFactory: native AutoTokenizer with explicit native config and trust false]
    M --> P[Local tiktoken.model and processor data only]
    N --> P
    P -->|missing or malformed data| Q[Fail without executing checkpoint Python; repair and retry]
```

The selected package loader strips `trust_remote_code` and `code_revision`.
Checkpoint `auto_map` remains ordinary data, including when caller trust and the
existing security bypass are both enabled. No global HF registration, downloaded
Python, checkpoint class imports, or new environment switches are involved.
Tokenizer selection does not normalize generic inputs here: Tokenicer retains
ownership of shared tokenizer normalization.

| Before | After |
| --- | --- |
| Base `22bea1201`, `definitions/kimi_k3.py:26`: `require_trust_remote_code = True`; inherited Auto loader | `definitions/kimi_k3.py:58,93`: definition-owned `LocalModelLoader` descriptors |
| Raw `kimi_linear` could be mistaken for K3 after injecting AttnRes defaults | `configuration_kimi_k3.py:291`: `KimiConfigFactory` inspects raw JSON before any K3 defaults |
| Checkpoint tokenizer selected `tokenization_kimi.TikTokenTokenizer` through `auto_map` | `tokenization_kimi.py`: package-owned PythonBackend subclass; K3 XTML segments retained |
| FLA/Triton and FlashAttention imports prevented portable execution | `eager_kda.py`: Torch reference recurrence/convolution; vision eager/SDPA paths available without those imports |
| Gate parameters could be mistaken for corrupt mixed-dtype state | `modeling_kimi_linear.py:982`: strict FP32 `A_log`/`dt_bias` metadata matches the model's native ownership |

The parent integration owns the central `docs/` reference page, README link and
PR diagram; this port-local page supplies their mechanism and evidence.

## Cache and quantization boundaries

```mermaid
flowchart TD
    A[KimiLinearModel.forward] --> B{Package-owned KimiDynamicCache or None?}
    B -->|foreign cache| X[ValueError before mutating state]
    B -->|valid| C[_causal_mask validates mask before decoder execution]
    C -->|bad mask| X
    C --> D[KimiDecoderLayer: prefix_sum plus block_residual]
    D --> E{KDA or MLA layer?}
    E -->|KDA| F[Split q/k/v convolution states and recurrent KDA state]
    E -->|MLA| G[Rotary MLA keys and values]
    F --> H[Per-layer AttnRes cache continuation]
    G --> H
    H --> I[Final residual-bank reduction and logits]
    J[KimiLinearQModel.quantize or KimiK3QModel.quantize] --> K[Explicit rejection: ordinary looper loses residual-bank state]
    L[awq_get_modules_for_scaling] --> M[Explicit rejection: AWQ relationship not qualified]
```

Only routed `w1/w3/w2` leaves are declared quantizable. Norms, residual projection
owners, attention, routers, latent adapters and shared experts carry explicit
exclusion tags. The standard GPT-QModel looper and AWQ still reject this layout
before capture/mutation; declaring its experts does **not** qualify AWQ folding,
search, replay, packing, or fused execution. The existing bounded
`KimiK3Source` calibration runner is unchanged. No new fusion admission is made.

`module_plan_cases.json` holds the exact proposed replacements for the parent
registry's `KimiK3QModel/gptq` and `KimiLinearQModel/gptq` cases. It adds nine
explicit exclusion groups, leaving the expert groups and their order unchanged.
The shared `fusable_models.json` entries need no structural change.

## Runtime and measured qualification

Validated environment: CPython 3.14t with the GIL disabled, Transformers 5.23.0,
Torch 2.15.0 CPU, and **tiktoken 0.12.0**. K3 needs `tiktoken>=0.12.0` and Pillow
for the qualified image path. There is no FLA, FlashAttention, Triton, or einops
requirement for the portable execution path.

From `tests/`:

```sh
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest \
  test_kimi_k3_local_port.py module_tree/test_kimi_k3_local_port.py \
  test_kimi_k3_support.py module_tree/test_kimi_linear_native_tree.py \
  module_tree/test_kimi_k3_fork_plan.py module_tree/test_kimi_linear_fork_plan.py \
  -q --disable-warnings
```

Measured with the processor trust/bypass matrix: **113 passed, no skips, 10.17s**.
Tests use tiny real CPU models, not mocked modeling kernels. In particular:

- Hostile config/model/tokenizer/processor maps across trust and bypass values.
- Exact native ordinary Kimi-Linear dispatch and raw metadata preservation.
- CPU/meta checkpoint-key ownership for text/multimodal MLA, hybrid and KDA.
- FP32/BF16/FP16 logits and chunked cache parity, padding and cache reorder.
- Independent float64 MLA and KDA transition-matrix oracles, plus MoE oracle.
- Full/low-rank KDA gates, bounded/unbounded decay and MLA projection variants.
- Real image preprocessing, image-conditioned forward, continuation and generation.
- Public GPT-QModel FP32/BF16 loading and exact matched-dtype safetensors reload.
- Real pinned TikToken vocabulary, IDs, UTF-8 decode, XTML and mutable-asset retry.

The real-vocabulary test uses only a locally cached, hash-checked pinned asset;
it explicitly skips if unavailable rather than downloading code or weights.
Here it ran without skips. Base vocabulary size is 163584, with 256 reserved
special IDs. `Hello, 世界!` encodes to `[19180, 11, 220, 2243, 0]`; `[BOS]`, `[EOS]`
and `<|media_pad|>` are 163584, 163585 and 163605. Structural XTML segments use
special tokens while marker-like ordinary user text stays escaped text.

### Red/green evidence

1. `test_public_config_ignores_checkpoint_code` failed for both aliases before
   definition-owned loading; both now select packaged config with trust false.
2. Vision `sdpa` selection raised `KeyError` before adding its portable mapping;
   the real-pixel test now executes vision, text continuation and safetensors
   reload; packed unequal-length SDPA also matches independent eager attention.
3. The public BF16 load had undeclared FP32 gate parameters; the explicit strict
   metadata now permits those parameters and asserts every loaded dtype.
4. Mutable `tiktoken.model` repair could reuse a stale tiktoken cache. Supplying
   the actual local file's SHA-256 as reader cache integrity metadata now makes
   corruption fail and a repaired file succeed at the same path, including the
   real pinned vocabulary. This hash is not a hardcoded vocabulary allowlist.
5. Malformed attention masks and foreign caches fail before state mutation;
   retrying the same valid continuation matches a clean full forward.

The old `test_kimi_k3_support.py` fixture now imports the packaged model rather
than executing checkpoint Python. Its original independent MoE tensor oracle
is retained; separate MLA/KDA float64 oracles avoid self-comparison as numerical
evidence. The native dispatch unit mocks the current `load_model_config` boundary
instead of the obsolete `AutoConfig` boundary.

## Explicit limits

No trained K3 weights, accelerator backend, fused FLA/FlashAttention, multi-GPU,
threaded execution, compilation, long-context accuracy, training, MTP, or video
qualification is claimed. CPU eager KDA is a correctness-oriented Python loop,
not a high-throughput replacement for fused kernels. Only the package-owned
dynamic cache is supported; foreign/static/offloaded cache implementations and
speculative rollback are not qualified. Tiny cache/generation/beam checks are
not a claim of arbitrary beam/search compatibility or trained-model quality.
Accelerator paths retained from upstream require separate actual-device checks.
