# Converted DBRX Transformers 5 port

This is GPT-QModel's package-owned **converted** DBRX implementation. It is not
the native Transformers `dbrx` architecture: those original checkpoints have
fused QKV and packed expert weights. The converted `dbrx_converted` checkpoints
have separate Q/K/V projections and `ModuleList` experts containing `w1/v1/w2`.
The existing `definitions/dbrx.py` placeholder directs users to ModelCloud's
converted repositories. Transformers 5.23.0 has native `dbrx`, but no
`dbrx_converted` config or causal-LM mapping.

## Pinned source and attribution

| Checkpoint/source | Inspected immutable revision |
| --- | --- |
| [ModelCloud/dbrx-base-converted-v2](https://huggingface.co/ModelCloud/dbrx-base-converted-v2/tree/8ba7c19c8ee5b8d995ea3736cab6fcbf93054d67) | `8ba7c19c8ee5b8d995ea3736cab6fcbf93054d67` |
| [ModelCloud/dbrx-instruct-converted-v2](https://huggingface.co/ModelCloud/dbrx-instruct-converted-v2/tree/8ae889542f05605a028d09286cf9d1f99c50f0eb) | `8ae889542f05605a028d09286cf9d1f99c50f0eb` |

The source files `configuration_dbrx.py`, `modeling_dbrx.py`, `config.json`,
`model.safetensors.index.json`, `LICENSE.txt`, and `NOTICE.txt` were downloaded
as text and inspected, **not imported or executed**. Both source implementations
and both weight indexes are byte-identical; the base/instruct configs differ in
router jitter and explicit/default SiLU activation. No trained tensor shards were
downloaded. Copyright Databricks, Inc.; DBRX is provided under the Databricks Open
Model License. The original [license](LICENSE.txt) and [notice](NOTICE.txt) are
preserved verbatim. The source identifies its rotary and auxiliary-loss equations
as derived from Llama and Mixtral respectively. ModelCloud's changes adapt the
converted implementation to current Transformers APIs and package-local loading.

| Shared upstream file | SHA-256, identical in both revisions |
| --- | --- |
| `configuration_dbrx.py` | `29344455f9fe3ae8a2544b625efa65e6a8e3628ea61d9506bb7f65edb05799dc` |
| `modeling_dbrx.py` | `c4e09b87e52f2df8a8bb0ef569d93e19f066800e61720ac5ddedb2ac16309a18` |
| `model.safetensors.index.json` | `f0fdff8dd07c6a9ac0be7b03c65617422f0e6f13e276b1cb73b90e08cb6693bf` |
| `LICENSE.txt` | `516246c751d9d7fd329f66f7d90f48688aaa5959b6af1bc87ed7bf0d181d8293` |
| `NOTICE.txt` | `bd164c296690cd021db9cd3ba620c151b550c19fc1933c266b3d55adeaf754f1` |

Both published indexes have exactly **2,203** keys. SHA-256 of their sorted keys
joined by `\n` without a trailing newline is
`6cbda8bf37a5d87aa14c21e72ec5f5a2889e11f9a7267828b7d848ef53036b51`.
The module-tree test independently reconstructs this key set and checks a real
meta model. Both full published configs were also instantiated on meta and their
state dictionaries compared directly to the downloaded indexes, without key
conversion. Rotary `inv_freq` is nonpersistent, as in the converted source.

## Adaptation and ownership

- Keep `transformer.wte`, `transformer.blocks`, `transformer.norm_f`, `lm_head`.
- Preserve split attention, clipping, grouped KV heads and full-head rotary.
- Preserve two frozen block norms and the frozen probability-valued router.
- Preserve routed SiLU/GELU experts and configurable top-k/normalization.
- Use current `PreTrainedConfig`, `GenerationMixin`, masking and `Cache` APIs.
- Keep post-rotary keys in caller-owned dynamic or static caches.
- Preserve FP32 LM-head outputs and the source's second auxiliary-loss softmax.
- Index auxiliary-loss padding masks by current physical cache positions.
- Invalidate caches on post-write failures; require reset and full-prefix refill.
- Write dense and packed checkpoints as safetensors.

Runtime code imports only package-local config/model classes and installed Torch
and Transformers APIs. It never downloads Python, registers a global HF mapping,
or delegates to checkpoint `auto_map`. Package ownership wins with API trust and
the existing security bypass either enabled or disabled. Tuple caches are
explicitly rejected; use current Transformers cache objects. A supplied cache
with `use_cache=False` is rejected. Auxiliary router-loss output requires a 2D
padding mask; ordinary forward also accepts 4D additive attention masks.

## Definition-owned loading and failure paths

Before/after evidence is relative to the pre-port definition at
`22bea120181e739161ee2969b260fe94f5b4e791`.

| Mechanism | Before | After |
| --- | --- | --- |
| Class selection | `definitions/dbrx_converted.py:10-11` required checkpoint custom code | `definitions/dbrx_converted.py:11-14` declares the exact local config/model |
| Model layout | Checkpoint Python owned split projections and experts | Package code owns those exact weights, without converting to original DBRX |
| AWQ scaling | Inherited generic source inference | `definitions/dbrx_converted.py:22-31` disables unused root capture and all scale groups |
| Cache continuation with auxiliary loss | Full-history padding mask mismatched current router tokens | The LM head selects current physical cache positions; attention still receives full key history |

```mermaid
flowchart TD
    A[GPTQModel.load: checkpoint JSON] --> B[load_model_config: dbrx_converted]
    B --> C[DbrxConvertedQModel.loader: LocalModelLoader]
    C --> D[DbrxConfig: fixed nested config classes]
    D --> E{Dimensions and MoE top-k valid?}
    E -- no --> X[ValueError before model allocation]
    E -- yes --> F[DbrxForCausalLM: package-owned classes]
    F --> G{Forward input and Cache valid?}
    G -- no --> Y[ValueError or TypeError before cache update]
    G -- yes --> H[DbrxBlock: split QKV and routed experts]
    H --> I[Caller-owned Cache and FP32 logits]
    F --> J[ModulePlan: quantizable projections and excluded norms/router]
    J --> K[AWQ: no scale groups or shared-root capture]
    K --> L[Plain quantization of declared linears]
    L --> M[Packed AWQ safetensors save and local reload]
    N[Checkpoint auto_map, trust flag, code_revision] -. cannot replace local classes .-> C
    classDef failure fill:#fee,stroke:#c00,color:#900;
    class X,Y failure;
```

AWQ is deliberately **plain quantization only**. Clipped QKV and routed expert
fan-out have not been qualified for source folding, scale search or compensation.
No source/consumer relationship is inferred from width or adjacency. The unused
`ffn` shared-root feature capture is disabled because no scaling group consumes
it; retaining it caused public quantization with `fallback=None` to fail on an
empty activation bucket. Module projections still collect their ordinary inputs.
Tests run the real AWQ consumer and public calibration/packing/save/reload, not a
mock quantizer. Router/norm weights remain unquantized. Fusion is unqualified.

## Cache failure lifecycle and padding-mask ownership

Both public `DbrxForCausalLM.forward` and decoder-core `DbrxModel.forward` guard
the caller's cache through output construction and tuple encoding. Each attempted
attention cache update advances a cache-local write version **before** the write.
Any escaping exception after a write attempt marks that cache invalid. This
includes a partial decoder update, complete decoder followed by head/loss/auxiliary
failure, output-construction failure, and a decoder forward-hook failure inside
the LM call. Validation failures before any write do not invalidate the prefix.

Invalid caches are **not rolled back or copied**. Both entry points refuse reuse
until every layer reports zero length: the caller must run `cache.reset()`, then
prefill the **entire** prompt again. Resetting just the first layer does not
qualify. This avoids copying a trained-scale KV prefix on every token. The marker
is cache-local, not stored on the model; concurrent calls require independent
caches. Failure invalidation covers these forward bodies, not arbitrary user
hooks running after the outer LM forward has returned.

`DbrxForCausalLM.create_masks_for_generate` leaves the generation padding mask
unconverted. `DbrxModel` constructs the causal mask exactly once while auxiliary
loss retains its 2D validity metadata. Transformers may omit an all-valid mask;
that remains equivalent to no padding. Arbitrary caller 4D additive masks still
work for attention without router loss, but do **not** qualify as router padding
metadata. This avoids guessing validity from an internal additive mask.

The LM snapshots the physical cache length before the decoder mutates it. For a
query of length `q`, auxiliary loss selects padding columns `start + arange(q)`.
This is neither the last `q` capacity columns nor padding-adjusted RoPE position
IDs. The scalar StaticCache length is cloned because its counter is updated
in-place by every decoder block. Full-capacity masks with future zero padding
therefore yield the same finite auxiliary/labeled loss as minimal-width masks.

Evidence below compares the original local port `b9443ebb47b258527693641468864f590ef2ce39`
to the corrected runtime; all locations are in this package's `modeling_dbrx.py`.

| Mechanism | Original local port | Corrected runtime |
| --- | --- | --- |
| Failed decoder/head/loss/encoding | `:263-298`, `:354-379`: cache remained partially or fully advanced | `:27-42`, `:283-319`, `:380-416`: nested cache guards invalidate and reject reuse until reset |
| Static generation mask | `:270-271`: valid caller 2D mask rejected after HF 4D conversion | `:368-371`: model-owned generation hook preserves padding metadata; `:293-296` builds causal attention |
| Auxiliary current tokens | `:367-369`: last query-width columns could be future padding | `:383-387`, `:400-408`: snapshot physical start and index actual current positions |

```mermaid
flowchart TD
    A["DbrxForCausalLM.forward / DbrxModel.forward"] --> B{"_cache_transaction: cache invalid?"}
    B -- yes --> C{"All cache layers reset to zero?"}
    C -- no --> X["ValueError: reset and full prefill required"]
    C -- yes --> D["Clear invalid marker; prefill full prompt"]
    B -- no --> E["Snapshot cache-local write version"]
    D --> E
    E --> F["DbrxAttention: advance version before Cache.update"]
    F --> G["Decoder, LM head, loss, auxiliary loss, output encoding"]
    G -- success --> H["Return valid caller-owned cache"]
    F -- exception --> I["Mark cache invalid; re-raise original exception"]
    G -- exception --> I
    I --> B
    classDef failure fill:#fee,stroke:#c00,color:#900;
    class X,I failure;
```

```mermaid
flowchart TD
    A["GenerationMixin: caller 2D padding mask"] --> B["create_masks_for_generate: keep original mask"]
    B --> C["DbrxModel: create_causal_mask"]
    C --> D["DbrxAttention: additive causal attention"]
    B --> E["LM: snapshot physical cache start before decoder"]
    E --> F["index_select: start plus current-token positions"]
    F --> G["load_balancing_loss_func: current-token validity"]
    H["Arbitrary caller 4D mask with router loss"] --> X["ValueError before cache mutation"]
    classDef failure fill:#fee,stroke:#c00,color:#900;
    class X failure;
```

## Qualification and limits

The two model-specific suites use real CPU models, an independent dense-expert
Torch oracle, and full-dimension meta models. They cover MQA/GQA/MHA; expert count,
top-k, normalization, uniform assignment and activation variants; left padding and
causality; FP32/FP16/BF16 forward/reload; dynamic/static continuation; greedy/beam
generation; inputs-embeds, loss and tuple outputs; failed-input/cache retry;
simultaneous CPU calls with independent caches; malicious config/model `auto_map`
across all four API-trust/security-bypass combinations; public loading; exact
checkpoint ownership; module-plan flags/MoE families; and plain AWQ packed reload.

Exact commands, run from the repository root with
`/home/ubuntu/.venvs/gptqmodel-tf5/bin/python` (Python 3.14t, Torch 2.15.0+cpu,
Transformers 5.23.0):

```sh
python -s -m pytest tests/test_dbrx_converted_local_port.py tests/module_tree/test_dbrx_converted_local_port.py tests/module_tree/test_registered_module_plans.py -q --tb=short -o log_cli=false
ruff check gptqmodel/models/definitions/dbrx_converted.py gptqmodel/models/ports/dbrx_converted tests/test_dbrx_converted_local_port.py tests/module_tree/test_dbrx_converted_local_port.py
pyright --pythonpath /home/ubuntu/.venvs/gptqmodel-tf5/bin/python gptqmodel/models/ports/dbrx_converted
git diff --check
```

Peer-review regression command:

```sh
python -s -m pytest tests/test_dbrx_converted_local_port.py -q --tb=short -o log_cli=false -k 'failed_cached_forward or static_generation_preserves or static_capacity_mask or arbitrary_4d'
```

On the original local-port runtime this gave **30 failed, 1 passed** (5.22s):
24 missing cache-reuse errors, four static generation 4D-mask rejections, and two
NaN capacity-mask auxiliary/labeled losses. After repair it gave **31 passed**
(3.92s). The combined runtime/module-plan/registry command above gave **494 passed**
(7.62s), with no skips. Tests assert failed-state refusal through **both** LM and core, unchanged
KV state on refused retries, rejection after only one layer is reset, and numerical
parity after full reset/prefill for DynamicCache and StaticCache. Failure injection
covers decoder pre-hooks, final norm, cache update, decoder output construction,
tuple encoding, decoder forward-hooks inside the LM, LM head, label loss,
auxiliary loss, and LM output construction/tuple encoding. Static generation
checks greedy/beam tokens and scores against uncached controls, finite auxiliary
loss on every step, and preserved padding semantics. Capacity-mask cases include
left padding with logical RoPE IDs different from physical cache positions.

Red evidence: the four `test_definition_owns_config_before_checkpoint_python`
cases fail on the pre-port definition (including actual checkpoint Python
execution in the trust+bypass case). The public tokenizer-discovery trust+bypass
case also exposed the separately repaired shared loader issue. The new
`test_router_aux_loss_with_padded_cache_continuation` fails with a real tensor
shape mismatch before slicing its auxiliary mask to current tokens. Public
`test_plain_awq_public_quantization_and_packed_reload` fails with missing `ffn`
activation features before the model-specific root-capture policy is applied.

Not qualified here: trained 132B checkpoint perplexity/chat quality, GPU kernels,
multi-GPU/offload behavior, GPU GIL=0 concurrency, FlashAttention/SDPA, compiled
execution, AWQ folding/search, fusion, and training/gradient checkpointing. The
ported attention path is eager. No multi-GB trained weights are needed for these
tests, and passing tiny CPU checks does not assert trained-checkpoint quality.
