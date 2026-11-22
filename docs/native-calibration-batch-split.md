# Native calibration batch split

GPTAQ and FOEM (with `alpha > 0`) compare each calibration sample's native
activation against the GPTQ-rescaled activation. They therefore consume the
captured native inputs one entry per **sample**, while the capture stage used to
store one entry per **forward batch**. The two lists were only aligned at
`batch_size=1`, so a larger calibration batch made sample 0 consume the whole
batch tensor and the remaining samples pop from an empty list.

This page describes the split that keeps the capture and consumer aligned. It
was introduced with PR #3995 (issue #3978).

## New flow

```mermaid
flowchart TD
    A["ModuleLooper._set_processor_mask(processor, keep_mask)"] --> B["_masked_hook_wrapper(processor, inner_hook)"]
    B --> C{"processor.preserve_batch_keep_mask"}
    C -- "True for GPTAQ/FOEM" --> D["inner_hook receives raw [B, S, H]"]
    C -- "False" --> E["apply_keep_mask_bt flattens to [N_kept, H]"]
    D --> F["NativeProcessor._split_batch_samples(inp, keep_mask, batch_index)"]
    F --> G{"input layout vs keep_mask [B, S]"}
    G -- "[B, S, H], shape[:2] == mask.shape" --> S{"same batch, source storage and mask already cached?"}
    G -- "[B*S, H], shape[0] == mask.numel()" --> S
    G -- "neither" --> I["one entry per batch: inp.to(target_device)"]
    S -- "no" --> H["one masked copy per sample: inp[b][mask[b]] or inp[b*S:(b+1)*S][mask[b]], empty samples skipped"]
    S -- "yes" --> R["reuse the cached per-sample entry"]
    H --> J["native_inp_caches[name] += (batch_index, entry)"]
    R --> J
    I --> J
    J --> K["NativeProcessor.process -> module.state[NATIVE_INPUTS_STATE_KEY]"]
    K --> L["GPTAQ.__init__ / FOEM.__init__ pops the list"]
    L --> M["GPTQProcessor hook calls add_batch once per kept sample with batch_index"]
    M --> N["GPTAQ._take_native_input / FOEM._take_native_input scans for the matching batch_index"]
    E --> O["single [N_kept, H] entry per batch (unchanged for other quantizers)"]
```

The split mirrors `GPTQProcessor`'s own per-sample contract: samples whose keep
mask is empty are skipped in both places, so the entry order stays aligned even
with padded calibration rows. Both layouts the quantized-pass hook splits per
sample are split here too (`native_processor.py` `_split_batch_samples`, the
`flattened` branch): the regular `[B, S, H]` input and the flattened `[B*S, H]`
rows that HF `qwen2_moe` feeds its `shared_expert` (`hidden_states.view(-1, H)`),
which `GPTQProcessor` handles in its `inp_tensor.shape[0] == keep_mask.numel()`
branch (`gptq_processor.py:635`). Each entry is stored as `(batch_index, tensor)`
because the data-parallel forward does not replay batches in capture order; see
[Multi-GPU quantization](multi_gpu_quantization.md#gptaqfoem-per-device-hessian-reduction).

## Sibling input sharing

The sibling projections fed by one source activation — `q_proj`/`k_proj`/`v_proj`
and `gate_proj`/`up_proj` — are called with the same `[B, S, H]` tensor in the
same forward pass. Their masked slices are therefore bitwise-identical, but the
split used to build a fresh copy for every module and kept all copies alive
until the module was consumed. `NativeProcessor._split_batch_samples` now keeps
one entry per `(batch_index, source storage, mask)` in a thread-local cache and
returns the same tensor object to every sibling.

The key is `(untyped_storage().data_ptr(), storage_offset, shape, stride, dtype,
device)`; the cache token carries the calibration batch index and the keep-mask
identity. Capture hooks run on data-parallel workers, so every capture thread
owns one slot in `NativeProcessor._native_share_caches`, keyed by thread id and
guarded by `_native_share_lock`. A thread's slot is replaced when its batch
index or mask changes, and `_release_native_share_cache()` drops every slot at
once from `process()` (the layer's forward pass is complete) and from
`finalize()`. The cache therefore never pins a finished layer's activations or
survives `quantize()`, and a recycled source address in a later layer cannot hit
a stale entry. Entry tagging and the
consume-by-`batch_index` contract are unchanged: each module still stores its
own `(batch_index, tensor)` list and `_take_native_input` still pops one entry
per `add_batch`.

Two different views of one storage are treated as the same source, which is
safe because their values are identical. An in-place write to the source
between sibling hooks is not detected; normal decoder forwards do not mutate
the hidden state before its sibling projections, and the `batch_size=1` alias
path already makes that same assumption.

Consumers treat entries as read-only. `GPTAQ.process_batch` and
`FOEM.process_batch` only call `.to`, `.reshape`, `.unsqueeze` and `.t` on the
native tensor and build the delta with out-of-place arithmetic, so sharing the
storage does not change the Hessian or `dXXT`. The source activation is likewise
assumed to be unchanged between sibling hooks; that is the same assumption the
`batch_size=1` alias path already makes (there, `.to` returns the source tensor
itself).

| Mechanism | Before | After |
| --- | --- | --- |
| Sibling masked copy | every hook call built `inp[b][mask[b]].to(target_device)` | the first call builds it, `q`/`k`/`v` and `gate`/`up` reuse the same storage |
| Entry layout | `(batch_index, tensor)` per module | unchanged |
| `add_batch` calls | one per kept sample per module | unchanged |
| Cache lifetime | none | one slot per capture thread, replaced on batch-index or keep-mask change, all slots dropped by `process()` and `finalize()` |

Measured on cached `HuggingFaceTB/SmolLM2-135M` with GPTAQ at `batch_size=4`
(8 calibration rows): the same 1680 native entries total 44,029,440 bytes in
both runs, but they occupy 1680 distinct storages (44,029,440 bytes) before the
change and 960 distinct storages (28,788,480 bytes) after. The 15,240,960 bytes
(34.6%) of duplicated sibling copies disappear, and the saved checkpoint
(`model.safetensors`) is byte-identical between the two runs.

## Cache release (#4463)

The first version kept the cache in a `threading.local()`. `process()` and
`finalize()` run on a different thread than the capture workers, so they could
only bump an epoch (`native_processor.py:256` on the parent commit); each
worker's `tls.entries` (`:109`), holding `(source, mask, entries)`, stayed alive
until that worker's next capture or until the processor was destroyed. The last
batch of every worker was therefore pinned across the rest of the layer and
after `quantize()` returned.

```mermaid
flowchart LR
    subgraph Before
      A1["capture worker: tls.entries = {key: (inp, mask, entries)}"] --> B1["process(): _native_share_epoch += 1"]
      B1 --> C1["tls.entries still referenced by the idle worker"]
      C1 --> D1["source activation pinned until the worker's next capture"]
    end
    subgraph After
      A2["capture worker: _native_share_caches[thread_id] = (token, {key: ...})"] --> B2["process() / finalize(): _release_native_share_cache()"]
      B2 --> C2["_native_share_caches.clear() under _native_share_lock"]
      C2 --> D2["only module-owned per-sample copies remain"]
    end
    style D1 fill:#f8d7da,stroke:#b02a37,color:#842029
    style D2 fill:#d1e7dd,stroke:#0f5132,color:#0f5132
```

Measured on Qwen2.5-0.5B-Instruct, GPTAQ, `batch_size=4`, 32 variable-length
rows on an RTX 5090 (`torch.cuda.memory_allocated()` after `quantize()`, minus
the value after `load()`): 11.37 MiB on the parent commit, 5.85 MiB after this
change, and 5.91 MiB with sibling sharing disabled entirely; peak allocation
670.2 → 664.8 MiB (664.8 MiB without sharing). Llama-3.2-1B-Instruct, GPTAQ,
`batch_size=4`, 16 rows: 16.90 → 6.38 MiB after `quantize()`, peak 1901.0 →
1890.4 MiB. The saved checkpoints are bitwise identical to the parent commit.

## Flattened `[B*S, H]` inputs (#4457)

`_split_batch_samples` used to return one entry for every input with
`inp.dim() < 3` (`native_processor.py:138` on the parent commit), while the
quantized-pass hook split flattened `[B*S, H]` rows per sample. On
`qwen2_moe`, whose `shared_expert` receives `hidden_states.view(-1, H)`, GPTAQ
and FOEM then paired one `B*S`-row native entry with an `S_kept`-row sample:

```mermaid
flowchart TD
    A["qwen2_moe shared_expert called on [B*S, H]"] --> B["NativeProcessor: inp.dim() < 3"]
    B --> C["one native entry with B*S rows, pad rows included"]
    A --> D["GPTQProcessor 2-D branch: add_batch per sample, pad rows dropped"]
    C --> E["GPTAQ._accumulate_device_batch / FOEM: native [B*S] vs quantized [S_kept]"]
    D --> E
    E --> F["RuntimeError: size of tensor a (B*S) must match tensor b (S_kept)"]
    style F fill:#f8d7da,stroke:#b02a37,color:#842029
```

Now the same keep mask slices `inp[b*S:(b+1)*S][keep[b]]`, so each native entry
has exactly the rows of the matching `add_batch` call. A 2-layer
Qwen1.5-MoE-A2.7B truncation quantizes with GPTAQ and FOEM (`alpha=0.25`) at
`batch_size` 1, 2 and 4 with variable-length rows (`tests/test_moe_gptaq_foem.py`
covers the tiny-model equivalent at `batch_size=4`). OPT's decoder flattens the
hidden state before `fc1`/`fc2` too, so OPT-125m GPTAQ at `batch_size=4` hit the
same error and now completes (wikitext-2 PPL 48.34 vs 48.03 for GPTQ and 42.91
for fp16). Models that call every captured module on `[B, S, H]` take the
unchanged 3-D branch: Llama-3.2-1B GPTAQ/FOEM and Qwen2.5-0.5B GPTAQ checkpoints
are bitwise identical to the parent commit, and so is the `batch_size=1`
Qwen1.5-MoE GPTAQ checkpoint.

## Before / after

| Mechanism | Before | After |
| --- | --- | --- |
| Capture unit | One entry per forward batch, `native_processor.py:101` (main) | One entry per kept sample when GPTAQ/FOEM is configured, `native_processor.py` `_split_batch_samples` |
| Mask ownership | The hook wrapper masked the batch to `[N_kept, H]`; the native entry had no per-sample boundary | `NativeProcessor.preserve_batch_keep_mask` is enabled for GPTAQ/FOEM, so the hook sees `[B, S, H]` and applies the same `[B, S]` mask per sample |
| Consumer | `gptaq.py:79` / `foem.py:160` did `native_inps.pop(0)` per `add_batch`, but the list held one batch entry | One keyed lookup per `add_batch` matches one captured sample; the entry also carries the calibration batch index used for multi-GPU pairing |
| Other quantizers | One entry per batch | Unchanged: `_native_per_sample` is false, so the single-entry layout is kept |

## Failure contract

```mermaid
flowchart TD
    A["batch_size = 2, GPTAQ/FOEM capture"] --> B["one native entry holds [2*S_kept, H]"]
    B --> C["sample 0 add_batch"]
    C --> D["native_inp - inp shape mismatch"]
    D --> E["RuntimeError: size of tensor a (6) must match tensor b (3)"]
    E --> F["sample 1 add_batch"]
    F --> G["IndexError: pop from empty list"]
    G --> H["FAIL-CLOSED before this change: quantization aborts"]
```

After the split the same input produces two entries of `[S_kept, H]`, the
Hessian and `dXXT` are finite, and the batched result matches a `batch_size=1`
replay of the same samples (see `tests/test_gptaq_foem_batch.py`).
