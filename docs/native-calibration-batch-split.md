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
    D --> F["NativeProcessor._split_batch_samples(inp, keep_mask)"]
    F --> G{"keep_mask shape matches [B, S]"}
    G -- "yes" --> H["one entry per sample: inp[b][mask[b]], empty samples skipped"]
    G -- "no" --> I["one entry per batch: inp.to(target_device)"]
    H --> J["native_inp_caches[name]"]
    I --> J
    J --> K["NativeProcessor.process -> module.state[NATIVE_INPUTS_STATE_KEY]"]
    K --> L["GPTAQ.__init__ / FOEM.__init__ pops the list"]
    L --> M["GPTQProcessor hook calls add_batch once per kept sample"]
    M --> N["GPTAQ.process_batch / FOEM.process_batch pop one entry per add_batch"]
    E --> O["single [N_kept, H] entry per batch (unchanged for other quantizers)"]
```

The split mirrors `GPTQProcessor`'s own per-sample contract: samples whose keep
mask is empty are skipped in both places, so the entry order stays aligned even
with padded calibration rows.

## Before / after

| Mechanism | Before | After |
| --- | --- | --- |
| Capture unit | One entry per forward batch, `native_processor.py:101` (main) | One entry per kept sample when GPTAQ/FOEM is configured, `native_processor.py` `_split_batch_samples` |
| Mask ownership | The hook wrapper masked the batch to `[N_kept, H]`; the native entry had no per-sample boundary | `NativeProcessor.preserve_batch_keep_mask` is enabled for GPTAQ/FOEM, so the hook sees `[B, S, H]` and applies the same `[B, S]` mask per sample |
| Consumer | `gptaq.py:79` / `foem.py:160` did `native_inps.pop(0)` per `add_batch`, but the list held one batch entry | One pop per `add_batch` matches one captured sample |
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
