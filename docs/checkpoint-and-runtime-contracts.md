# Checkpoint metadata and runtime decision contracts

This page records the contracts that decide whether a checkpoint is portable,
whether a tied `lm_head` may be quantized, when a failed `torch.compile` call
may fall back to eager, how a quantizer's scale dtype survives loading, and
which stream owns a cached quantization scratch tensor.

These are the hardening items from the 7.11 review. The CPU int4pack dispatch
contracts are documented separately in
[CPU and offload kernel dispatch](kernel-dispatch-cpu.md).

## 1. Checkpoint metadata

A saved checkpoint describes tensors, not the machine that produced them.

| Mechanism | Before | After |
| --- | --- | --- |
| `offload_to_disk_path` in `meta` | published the quantizer's local `/tmp/gptqmodel_*` directory (`config.py:5396` before the change) | omitted; a loader chooses its own scratch directory |
| index `metadata.total_size` | summed on-disk safetensors file sizes, including headers and padding | sums `TensorSource.num_bytes` (`utils/model.py:3567`) |
| `max_shard_size` units | every `KB`/`MB`/`GB` suffix meant 1024^n | `KB`/`MB`/`GB` are decimal, `KiB`/`MiB`/`GiB` are binary, bare `K`/`M`/`G` keep the historical binary meaning (`models/writer.py:142`) |

```mermaid
flowchart TD
    A["save_quantized(state_dict)"] --> B["streaming_state_dict_to_shards"]
    B --> C["_plan_shards uses TensorSource.num_bytes"]
    B --> D["write or reuse each shard"]
    D --> E["index total_size = sum of entry.num_bytes"]
    A --> F["QuantizeConfig.to_dict"]
    F --> G["meta without offload_to_disk_path"]
    G --> H["portable quantize_config.json"]
    style E fill:#d1e7dd,stroke:#146c43,stroke-width:2px
    style H fill:#d1e7dd,stroke:#146c43,stroke-width:2px
```

`max_shard_size="4GB"` therefore means `4_000_000_000` bytes, matching
Transformers, while `"4GiB"` means `4 * 1024^3`. A legacy checkpoint that
still contains `meta.offload_to_disk_path` keeps loading; the field is only
dropped on the next save.

## 2. Tied `lm_head`

A checkpoint may redundantly store `lm_head.weight` even though the model ties
the head to the input embedding (Qwen3 does this). The stored tensor is not
evidence of an independent head: the loader ties the parameters after
materialization.

```mermaid
flowchart TD
    A["QuantizeConfig.lm_head = True"] --> B["find_tied_parameters(model)"]
    B -- "shared parameter" --> R["NotImplementedError<br/>tied weights"]
    B -- "no shared parameter" --> C{"config or text_config<br/>tie_word_embeddings?"}
    C -- "no" --> Q["quantize the head"]
    C -- "yes" --> D{"_tied_weights_keys declares<br/>lm_head.weight?"}
    D -- "yes" --> R
    D -- "no" --> E{"lm_head weight is meta?"}
    E -- "no" --> Q
    E -- "yes" --> F{"checkpoint has<br/>lm_head.weight?"}
    F -- "no" --> R
    F -- "yes" --> Q
    style R fill:#f8d7da,stroke:#b02a37,stroke-width:2px
    style Q fill:#d1e7dd,stroke:#146c43,stroke-width:2px
```

The config check walks `text_config` / `get_text_config()` so a composite model
whose root config leaves `tie_word_embeddings` unset is still detected. The
checkpoint-tensor fallback remains for configs that do not declare tying.

## 3. `torch.compile` eager fallback

A deferred compiler failure may retry eagerly, but only when the failure is
demonstrably compiler-owned. A compiler-package frame is not enough: user code
can raise from a module whose file lives under `torch/_dynamo`.

```mermaid
flowchart TD
    A["compiled call raises exc"] --> B{"KeyboardInterrupt<br/>or SystemExit?"}
    B -- "yes" --> P["propagate"]
    B -- "no" --> C{"BackendCompilerFailed<br/>or TorchRuntimeError?"}
    C -- "yes" --> F["eager fallback"]
    C -- "no" --> D{"fullgraph?"}
    D -- "yes" --> P
    D -- "no" --> E{"TorchDynamoException<br/>in the cause chain?"}
    E -- "yes" --> F
    E -- "no" --> G{"innermost frame under<br/>_dynamo/_inductor/_functorch?"}
    G -- "no" --> P
    G -- "yes" --> H{"RuntimeError, or dynamo's<br/>Guard failed message?"}
    H -- "no" --> P
    H -- "yes" --> F
    style P fill:#f8d7da,stroke:#b02a37,stroke-width:2px
    style F fill:#d1e7dd,stroke:#146c43,stroke-width:2px
```

This keeps the #4094 guard-assertion fallback and inductor's closed-pool
`RuntimeError` while letting a `ValueError` or an unrelated `AssertionError`
raised from a compiler frame reach the caller.

## 4. Checkpoint scale dtype

Accelerate casts floating buffers to the requested model dtype. That is wrong
for `scales`: narrowing an FP16 scale to BF16 changes the dequantized weight
and a later `save_pretrained()` republishes the narrowed value.

```mermaid
flowchart TD
    A["accelerate.load_checkpoint_in_model"] --> B{"BaseQuantLinear<br/>has a scales buffer?"}
    B -- "yes" --> C["restore scales from checkpoint dtype"]
    B -- "no" --> D["leave the buffer"]
    C --> E["post_init/transform may still cast<br/>runtime operands"]
    style C fill:#d1e7dd,stroke:#146c43,stroke-width:2px
```

Torch AWQ additionally restores `bias` because its forward makes temporary
activation-dtype copies. Other kernels keep the checkpoint scale dtype and
cast only their per-forward operands.

## 5. Stream-scoped quantization scratch

`_uniform_column_group` caches a zero group index. `torch.zeros` queues its
fill on the creating stream, so a tensor created by one worker must not be
reused by a worker on another stream.

```mermaid
flowchart TD
    A["_uniform_column_group(count, device)"] --> B{"device.type is cuda?"}
    B -- "no" --> C["cache key stream = None"]
    B -- "yes" --> D["torch.cuda.current_stream(device)"]
    D --> E["cache key stream object"]
    C --> F["torch.zeros(count, int32)"]
    E --> F
    style F fill:#d1e7dd,stroke:#146c43,stroke-width:2px
```

The stream object is part of the cache key, which also keeps it alive while an
entry exists. Steady-state callers on one stream still hit the cache.

## 6. VPTQ per-channel normalization

With `enable_norm=True, norm_dim=0`, VPTQ divides each input channel by its own
scale: `W' = (W - b) / s`. The reconstruction error in the original space is
`e = D e'` with `D = diag(s)`, so error feedback must minimize
`e'^T (D H D) e'`.

```mermaid
flowchart LR
    A["W, H"] --> B["normalize per column<br/>W' = (W - b) / s"]
    B --> C["error feedback with<br/>D H D, D = diag(s)"]
    C --> D["de-normalize<br/>W_hat = W_hat' * s + b"]
    D --> E["report loss with the<br/>original H"]
    style C fill:#d1e7dd,stroke:#146c43,stroke-width:2px
```

Measured on a 16x32 Gaussian weight and a well-conditioned Hessian (seed 0,
`vector_lens=[-1, 4]`, 16 centroids, block size 8):

| Error-feedback Hessian | True Hessian-weighted loss |
| --- | --- |
| raw `H` (before) | 0.2602 |
| `D^-1 H D^-1` (wrong direction) | 0.3535 |
| `D H D` (this change) | 0.2200 |

The independent oracle in `tests/test_vptq.py` performs its own arithmetic and
matches the quantizer's codes exactly for the `norm` and `all` variants.

## Not in scope

Auto-detected module trees still require calibration data that exercises every
discovered branch. A text-only calibration of a multimodal model must pass an
explicit text-only `module_tree`; silently leaving a vision tower unquantized
would publish a partially quantized checkpoint.
