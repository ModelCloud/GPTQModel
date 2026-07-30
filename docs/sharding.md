# Per-Layer Checkpoint Sharding

GPT-QModel can write quantized checkpoints as one `safetensors` file per
transformer layer, with a separate final shard for non-layer tensors
(`embed_tokens`, `norm`, `lm_head`, etc.). This is exposed both as a
post-processing `reshard()` API and as a save-time option on
`QuantizeConfig`/`model.save()`.

## Why per-layer sharding matters

Large dense checkpoints are often released as multi-GB `safetensors` shards
(~4 GB each). During layer-by-layer quantization and `LazyTurtle` meta-device
loading, every layer re-reads its tensors from those huge shards. On systems
with finite OS page cache / ZFS ARC, the later layers repeatedly fault the
same large files back into memory. The result is a non-linear slowdown: loading
the first layers is fast, but later layers take progressively longer.

For example, with `Laguna-S-2.1` the original dense checkpoint is split into
46 shards (~235 GB). Emulating the `LazyTurtle` lifecycle showed per-layer
read time growing by roughly `1.9x` from the first quarter of layers to the
last quarter. After resharding into per-layer files, the growth disappeared
(`~0.9x`) because each small layer shard stays cacheable and is read
sequentially.

Per-layer sharding does not reduce total bytes read, but it makes each read
`O(layer size)` and cache-friendly instead of `O(checkpoint shard size)` with
random access into a multi-GB file.

## Resharding an existing dense checkpoint

The standalone `reshard()` function streams the source checkpoint one shard at
a time, so it does not need to load the whole model into memory.

```python
from gptqmodel import reshard, ShardStrategy

reshard(
    "/monster/data/model/Laguna-S-2.1",
    "/monster/data/model/Laguna-S-2.1-PER-LAYER",
    strategy=ShardStrategy.PER_LAYER,
)
```

### Layer-prefix detection

`reshard()` no longer hardcodes `model.layers`. It determines the layer-node
prefixes automatically:

1. Load `config.json` and call the architecture-specific
   `extract_layers_node()` from the matching `GPTQModel` definition. This
   handles `model.layers`, `language_model.model.layers`, `transformer.h`,
   `model.L_module.layers` / `H_module.layers`, and other model-family
   variations.
2. If the model type is unsupported or `config.json` is missing, fall back to
   scanning tensor names. The first numeric segment in each tensor path is
   treated as the layer index, and prefixes ending in common layer containers
   (`layers`, `h`, `blocks`, `block`, `encoder`, `layer`) are selected.
3. If automatic detection fails, you can pass explicit prefixes:

```python
reshard(
    "/path/to/checkpoint",
    "/path/to/output",
    strategy=ShardStrategy.PER_LAYER,
    layer_prefixes=["language_model.model.layers"],
)
```

Memory stays bounded by a single source shard plus the largest output staging
buffer, making the tool usable on hosts with ~64 GB CPU RAM even for
>200 GB checkpoints. Progress and disk telemetry are logged through the
standard `LogBar` pipeline.

The output directory uses flat `model-XXXXX-of-YYYYY.safetensors` names and
includes a `model.safetensors.index.json` weight map:

```text
Laguna-S-2.1-PER-LAYER/
  config.json
  model-00001-of-00049.safetensors   # layer 0
  model-00002-of-00049.safetensors   # layer 1
  ...
  model-00048-of-00049.safetensors   # layer 47
  model-00049-of-00049.safetensors   # embed / norm / lm_head
  model.safetensors.index.json
```

## Saving a quantized model directly into per-layer shards

You can request per-layer sharding at save time through `QuantizeConfig` or as
a `save()` argument. This avoids a separate post-quantization `reshard()` pass
and is useful when the quantized model is being written straight from the
quantization pipeline.

```python
from gptqmodel import GPTQModel, QuantizeConfig, ShardStrategy

model = GPTQModel.load("modelcloud/Laguna-S-2.1-bf16")
model.quantize(
    examples,
    batch_size=1,
    quantize_config=QuantizeConfig(
        bits=4,
        group_size=128,
        shard_strategy=ShardStrategy.PER_LAYER,
    ),
)
model.save("/path/to/Laguna-S-2.1-GPTQ-PER-LAYER")
```

You can also pass it directly to `save()`, `save_quantized()`, or
`save_pretrained()`:

```python
model.save(
    "/path/to/output",
    shard_strategy=ShardStrategy.PER_LAYER,
)

# or a string for convenience
model.save("/path/to/output", shard_strategy="per_layer")
```

The same flat `model-XXXXX-of-YYYYY.safetensors` layout is produced.

## Output layout and index

* One safetensors file per layer (`model.layers.{idx}`).
* One final safetensors file for all non-layer tensors
  (`embed_tokens`, `norm`, `lm_head`, and any other top-level weights).
* `model.safetensors.index.json` maps every tensor name to its shard file.

This layout is compatible with Hugging Face `transformers` and with the
`LazyTurtle` meta-device path because a layer can be loaded by opening a
single small shard.

## Memory and performance notes

* `reshard()` streams; it does **not** load all source tensors into memory.
* Save-time sharding groups tensors in memory using the same streaming
  `TensorSource` path used for normal checkpoint saves, then writes each group
  to a staging directory and renames the final files sequentially. Peak memory
  is still bounded by the active layer/group.
* For very large layers, `max_shard_size` is still respected and may split a
  layer across multiple sequential files. If you need strictly one file per
  layer, ensure `max_shard_size` is larger than the largest layer.

## When to use it

* Quantizing or loading >100 GB dense models with `LazyTurtle` or
  `offload_to_disk`.
* You observe per-layer materialize/load times growing as quantization
  progresses.
* You want checkpoint files that map 1:1 to model layers for easier
  distribution or partial loading.
