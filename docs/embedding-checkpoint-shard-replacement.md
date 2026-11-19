# Embedding checkpoint shard replacement

## Public save dispatch

`GPTQModel.save()` and `GPTQModel.save_pretrained()` route on the same model
state before either writer runs (`gptqmodel/models/writer.py:2034-2056`,
`gptqmodel/models/base.py:4267-4363`). A dense wrapper delegates to the wrapped
Transformers model. A checkpoint whose only changed payload is a module
replacement uses `save_quantized_embeddings()`. An ordinary quantized
checkpoint keeps the full `save_quantized()` writer.

Before this change `save_pretrained()` always called `save_quantized()`:

```mermaid
flowchart TD
    S["save_pretrained"] --> Q["save_quantized"]
    D["dense wrapper"] --> S
    R["module-replacement checkpoint"] --> S
    Q --> E["ValueError: model is not quantized"]
    Q --> W["dense model.embed_tokens.weight; packed endpoints dropped"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class E,W failure;
```

The current flow is shared by both public names:

```mermaid
flowchart TD
    A["GPTQModel.save"] --> B{"self.quantized?"}
    P["GPTQModel.save_pretrained"] --> B
    B -- No --> C["self.model.save_pretrained"]
    B -- Yes --> D{"_model_free_weight_only_replacements_only?"}
    D -- Yes --> E["save_quantized_embeddings"]
    D -- No --> F["save_quantized"]
    C -.-> X["writer errors propagate"]
    E -.-> X
    F -.-> X
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X failure;
```

| Model state | Before `save_pretrained()` | Current contract |
| --- | --- | --- |
| Dense wrapper | `save_quantized()` raised `ValueError` (`gptqmodel/models/writer.py:2033-2042` before this change) | `self.model.save_pretrained()` owns config, tied weights and sharding (`gptqmodel/models/writer.py:2044-2045`) |
| Weight-only module replacement | `save_quantized()` rebuilt the checkpoint and wrote the dense embedding tensor, dropping the packed payload | `save_quantized_embeddings()` preserves the replaced modules (`gptqmodel/models/writer.py:2046-2052`) |
| Ordinary quantized | `save_quantized()` | unchanged (`gptqmodel/models/writer.py:2053-2054`) |

`save()` already selected the writer by state
(`gptqmodel/models/base.py:4319-4363`); the override now follows the same
branches, so the two public names cannot disagree.

Introduced by [PR #5480](https://github.com/ModelCloud/GPTQModel/pull/5480).

`save_quantized_embeddings()` copies missing source files, then rewrites only
the safetensors shards that contain replaced modules. Each rewritten shard is
read from the existing output when present, or from the source checkpoint
otherwise. The writer stages the new shard beside its destination so a failed
serialization leaves that destination shard intact. On a repeated save, indexed
prior tensor keys under each replaced module prefix, plus matching keys found
in affected shards, are removed before the current `module.state_dict()` is
written. An index, when present, is updated to match.

```mermaid
flowchart TD
    A["save_quantized_embeddings"] --> B["_copy_missing_checkpoint_files"]
    B --> C["_save_module_replacement_safetensors resolves replacement prefixes"]
    C --> D{"Matching checkpoint shard?"}
    D -- No --> X["ValueError before shard rewrite"]
    D -- Yes --> E["Collect source and indexed output keys by runtime prefix"]
    E --> F["Read existing output shard or source shard"]
    F --> G["Drop old runtime-prefix keys found in the read shard"]
    G --> H["Insert current module.state_dict replacements"]
    H --> I["_atomic_shard_path stages sibling temporary with destination mode"]
    I --> J["save_file writes temporary safetensors shard"]
    J --> K{"Write completed?"}
    K -- No --> Y["Remove temporary; current destination shard stays intact; raise"]
    K -- Yes --> L["os.replace publishes rewritten shard"]
    L --> M{"More affected shards?"}
    M -- Yes --> F
    M -- No --> N["save_quantized_embeddings removes old index keys and adds current keys"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,Y failure;
```

| Step | Prior behavior or issue | Current contract |
| --- | --- | --- |
| Rewrite one affected shard | `save_file(tensors, target_path)` opened the destination directly (`gptqmodel/models/writer.py:448` before PR #3623). An I/O failure after opening it could leave a partial shard. | `save_file` writes a sibling temporary under `_atomic_shard_path`; only a successful write reaches `os.replace` (`gptqmodel/models/writer.py:913`). |
| Existing destination mode | A direct rewrite retained the file's mode. | `_atomic_shard_path` applies the destination's mode to the temporary before replacement (`gptqmodel/utils/model.py:2790-2830`). A new shard follows the process umask. |
| Failed write | The destination could be truncated, while its index still pointed to it. | The temporary is removed and the destination shard remains readable. The exception stops the save before the index update. |
| Prior module keys on repeated save | Only source checkpoint keys and existing `weight`/`bias` keys were dropped. Old layout keys such as `proj.old_scale` stayed in the shard and index (`gptqmodel/models/writer.py:396-405` before this change). | `_save_module_replacement_safetensors` collects every indexed key under each runtime prefix and also filters such keys while reading a shard (`gptqmodel/models/writer.py:398-405, 442-449`). `save_quantized_embeddings` removes those names from the index before adding current keys (`gptqmodel/models/writer.py:942-945`). |

The guarantee is per shard. If a later shard fails, earlier successfully
rewritten shards stay published; the export as a whole is not transactional.
The normal streaming writer has used atomic shard replacement since
[PR #3584](https://github.com/ModelCloud/GPTQModel/pull/3584).
The module-replacement writer gained the same per-shard protection in
[PR #3623](https://github.com/ModelCloud/GPTQModel/pull/3623).
Repeated saves also remove keys from prior replacement layouts as described in
[PR #3624](https://github.com/ModelCloud/GPTQModel/pull/3624).
