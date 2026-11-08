# Reshard source selection

`reshard(from_model, ...)` accepts a checkpoint directory, a specific
`.safetensors` file, a `.safetensors.index.json` file, or a loaded GPTQModel
whose `model_local_path` is a directory. An explicit index selects its shards.
A standalone file selects only itself; a file listed by a standard index selects
the complete indexed checkpoint. If two standard indexes list that file, their
resolved tensor-to-shard maps must agree; otherwise name the intended index
explicitly. A directory retains standard index-first
discovery. Auxiliary config and tokenizer files still come from the source
directory.

```mermaid
flowchart TD
    A["reshard from_model"] --> B["_resolve_source_input"]
    B --> C{"Source kind?"}
    C -- Invalid --> X["FileNotFoundError or TypeError"]
    C -- Directory or loaded model --> D["_scan_checkpoint without selected_file"]
    C -- Safetensors file --> E["_index_containing_file"]
    C -- Index file --> F["_scan_indexed_checkpoint with named index"]
    E --> P{"Listed in a standard index?"}
    P -- Yes --> R{"Matching index maps agree?"}
    R -- Yes --> H
    R -- No --> Z["ValueError; select an index explicitly"]
    P -- No --> Q["_scan_unindexed_checkpoint with one file"]
    D --> G{"Standard index present?"}
    G -- Yes --> H["_scan_indexed_checkpoint with standard index"]
    G -- No --> I["_scan_unindexed_checkpoint with discovered files"]
    Q --> J{"Valid header and tensor references?"}
    F --> J
    H --> J
    I --> J
    J -- No --> Y["ValueError or FileNotFoundError"]
    J -- Yes --> K["_group_tensors and write staging"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,Y,Z failure;
```

Previously, `_resolve_source_dir` discarded an explicit filename at
`gptqmodel/reshard.py:279-285`, and `_scan_checkpoint` then preferred a
standard index or discovered every safetensors file in that directory at
`gptqmodel/reshard.py:436-443`. The current `_resolve_source_input` keeps the
selected path at `gptqmodel/reshard.py:279-293`. `_index_containing_file` keeps
a selected shard attached to its full checkpoint at
`gptqmodel/reshard.py:439-468`; `_scan_checkpoint` branches at
`gptqmodel/reshard.py:471-492`, called from `gptqmodel/reshard.py:827`.

| `from_model` | Previous selection | Current selection |
| --- | --- | --- |
| Directory | Standard index if present, otherwise all discovered safetensors files | Same |
| Standalone `.safetensors` file | Standard index or all files in its directory | Only that file |
| Shard listed in a standard index | Complete indexed checkpoint | Complete indexed checkpoint |
| Shard listed in two conflicting standard indexes | First standard index wins, even if the second names a different complete checkpoint | `ValueError` before writing; pass the intended `.safetensors.index.json` path |
| Shard listed in equivalent standard indexes | First standard index wins | Complete indexed checkpoint; equivalent normalized paths are accepted |
| Specific `.safetensors.index.json` file | Standard index or all files in its directory | Only shards named in that index |

An explicitly selected index must pass the same shard and tensor-name checks
as a standard index. If it is invalid, resharding fails before publishing the
destination. An unrelated invalid standard index does not override a named
standalone file.
`_index_containing_file()` now compares the complete resolved weight maps for
every standard index that lists the selected shard. It raises on conflicting
maps and accepts equivalent paths such as `shard.safetensors` and
`./shard.safetensors`. An explicit index path bypasses this choice.
See [PR #3589](https://github.com/ModelCloud/GPTQModel/pull/3589) for the
reproducer and rationale.
The indexed-shard compatibility rule is covered by
[PR #3590](https://github.com/ModelCloud/GPTQModel/pull/3590).
The conflicting-index guard was added in
[PR #3622](https://github.com/ModelCloud/GPTQModel/pull/3622).
