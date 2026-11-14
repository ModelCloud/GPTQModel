<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Quantized checkpoint subfolder resolution

A Hub repository or a local directory can keep the quantized checkpoint in a
subdirectory. `from_quantized(..., subfolder="nested")` therefore has to read
the architecture config, the quantize config, the weight shards, and any
relative adapter manifest from that one directory. Reading some of them from the
repository root mixes two checkpoints: the loader can pair a config from
`nested/` with weights that happen to sit at the root, or fail outright when the
root holds nothing.

## New flow

```mermaid
flowchart TD
    A["from_quantized(model, subfolder)"] --> B["resolve_checkpoint_subfolder(model_local_path, subfolder)"]
    B --> C{"absolute, drive letter, backslash, or dot segment?"}
    C -- Yes --> R1["ValueError"]
    C -- No --> D{"resolves outside the checkpoint?"}
    D -- Yes --> R1
    D -- No --> E["checkpoint_dir"]
    E --> F["AutoConfig.from_pretrained(root, subfolder)"]
    E --> G["get_checkpoints(checkpoint_dir)"]
    E --> H["QuantizeConfig.from_pretrained(root, subfolder)"]
    E --> I["resolve_checkpoint_adapter_path(manifest, checkpoint_dir)"]
    H --> J["adapter provenance = (abspath(checkpoint_dir), manifest)"]
    J --> K["Lora.post_init resolves the manifest inside checkpoint_dir"]
    G --> L["weight shards from checkpoint_dir"]
    style R1 fill:#fee,stroke:#c00
```

`resolve_checkpoint_subfolder` (`gptqmodel/utils/hf.py`) only narrows the base
path. It rejects absolute paths, Windows drive letters, backslashes, and `.`/`..`
segments instead of normalizing them away, and it rejects a symlinked subfolder
that resolves outside the checkpoint. `""`, `None`, and `"."` keep the previous
root behavior, so callers that never pass a subfolder are unaffected.

The same helper is used by `get_checkpoints` (`gptqmodel/utils/model.py`): a
downloaded snapshot is a local directory, so the Hub branch that forwards
`subfolder` to `cached_file` is never reached for it. Without the join, a nested
checkpoint was searched at the repository root.

## Previous flow

```mermaid
flowchart TD
    A["from_quantized(model, subfolder)"] --> B["AutoConfig.from_pretrained(root, subfolder)"]
    A --> C["get_checkpoints(root)"]
    A --> D["QuantizeConfig.from_pretrained(root)"]
    A --> E["resolve_checkpoint_adapter_path(manifest, root)"]
    C --> F["FileNotFoundError: no shards at the root"]
    D --> G["FileNotFoundError: no config at the root"]
    E --> H["ValueError: adapter is not inside the checkpoint"]
    style F fill:#fee,stroke:#c00
    style G fill:#fee,stroke:#c00
    style H fill:#fee,stroke:#c00
```

| Mechanism (at the pre-change head) | Previous behavior | New behavior |
| --- | --- | --- |
| `models/loader.py:1756` pops `subfolder` and forwards it only through `cached_file_kwargs` to `AutoConfig` | Architecture config honored the subfolder; every other artifact used the root | One `checkpoint_dir` is resolved and reused |
| `utils/model.py:1714` local branch of `get_checkpoints` | Searched `model_id_or_path` directly; `subfolder` was ignored | Joins the validated subfolder before searching |
| `quantization/config.py:4753` `QuantizeConfig.from_pretrained` | Popped only `format`; joined `QUANT_CONFIG_FILENAME_COMPAT` under `save_dir` (`:4759`) | Honors `subfolder` and records the resolved directory as the adapter provenance |
| `models/loader.py:1828` adapter manifest resolution | Used `model_local_path`, so a manifest under `nested/` was rejected as outside the checkpoint | Uses `checkpoint_dir` |
| `quantization/config.py:4776` adapter provenance | `(abspath(save_dir), manifest)` | `(abspath(checkpoint_dir), manifest)` |

## Failure and ambiguity contract

| Situation | Result |
| --- | --- |
| `subfolder=""`, `None`, or `"."` | Repository root, exactly as before |
| `subfolder="nested"` with the checkpoint under `nested/` | Every artifact resolves under `nested/` |
| `subfolder="../outside"`, `/abs`, `C:\repo`, or `a\b` | `ValueError` naming `subfolder` |
| Symlinked `nested` that resolves outside the checkpoint | `ValueError` naming the checkpoint boundary |
| Subfolder exists but holds no quantize config | `FileNotFoundError` from the config read (unchanged error class) |
| Hub repository id with `subfolder` | `cached_file` receives `subfolder`, unchanged |

Tokenizer resolution is deliberately out of scope: `load_hf_tokenizer`
delegates to Tokenicer, which owns tokenizer and chat-template normalization, so
threading `subfolder` through it is a Tokenicer change rather than a loader one.

## Validation

| Command | Result |
| --- | --- |
| `pytest tests/test_checkpoint_subfolder_resolution.py -q` | 11 passed |
| same file with the consumer changes reverted | 7 failed, 3 passed |
| `pytest tests/test_adapter_remote_security.py -q` | 51 passed |
| `pytest tests/test_local_model_paths.py tests/qcfg -q` | 111 passed |
| issue repro `QuantizeConfig.from_pretrained(root, subfolder="nested")` | returned `bits=4`; `FileNotFoundError` before the fix |

The loader-level test (`test_loader_resolves_adapter_manifest_under_subfolder`)
drives `from_quantized` far enough to reach the manifest resolver and asserts it
receives `nested/`; it fails when the loader passes the repository root again.
