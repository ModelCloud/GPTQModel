# Offload run directories

`QuantizeConfig.offload_to_disk_path` (or the automatic `/tmp/gptqmodel_*`
directory) is a **root**, not a bundle directory. Every quantization run creates
one private child, `run-<uuid16>`, and writes all of that run's module bundles
beneath it. The isolation is required: bundle directory names are derived from
the module path alone, so two models sharing one config would otherwise
overwrite each other's finalized bundles and the second `save()` would silently
write the first model's weights ([#4202], [PR #4297]).

The run directory is scratch space owned by the model that created it. It is
removed when that model is collected, when the quantization run fails, and
before the same model binds another run. A successful `save()` does **not**
remove it, because the live model still reads those bundles. The automatic
temporary root keeps its existing interpreter-exit cleanup; a run directory
under an explicit root is removed by the same ownership rule.

## Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Bound: bind_private_offload_root creates run-uuid
    Bound --> InUse: looper writes module bundles
    InUse --> Saved: save() streams bundles, model stays meta-backed
    InUse --> Failed: loop raises
    Saved --> InUse: in-session forward reads bundles
    Saved --> Saved: a later save() reads the same bundles
    InUse --> Released: model collected or rebind
    Saved --> Released: model collected
    Failed --> Released: release_private_offload_root in loop except
    Released --> [*]: run directory deleted
```

```mermaid
flowchart TD
    A["bind_private_offload_root(model)"] --> B{"model._private_offload_run_root set?"}
    B -- Yes --> C["release_private_offload_root(model)"]
    B -- No --> D["base = model._private_offload_base_root or configured root"]
    C --> D
    D --> E["makedirs base/run-uuid; store run root and base root"]
    E --> F["weakref.finalize(model, _remove_private_offload_root, run_root)"]
    F --> G["private config copy points at run-uuid"]
    G --> H{"run outcome"}
    H -- "loop raises" --> I["looper except: release_private_offload_root"]
    H -- "save() completes" --> K["bundles retained: model still materializes from them"]
    H -- "model collected" --> L["finalizer: _remove_private_offload_root"]
    K --> L
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class I failure;
```

`release_private_offload_root()` detaches the model's finalizer, clears the
ownership attributes, and calls `_remove_private_offload_root()`, which runs
`shutil.rmtree(..., ignore_errors=True)`. It is therefore safe from a finalizer
thread and at interpreter shutdown. A model that cannot be weakly referenced (a
lightweight test double) has no per-model finalizer; the explicit release helper
still works for it.

## Why `save()` does not delete the run directory

`save_quantized()` resolves every meta module through
`get_state_dict_for_save(model, offload_root=...)`, streams the tensors to the
checkpoint, and clears the temporary state dict. It does **not** materialize
the quantized modules back into memory. The modules stay on the meta device
with their accelerate offload hook pointing at the run directory, so an
in-session `model.generate()` after `save()` still reads those bundles
(`tests/test_planar_model_e2e.py` does exactly that). Deleting the directory at
the end of `save()` turns a supported in-session forward into a
`FileNotFoundError`. The bundles are released at teardown, which is when the
model stops being able to use them.

## Before and after

| Mechanism | Before #4409 | After |
| --- | --- | --- |
| Run directory creation | `bind_private_offload_root()` created `run-<uuid16>` and stored the private config (`gptqmodel/utils/offload.py:79-110` at `0b7582fea`) | Same isolation, plus ownership attributes and a `weakref.finalize` |
| Model teardown | Nothing removed the directory; an explicit root grew by one full model per run | The finalizer removes it when the model is collected |
| Failed quantization | Nothing removed the directory | `ModuleLooper.loop()` and `WeightOnlyLooper.loop()` release it in the failure branch |
| Same model, new run | A second run nested its directory inside the previous private path, and the old path was never removed | The previous root is released before the new sibling is created |
| `save()` | Nothing removed the directory | Still nothing: the live model reads the bundles, and a second `save()` reuses them |
| Interpreter exit | Auto temporary root removed, explicit run directory leaked | The finalizer runs on collection; the automatic root keeps its exit cleanup |

## Failure and ambiguity contract

| Situation | Behavior |
| --- | --- |
| Quantization loop raises (including a checkpoint stop) | Run directory is deleted; published checkpoint artifacts are store-owned copies and resume recreates the disposable indexes |
| `save()` raises | Run directory is kept so the caller can retry the save; teardown removes it later |
| `save()` succeeds | Run directory is kept because the live model still reads it |
| Model is not weak-referenceable | No per-model finalizer; callers can use `release_private_offload_root()` |
| Process is killed with `SIGKILL` | Out of scope: no Python cleanup hook runs. The directory remains under the explicit root |

The introducing change is [PR #4297] for isolation; [#4368] and [PR #4409]
added the cleanup contract, and [PR #4412] keeps `save()` from deleting bundles
the live model still needs.

[#4202]: https://github.com/ModelCloud/GPTQModel/issues/4202
[#4368]: https://github.com/ModelCloud/GPTQModel/issues/4368
[PR #4297]: https://github.com/ModelCloud/GPTQModel/pull/4297
[PR #4409]: https://github.com/ModelCloud/GPTQModel/pull/4409
[PR #4412]: https://github.com/ModelCloud/GPTQModel/pull/4412
