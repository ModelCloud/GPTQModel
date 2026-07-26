# Submodule Finalization Lifecycle and Thread Safety

## What finalization does

During GPTQ/AWW/QQQ/ParoQuant/EXL3/weight-only quantization, each layer's
modules go through a **finalize** step that:

1. **Creates** the quantized runtime module (`create_quant_module` / `create_exllamav3_module`).
2. **Packs** the quantized weights, scales, and zeros into that module (`pack_module` / `pack_original`).
3. **Replaces** the original `nn.Linear`/`Conv1D` leaf in the model tree with the new module.
4. Optionally **offloads** the packed module's tensors to disk (`offload_to_disk`).

On large MoE models this step dominates wall time because there may be
hundreds of experts per layer and each finalize does packing plus a disk write.

## Lifecycle controls

There are two independent concurrency switches.

### 1. `stage_layer` (GPTQ / AWQ / QQQ / ParoQuant / EXL3)

Controlled by `_should_drain_finalize_futures_synchronously()` in
`gptqmodel/looper/stage_layer.py`.

- The default (`wait_for_submodule_finalizers=False`) is **parallel / async**:
  finalizer futures for the current layer are submitted to
  `DEVICE_THREAD_POOL` and a background `SubmoduleFinalizeWatcher` drains them
  while the main loop moves on to the next layer.
- `QuantizeConfig.wait_for_submodule_finalizers=True` forces **synchronous**
  draining.
- AWQ and ParoQuant finalizers are always drained synchronously for
  correctness / VRAM reasons unrelated to the module-tree race.

### 2. `weight_only_looper` (RTN / GGUF / FP8 / BitsAndBytes)

Controlled by `_finalize_subset_modules()` in
`gptqmodel/looper/weight_only_looper.py`.

- Previously `use_parallel_finalize` was only `True` when multiple **target
  devices** were present, so CPU-only or single-device runs serialized every
  module finalize.
- After the thread-safety fix it defaults to parallel whenever more than one
  module needs finalization. The device thread pool handles worker limits.

## The thread-safety problem

`nn.Module.named_modules()`, `state_dict()`, and helper functions built on
them (e.g. `find_modules(model.model)`, `get_module_fullname()`) are Python
generators over mutable `nn.Module` internal dicts (`_modules`, `_parameters`,
`_buffers`).

When one thread replaces a leaf in the model tree with `setattr` while another
thread is iterating `named_modules()` over a shared ancestor, the iterator can
return a stale object for a path. The walker then has a module whose `id`
does not match `get_submodule(path)`, producing silent data corruption:

```text
RuntimeError: [finalizer] named_modules/get_submodule mismatch at layers.1.experts.13.down:
  named_modules id=6030404374032 get_submodule id=6030404366608
```

A CPU-only `PYTHON_GIL=0` reproducer (`reproduce_module_tree_race_layers.py`)
confirms this by racing a finalizer that replaces a leaf in layer 0 against a
second thread that replaces a leaf in layer 1 while repeatedly calling
`model.named_modules()`.

Importantly, the race is **not** from two threads touching the same leaf. It is
from two threads touching **different leaves under a shared ancestor** (e.g.
different experts in the same `mlp.experts` `ModuleList`) while one of them is
walking the whole tree.

## The fix

The fix follows **Option A**: remove whole-tree scans from the hot finalize
path instead of adding a global reader-writer lock around `nn.Module`.

### Per-processor finalize

Every processor's `submodule_finalize()` now:

1. Captures `original_layer = module.module` before replacement.
2. Calls `create_quant_module(...)` and receives the new quantized module back
   as a return value.
3. Builds local one-entry dicts:

   ```python
   qModules = {module.full_name: qmodule}
   layers   = {module.full_name: original_layer}
   ```

4. Calls `pack_module()` with those dicts, so `pack_module` does not need to
   call `find_modules(model.model)` to locate the target.

This removes `find_modules(model.model)` from `submodule_finalize()` in
`gptqmodel/looper/gptq_processor.py`,
`gptqmodel/looper/weight_only_processor.py`,
`gptqmodel/looper/awq_processor.py`,
`gptqmodel/looper/qqq_processor.py`,
`gptqmodel/looper/paroquant_processor.py`, and
`gptqmodel/looper/exllamav3_processor.py`.

`create_exllamav3_module()` was also updated to return the newly installed
module so EXL3 follows the same pattern.

### `offload_to_disk()`

`offload_to_disk()` in `gptqmodel/utils/offload.py` now takes an optional
`module_full_name` argument. Callers pass the already-known dotted path, which
lets `_offload_to_disk_impl()` skip `get_module_fullname()` — and therefore
`model.named_modules()` — entirely.

`stage_layer._finalize_on_worker()` and `weight_only_looper._offload_quantized_module()`
were updated to pass the quantized module object returned by
`submodule_finalize()` and its `full_name` into `offload_to_disk()`.

### Parent locking

The actual tree mutation (`recurse_setattr` inside `create_quant_module` and
`disk_offload` hook installs inside `_offload_disk`) is still wrapped with
`parent_module_lock(parent_key)`. This lock is keyed on the **direct parent**
of the leaf being replaced, not the whole model:

- Replacing `model.layers.7.mlp.experts.13.down_proj` acquires the lock for
  `model.layers.7.mlp.experts.13`.
- A concurrent finalizer for `model.layers.7.mlp.experts.14.down_proj`
  acquires a different lock for `model.layers.7.mlp.experts.14`.
- Whole-tree scans are gone, so iterators no longer see stale objects.

### `weight_only_looper` default parallel

With the tree scans removed, `weight_only_looper._finalize_subset_modules()`
was changed from

```python
use_parallel_finalize = len(finalize_tasks) > 1 and len(unique_targets) > 1
```

to

```python
use_parallel_finalize = len(finalize_tasks) > 1
```

so a layer with many CPU-finalized experts is packed/offloaded in parallel
through the device thread pool.

## Files involved

- `gptqmodel/utils/model.py` — `create_quant_module()` now returns the new quantized module.
- `gptqmodel/utils/offload.py` — `offload_to_disk()` accepts an optional `module_full_name` and avoids `get_module_fullname()` when it is supplied.
- `gptqmodel/looper/gptq_processor.py`
- `gptqmodel/looper/weight_only_processor.py`
- `gptqmodel/looper/awq_processor.py`
- `gptqmodel/looper/qqq_processor.py`
- `gptqmodel/looper/paroquant_processor.py`
- `gptqmodel/looper/exllamav3_processor.py`  
  All use the captured original layer and the returned quantized module instead of `find_modules(model.model)`.
- `gptqmodel/looper/stage_layer.py` — passes the returned `qmodule` and `module.full_name` into `offload_to_disk()`; documents the lifecycle control for async vs. sync draining.
- `gptqmodel/looper/weight_only_looper.py` — defaults to parallel finalization for any multi-module subset.

## Validation

- `ruff check` on all changed paths: passed.
- `git diff --check`: passed.
- CPU-only tests passed locally:
  - `python3 -m pytest tests/test_weight_only_looper.py -q`
  - `python3 -m pytest tests/test_looper_helpers.py -q`
  - `python3 -m pytest tests/test_weight_only_config.py -q`
- A standalone `PYTHON_GIL=0` CPU reproducer demonstrates the old race and
  confirms that the scan-free pattern does not reproduce it.

## Notes for reviewers

- Do **not** add `find_modules(model.model)` or `get_module_fullname()` calls
  back into the finalize path. If you need to locate a module by name, use the
  `full_name` already carried on the `NamedModule` or the module object returned
  by `create_quant_module()`.
- `parent_module_lock` is intentionally per-parent, not global. That is what
  allows sibling experts to finalize in parallel while still serializing writes
  that touch the same direct parent container.
- AWQ and ParoQuant still drain synchronously in `stage_layer` because their
  correctness / memory constraints are independent of this race fix.
