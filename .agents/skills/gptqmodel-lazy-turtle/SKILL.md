---
name: gptqmodel-lazy-turtle
description: Materialize large local dense or safetensors-sharded checkpoints without building a full CPU "turtle" model. Use for LazyTurtle, meta-device shells, batch submodule loading, pinned-memory shard caches, pre-quantize loading, and checkpoint index conversion.
---

# GPT-QModel LazyTurtle loading

Use when the task touches `LazyTurtle`, `materialize_submodule`, batched safetensors loading,
pinned-memory LRU, or pre-quantize checkpoint loading.

## When to use

- Loading a local dense or sharded checkpoint that is too large for full CPU materialization.
- Converting `.bin`/`.pt`/`.pth`/`.ckpt` checkpoints into temporary safetensors for LazyTurtle.
- Materializing only the submodules needed for the current quantization or inference stage.
- Diagnosing slow load, high host memory, or file-handle exhaustion during model loading.
- Adding batch-load, ancestor-map, or subtree-map optimizations for MoE or layered checkpoints.

## Key files

- `gptqmodel/utils/structure.py` (`LazyTurtle` class)
- `gptqmodel/models/base.py` (`load` path, `materialize_passthrough_modules_for_*`)
- `tests/test_lazy_turtle_*.py`, `tests/test_meta_materialize.py`, `tests/test_offload_files.py`
- `scripts/validate_lazy_turtle_layer_migration.py`, `scripts/benchmark_lazy_turtle_*.py`

## Workflow

1. **Verify the checkpoint is suitable for LazyTurtle.**
   - Local path with a safetensors `model.safetensors.index.json` or a single `model.safetensors`.
   - For pickle weights, LazyTurtle will convert them to temporary safetensors once; ensure the temp dir has space.

2. **Provide the structural context up front.**
   - Pass `module_tree` and `hf_conversion_map_reversed` so alias resolution does not require a full model walk.
   - Build `moe_alias_specs` once in `LazyTurtle.__init__` and reuse them.

3. **Bound host memory.**
   - Set `lazy_turtle_max_pinned_gb` (default 4 GB) based on the host headroom; use `<=0` for unlimited.
   - Pinned shard ranges are cached in an LRU; evict old shards before opening new ones when the cap is hit.

4. **Batch related submodules.**
   - Materialize all experts in a layer in one call path to reuse open safetensors file handles and pinned ranges.
   - Close shard handlers after a layer group is finished unless the next group needs the same shards.

5. **Validate with focused tests.**
   - `pytest -q tests/test_meta_materialize.py`
   - `pytest -q tests/test_lazy_turtle_materialize_map.py`
   - `pytest -q tests/test_offload_files.py`
   - Run `scripts/benchmark_lazy_turtle_materialize_map.py` for load-time regression checks.

## Anti-patterns

- Do not build a full CPU `turtle` model for large checkpoints; it defeats the purpose.
- Do not open/close safetensors files per tensor; reuse handles across a batch.
- Do not ignore `hf_conversion_map_reversed` aliases; mismatched names cause silent wrong-weight loads.
- Do not leave `max_pinned_gb` unbounded on memory-constrained hosts.
- Do not hold shard handlers across unrelated layers without an LRU eviction policy.
