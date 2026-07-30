---
name: gptqmodel-sharding
description: Reshard, save, and load quantized checkpoints per-layer or per-size. Use for the gptqmodel.reshard API, ShardStrategy, parallel shard writes, and multi-GPU save/load integration.
---

# GPT-QModel checkpoint sharding

Use when changing how quantized checkpoints are split into safetensors shards, saved with `ShardStrategy`,
or loaded across devices.

## When to use

- Calling or modifying `gptqmodel.reshard(...)`.
- Changing `ShardStrategy.PER_LAYER` (currently the only supported strategy) or adding a new one.
- Parallelizing final shard writes with `ThreadPoolExecutor`.
- Adding or fixing per-layer sharding integration tests.
- Ensuring a resharded checkpoint reloads cleanly into `GPTQModel.load` / `from_quantized`.

## Key files

- `gptqmodel/__init__.py` (`from .utils.reshard import ShardStrategy, reshard`)
- `gptqmodel/utils/reshard.py` (`reshard`, `_resolve_layer_split_group`, `_layer_sort_key`)
- `gptqmodel/quantization/config.py` (`ShardStrategy` enum)
- `gptqmodel/models/base.py` (`save` method, passes `shard_strategy` to `save_quantized`)
- `gptqmodel/models/writer.py` (`save_quantized`, `_normalize_shard_strategy_for_save`)
- `tests/test_shard*.py`, `tests/models/test_llama3_2_lazy_turtle_memory.py`

## Workflow

1. **Pick the shard strategy.**
   - `PER_LAYER` (currently the only supported `ShardStrategy`) puts each transformer layer and its submodules in one shard; good for per-layer lazy loading.
   - Non-layer tensors (embeddings, final norm, lm_head) are grouped into a separate non-layer shard.
   - Verify the strategy is recorded in `config.json` and the safetensors index.

2. **Build a streaming routing plan.**
   - Parse all source shard headers once.
   - Route tensors to per-output-group temporary partial files; close source shards before opening the next.
   - Keep peak memory bounded by one source shard plus `num_write_workers` output buffers.

3. **Parallelize final shard writes.**
   - Use a bounded `ThreadPoolExecutor` (`num_write_workers`) to merge partials into final safetensors shards.
   - Match output shard size limits to the target serving/loading pattern.

4. **Test reload.**
   - Save the resharded checkpoint, then `GPTQModel.load(..., device_map=...)` or `from_quantized`.
   - Run `tests/test_shard*` and a small end-to-end model test (Llama-3.2-1B, Qwen3-MoE-1L).
   - Compare a forward pass against the un-sharded reference.

## Anti-patterns

- Do not materialize the whole checkpoint in host RAM before writing shards.
- Do not spawn unbounded writer threads; cap by I/O bandwidth and CPU cores.
- Do not change `ShardStrategy` without updating `config.json`/index and reload tests.
- Do not mix `ShardStrategy` metadata with non-resharded checkpoints without a migration note.
