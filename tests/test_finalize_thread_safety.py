# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Concurrency regression tests for submodule finalization thread safety.

These tests exercise the core invariant that made background submodule
finalization safe: per-leaf replacement + per-parent locking + no whole-tree
`named_modules()` scans from inside the finalization worker. They only run under
a free-threaded Python build with GIL disabled, because that is the environment
where the original race was observed.
"""

import tempfile
from concurrent.futures import ThreadPoolExecutor, wait

import pytest
from torch import nn

from gptqmodel.utils.model import recurse_setattr
from gptqmodel.utils.module_locks import parent_module_lock
from gptqmodel.utils.offload import offload_to_disk
from gptqmodel.utils.python import has_gil_disabled


pytestmark = pytest.mark.skipif(
    not has_gil_disabled(),
    reason="Concurrent module-tree mutations only race under free-threaded GIL=0 Python",
)


class _TinyExpert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # One leaf per expert; 64x64 float32 is >4KB so `offload_to_disk`
        # writes an actual safetensors payload.
        self.down_proj = nn.Linear(dim, dim, bias=True)


class _TinyMlp(nn.Module):
    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.experts = nn.ModuleList([_TinyExpert(dim) for _ in range(num_experts)])


class _TinyLayer(nn.Module):
    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.mlp = _TinyMlp(num_experts, dim)


class _TinyMoEModel(nn.Module):
    def __init__(self, num_layers: int, num_experts: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(num_experts, dim) for _ in range(num_layers)])

    def down_proj_names(self):
        names = []
        for li, layer in enumerate(self.layers):
            for ei, expert in enumerate(layer.mlp.experts):
                names.append(f"layers.{li}.mlp.experts.{ei}.down_proj")
        return names


def _replace_and_offload(model: nn.Module, name: str, disk_path: str, dim: int) -> str:
    """Finalizes one MoE submodule: replace the leaf and offload it to disk."""
    # Simulate the post-quant state: a new module ready to be installed.
    new_module = nn.Linear(dim, dim, bias=True)

    # Replace the original leaf under its direct parent lock. This serializes
    # writes to siblings that share the same parent, but leaves unrelated
    # branches free to finalize concurrently.
    with parent_module_lock(name):
        recurse_setattr(model, name, new_module)

    # Offload using the already-known full name so `offload_to_disk()` never
    # calls `model.named_modules()` and races with sibling replacements.
    offload_to_disk(
        module=new_module,
        model=model,
        disk_path=disk_path,
        module_full_name=name,
    )
    return name


def test_concurrent_moe_leaf_finalization_is_race_free():
    """Many threads replace and offload different MoE experts in parallel.

    This matches the production background-finalization pattern used by
    `stage_layer` and `weight_only_looper`: each worker replaces exactly one
    leaf, serializes the `setattr`/hook mutations on that leaf's direct
    parent, and never performs a whole-model `named_modules()` scan. After all
    workers finish the model tree must still be internally consistent.
    """
    num_layers = 4
    num_experts = 16
    dim = 64
    model = _TinyMoEModel(num_layers, num_experts, dim)
    names = model.down_proj_names()

    with tempfile.TemporaryDirectory() as disk_path:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(_replace_and_offload, model, name, disk_path, dim)
                for name in names
            ]
            wait(futures)
            for future in futures:
                # Surface any exceptions from the worker threads.
                future.result()

    # Final tree sanity check: every named path resolves to the same object
    # that the final iteration of `named_modules()` returns. If a concurrent
    # replacement had corrupted a parent `_modules` dict, this would diverge.
    for name, module in model.named_modules():
        current = model.get_submodule(name)
        assert current is module, f"final mismatch at {name}"

    # Each replaced leaf should still be reachable by the name we finalized.
    for name in names:
        assert isinstance(model.get_submodule(name), nn.Linear)
