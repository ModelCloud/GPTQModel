"""Micro-benchmark of the module-map construction in LazyTurtle.materialize_submodule.

Reports the cost of the old `dict(target_model.named_modules())` scan versus the
targeted ancestor+descendant map used after the PR #134 optimization.
"""

from __future__ import annotations

import statistics
import time
from typing import List, Tuple

import torch
import torch.nn as nn


def _build_shell(num_leaves: int, depth: int = 4) -> nn.Module:
    """Build a shell with `num_leaves` meta Linear modules under a nested tree."""
    torch.set_default_dtype(torch.float16)
    root = nn.Module()
    root.layers = nn.ModuleList()
    leaves_per_layer = max(1, num_leaves // max(1, depth))
    for i in range(depth):
        layer = nn.Module()
        layer.blocks = nn.ModuleList()
        for j in range(leaves_per_layer):
            block = nn.Module()
            block.proj = nn.Linear(64, 64, bias=False, device="meta")
            layer.blocks.append(block)
        root.layers.append(layer)
    return root


def _old_map_build(target_model: nn.Module) -> dict[str, nn.Module]:
    return dict(target_model.named_modules())


def _new_map_build(target_model: nn.Module, target_submodule: nn.Module, module_path: str) -> dict[str, nn.Module]:
    modules_by_name: dict[str, nn.Module] = {"": target_model}
    parts = module_path.split(".")
    for i in range(len(parts)):
        prefix = ".".join(parts[: i + 1])
        try:
            modules_by_name[prefix] = target_model.get_submodule(prefix)
        except (AttributeError, IndexError, KeyError):
            break
    modules_by_name[module_path] = target_submodule
    for subname, submod in target_submodule.named_modules():
        full_name = f"{module_path}.{subname}" if subname else module_path
        modules_by_name[full_name] = submod
    return modules_by_name


def _timeit(fn, repeat: int = 5) -> Tuple[float, float]:
    times: List[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.mean(times), statistics.stdev(times)


def main() -> None:
    scales = [1_000, 5_000, 10_000, 27_000, 50_000]
    repeats = max(3, 30_000 // max(scales))  # fewer reps for larger models

    print("LazyTurtle module-map construction: old full-model scan vs. targeted map\n")
    header = f"{'total_modules':>13} {'old_mean_ms':>12} {'new_mean_ms':>12} {'speedup':>9} {'saved_ms':>10}"
    print(header)
    print("-" * len(header))

    for num_leaves in scales:
        shell = _build_shell(num_leaves)
        total_modules = len(list(shell.named_modules()))
        target_submodule = shell.layers[0].blocks[0]
        module_path = "layers.0.blocks.0"

        def old():
            _old_map_build(shell)

        def new():
            _new_map_build(shell, target_submodule, module_path)

        old_mean, _ = _timeit(old, repeats)
        new_mean, _ = _timeit(new, repeats)

        old_ms = old_mean * 1000
        new_ms = new_mean * 1000
        speedup = old_ms / new_ms if new_ms else 0.0
        saved_ms = old_ms - new_ms

        print(f"{total_modules:13d} {old_ms:12.3f} {new_ms:12.3f} {speedup:9.1f} {saved_ms:10.3f}")

    print("\nLarge MoE model extrapolation (total_modules ≈ 27,000, ~775 materialize_submodule calls/layer):")
    shell = _build_shell(27_000)
    target_submodule = shell.layers[0].blocks[0]
    module_path = "layers.0.blocks.0"
    old_mean, _ = _timeit(lambda: _old_map_build(shell), 5)
    new_mean, _ = _timeit(lambda: _new_map_build(shell, target_submodule, module_path), 5)
    old_per_call = old_mean
    new_per_call = new_mean
    calls_per_layer = 775
    layers = 33
    total_old = old_per_call * calls_per_layer * layers
    total_new = new_per_call * calls_per_layer * layers
    print(f"  old map time per call: {old_per_call * 1000:.3f} ms")
    print(f"  new map time per call: {new_per_call * 1000:.3f} ms")
    print(f"  calls per layer:        {calls_per_layer}")
    print(f"  layers:                 {layers}")
    print(f"  old total map time:     {total_old:.3f} s")
    print(f"  new total map time:     {total_new:.3f} s")
    print(f"  map time saved:         {total_old - total_new:.3f} s")
    print(f"  full-model named_modules() scans eliminated: {calls_per_layer * layers:,}")


if __name__ == "__main__":
    main()
