"""Lifecycle telemetry profiler for LazyTurtle turtle -> shell -> disk offload.

Emulates a 1/10-scale synthetic MoE model (33 layers, 25 experts per layer,
hidden 3072 -> 307, MoE intermediate 1024 -> 102). For each layer it materializes
all leaf modules from a synthetic safetensors checkpoint into a meta shell, then
offloads each leaf to disk, recording entry/exit timers for the materialize and
offload calls. No quantization math is run.
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from gptqmodel.utils.model import move_to
from gptqmodel.utils.offload import offload_to_disk
from gptqmodel.utils.structure import LazyTurtle


try:
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - script diagnostic
    raise SystemExit("`safetensors` is required for this profiler") from exc


NUM_LAYERS = int(os.environ.get("LAZY_LIFECYCLE_LAYERS", 33))
NUM_EXPERTS = int(os.environ.get("LAZY_LIFECYCLE_EXPERTS", 25))
HIDDEN_SIZE = int(os.environ.get("LAZY_LIFECYCLE_HIDDEN", 307))
MOE_INTERMEDIATE = int(os.environ.get("LAZY_LIFECYCLE_MOE_HIDDEN", 102))
DTYPE = torch.float16


@dataclass
class Telemetry:
    events: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        phase: str,
        name: str,
        start: float,
        end: float,
        layer: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            {
                "phase": phase,
                "name": name,
                "start": start,
                "end": end,
                "duration": end - start,
                "layer": layer,
                **kwargs,
            }
        )

    def durations_for_layer(self, phase: str, layer: int) -> List[float]:
        return [e["duration"] for e in self.events if e["phase"] == phase and e.get("layer") == layer]


class ShellModel(nn.Module):
    """Meta-device shell that mirrors the synthetic MoE structure."""

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        hidden: int,
        moe_hidden: int,
    ) -> None:
        super().__init__()
        torch.set_default_dtype(DTYPE)
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict()
        for i in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False, device="meta")
            layer.self_attn.k_proj = nn.Linear(hidden, hidden, bias=False, device="meta")
            layer.self_attn.v_proj = nn.Linear(hidden, hidden, bias=False, device="meta")
            layer.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False, device="meta")
            layer.mlp = nn.Module()
            layer.mlp.gate = nn.Linear(hidden, num_experts, bias=False, device="meta")
            layer.mlp.experts = nn.ModuleList()
            for _ in range(num_experts):
                expert = nn.Module()
                expert.gate_proj = nn.Linear(hidden, moe_hidden, bias=False, device="meta")
                expert.up_proj = nn.Linear(hidden, moe_hidden, bias=False, device="meta")
                expert.down_proj = nn.Linear(moe_hidden, hidden, bias=False, device="meta")
                layer.mlp.experts.append(expert)
            self.model.layers[str(i)] = layer

        for p in self.parameters():
            p.requires_grad = False


def make_checkpoint(
    checkpoint_dir: str,
    num_layers: int,
    num_experts: int,
    hidden: int,
    moe_hidden: int,
) -> None:
    torch.set_default_dtype(DTYPE)
    tensors: Dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"model.layers.{i}.mlp.gate.weight"] = torch.randn(num_experts, hidden)
        for j in range(num_experts):
            tensors[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = torch.randn(moe_hidden, hidden)
            tensors[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = torch.randn(moe_hidden, hidden)
            tensors[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = torch.randn(hidden, moe_hidden)
    save_file(tensors, os.path.join(checkpoint_dir, "model.safetensors"))


def iter_leaf_paths(layer_idx: int, num_experts: int) -> List[str]:
    paths: List[str] = []
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        paths.append(f"model.layers.{layer_idx}.self_attn.{proj}")
    paths.append(f"model.layers.{layer_idx}.mlp.gate")
    for j in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            paths.append(f"model.layers.{layer_idx}.mlp.experts.{j}.{proj}")
    return paths


def linear_fit(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Return (slope, intercept, r_squared) for a simple OLS fit."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return slope, intercept, r2


def run_lazy_turtle_lifecycle(
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
    hidden: int = HIDDEN_SIZE,
    moe_hidden: int = MOE_INTERMEDIATE,
) -> Dict[str, Any]:
    os.environ.setdefault("GPTQMODEL_LAZY_TURTLE_PARALLEL_LOAD_WORKERS", "1")
    torch.set_default_dtype(DTYPE)

    telemetry = Telemetry()
    with tempfile.TemporaryDirectory() as checkpoint_dir, tempfile.TemporaryDirectory() as offload_dir:
        make_checkpoint(checkpoint_dir, num_layers, num_experts, hidden, moe_hidden)
        shell = ShellModel(num_layers, num_experts, hidden, moe_hidden)
        device = torch.device("cpu")

        turtle = LazyTurtle(
            model_local_path=checkpoint_dir,
            config=SimpleNamespace(),
            target_model=shell,
            hf_conversion_map_reversed=None,
        )

        per_layer: List[Dict[str, Any]] = []
        for layer_idx in range(num_layers):
            layer_start = time.perf_counter()
            layer = shell.get_submodule(f"model.layers.{layer_idx}")
            paths = iter_leaf_paths(layer_idx, num_experts)

            materialize_total = 0.0
            for path in paths:
                sub = shell.get_submodule(path)
                t0 = time.perf_counter()
                turtle.materialize_submodule(
                    target_model=shell,
                    target_submodule=sub,
                    device=device,
                    module_path=path,
                    recurse=True,
                    tie_weights=False,
                    show_progress=False,
                )
                t1 = time.perf_counter()
                telemetry.record("materialize", path, t0, t1, layer=layer_idx)
                materialize_total += t1 - t0

            move_start = time.perf_counter()
            move_to(layer, device)
            move_end = time.perf_counter()
            telemetry.record("move_to", f"model.layers.{layer_idx}", move_start, move_end, layer=layer_idx)

            offload_total = 0.0
            for path in paths:
                sub = shell.get_submodule(path)
                t0 = time.perf_counter()
                offload_to_disk(
                    module=sub,
                    model=shell,
                    disk_path=offload_dir,
                )
                t1 = time.perf_counter()
                telemetry.record("offload", path, t0, t1, layer=layer_idx)
                offload_total += t1 - t0

            layer_end = time.perf_counter()
            per_layer.append(
                {
                    "layer": layer_idx,
                    "modules": len(paths),
                    "materialize_total": materialize_total,
                    "offload_total": offload_total,
                    "move_total": move_end - move_start,
                    "layer_total": layer_end - layer_start,
                }
            )

    return {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden": hidden,
        "moe_hidden": moe_hidden,
        "per_layer": per_layer,
        "telemetry": telemetry,
    }


def print_report(result: Dict[str, Any]) -> None:
    per_layer = result["per_layer"]
    print("\nLazyTurtle lifecycle telemetry (turtle -> shell -> disk)\n")
    header = f"{'layer':>5} {'modules':>7} {'mat_total':>10} {'off_total':>10} {'move_total':>10} {'layer_total':>11}"
    print(header)
    print("-" * len(header))
    for row in per_layer:
        print(
            f"{row['layer']:5d} {row['modules']:7d} "
            f"{row['materialize_total']:10.3f} {row['offload_total']:10.3f} "
            f"{row['move_total']:10.3f} {row['layer_total']:11.3f}"
        )

    xs = [r["layer"] for r in per_layer]
    layer_totals = [r["layer_total"] for r in per_layer]
    mat_totals = [r["materialize_total"] for r in per_layer]
    off_totals = [r["offload_total"] for r in per_layer]

    def summarize(name: str, ys: List[float]) -> None:
        slope, intercept, r2 = linear_fit(xs, ys)
        q = max(1, len(ys) // 4)
        first_avg = statistics.mean(ys[:q])
        last_avg = statistics.mean(ys[-q:])
        print(f"\n{name}:")
        print(f"  first_quarter_avg = {first_avg:.3f}s")
        print(f"  last_quarter_avg  = {last_avg:.3f}s")
        if first_avg:
            print(f"  last/first ratio  = {last_avg / first_avg:.2f}")
        else:
            print("  last/first ratio  = N/A")
        print(f"  linear slope      = {slope:.6f}s/layer")
        print(f"  R^2               = {r2:.4f}")

    summarize("layer_total", layer_totals)
    summarize("materialize_total", mat_totals)
    summarize("offload_total", off_totals)

    telemetry: Telemetry = result["telemetry"]
    early_layer_off = telemetry.durations_for_layer("offload", 0) + telemetry.durations_for_layer("offload", 1)
    late_layer_off = telemetry.durations_for_layer("offload", max(0, len(per_layer) - 2))
    if early_layer_off:
        print(f"\nper-module median offload early = {statistics.median(early_layer_off):.4f}s")
    if late_layer_off:
        print(f"per-module median offload late  = {statistics.median(late_layer_off):.4f}s")


def main() -> None:
    result = run_lazy_turtle_lifecycle()
    print_report(result)


if __name__ == "__main__":
    main()
