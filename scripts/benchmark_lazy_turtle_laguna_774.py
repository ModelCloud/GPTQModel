#!/usr/bin/env python
"""Benchmark LazyTurtle grouped loading of a Laguna-S-2.1 MoE layer.

This script synthesizes a 4-shard safetensors checkpoint that mirrors the
weight layout of one Laguna-S-2.1 MoE decoder layer (774 grouped tensors):

- self_attn: q/k/v/o/g_proj (5)
- mlp.gate (router) (1)
- mlp.experts.0..255: gate_proj, up_proj, down_proj (768)

It then loads all 774 tensors in one batched LazyTurtle.materialize_submodules
call on a single GPU and reports wall/GPU time, total bytes, and throughput.

Run on physical GPU 3 as the only visible CUDA device:

    CUDA_VISIBLE_DEVICES=3 PYTHON_GIL=0 python scripts/benchmark_lazy_turtle_laguna_774.py

"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from gptqmodel.utils.structure import LazyTurtle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LazyTurtle grouped loading for one Laguna-S-2.1 MoE layer."
    )
    parser.add_argument(
        "--physical-gpus",
        type=int,
        nargs="+",
        default=None,
        help="Physical GPU ids to verify are idle (default: first visible CUDA device).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Process-local CUDA ordinals to round-robin load into (default: cuda:0).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After loading, compare each parameter to the checkpoint source and assert equality.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Number of MoE experts (default: 256).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of safetensors shards to create (default: 4).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Worker counts to benchmark (default: 1 8).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs before each timed config (default: 0).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Number of timed iterations per config (default: 3).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Checkpoint dtype (default: float16).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Directory to write the synthetic checkpoint; a temp dir is used if not set.",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Do not delete the synthetic checkpoint after the benchmark.",
    )
    parser.add_argument(
        "--idle-samples",
        type=int,
        default=3,
        help="Consecutive idle nvidia-smi samples required (default: 3).",
    )
    parser.add_argument(
        "--idle-util-threshold",
        type=int,
        default=5,
        help="GPU utilization percent threshold for idle gate (default: 5).",
    )
    parser.add_argument(
        "--idle-mem-threshold",
        type=int,
        default=100,
        help="GPU memory-used MiB threshold for idle gate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible checkpoint contents.",
    )
    args = parser.parse_args()
    if args.physical_gpus is None:
        args.physical_gpus = [int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])]
    if args.devices is None:
        args.devices = ["cuda:0"]
    return args


def _nvidia_smi_query() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(", ")]
        if len(parts) < 7:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "pci.bus_id": parts[1],
                "uuid": parts[2],
                "name": parts[3],
                "memory.total_mib": float(parts[4]),
                "memory.used_mib": float(parts[5]),
                "utilization.gpu": float(parts[6]),
            }
        )
    return gpus


def _idle_gpu_gate(physical_gpus: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    print(f"[idle-gate] waiting for physical GPUs {physical_gpus} to be idle...")
    consecutive = 0
    accepted: list[dict[str, Any]] | None = None
    while consecutive < args.idle_samples:
        gpus = _nvidia_smi_query()
        selected = [g for g in gpus if g["index"] in physical_gpus]
        if len(selected) != len(physical_gpus):
            raise RuntimeError(f"physical GPUs {physical_gpus} not all found by nvidia-smi")
        if all(
            g["utilization.gpu"] <= args.idle_util_threshold
            and g["memory.used_mib"] <= args.idle_mem_threshold
            for g in selected
        ):
            consecutive += 1
            if accepted is None:
                accepted = selected
            print(
                f"[idle-gate] sample {consecutive}/{args.idle_samples}: "
                + ", ".join(f"gpu{g['index']} util={g['utilization.gpu']:.0f}% mem={g['memory.used_mib']:.0f}MiB" for g in selected)
            )
        else:
            consecutive = 0
            accepted = None
            print(
                "[idle-gate] rejected: "
                + ", ".join(f"gpu{g['index']} util={g['utilization.gpu']:.0f}% mem={g['memory.used_mib']:.0f}MiB" for g in selected)
            )
        if consecutive < args.idle_samples:
            time.sleep(0.5)
    assert accepted is not None
    for g in accepted:
        print(
            f"[idle-gate] accepted GPU {g['index']} {g['name']} "
            f"{g['pci.bus_id']} {g['uuid']}"
        )
    return accepted


def _laguna_config() -> dict[str, int]:
    return {
        "vocab_size": 100352,
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_experts": 256,
    }


def _shape_for_name(name: str, cfg: dict[str, int]) -> tuple[int, ...]:
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    num_heads = cfg["num_attention_heads"]
    num_kv = cfg["num_key_value_heads"]
    moe_hidden = cfg["moe_intermediate_size"]
    num_experts = cfg["num_experts"]

    if ".q_proj.weight" in name:
        return (num_heads * head_dim, hidden)
    if ".k_proj.weight" in name or ".v_proj.weight" in name:
        return (num_kv * head_dim, hidden)
    if ".o_proj.weight" in name:
        return (hidden, num_heads * head_dim)
    if ".g_proj.weight" in name:
        # "per-head" gating: one scalar gate per head.
        return (num_heads, hidden)
    if ".mlp.gate.weight" in name:
        return (num_experts, hidden)
    if ".gate_proj.weight" in name or ".up_proj.weight" in name:
        return (moe_hidden, hidden)
    if ".down_proj.weight" in name:
        return (hidden, moe_hidden)
    raise ValueError(f"unknown tensor name: {name}")


def _tensor_names(num_experts: int) -> list[str]:
    names: list[str] = []
    # 5 self-attention projections
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj"):
        names.append(f"model.layers.1.self_attn.{proj}.weight")
    # router gate
    names.append("model.layers.1.mlp.gate.weight")
    # 256 experts x 3 projections
    for i in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"model.layers.1.mlp.experts.{i}.{proj}.weight")
    return names


class _ShellModel(nn.Module):
    """A minimal shell with exactly the 774 Laguna-S-2.1 MoE layer leaves."""

    def __init__(self, num_experts: int, cfg: dict[str, int]):
        super().__init__()
        self.model = nn.Module()
        # Use a ModuleDict keyed by "1" so module paths exactly match checkpoint names.
        self.model.layers = nn.ModuleDict()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(cfg["hidden_size"], cfg["num_attention_heads"] * cfg["head_dim"], bias=False)
        layer.self_attn.k_proj = nn.Linear(cfg["hidden_size"], cfg["num_key_value_heads"] * cfg["head_dim"], bias=False)
        layer.self_attn.v_proj = nn.Linear(cfg["hidden_size"], cfg["num_key_value_heads"] * cfg["head_dim"], bias=False)
        layer.self_attn.o_proj = nn.Linear(cfg["num_attention_heads"] * cfg["head_dim"], cfg["hidden_size"], bias=False)
        # "per-head" gating projection
        layer.self_attn.g_proj = nn.Linear(cfg["hidden_size"], cfg["num_attention_heads"], bias=False)

        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(cfg["hidden_size"], cfg["num_experts"], bias=False)
        layer.mlp.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(cfg["hidden_size"], cfg["moe_intermediate_size"], bias=False)
            expert.up_proj = nn.Linear(cfg["hidden_size"], cfg["moe_intermediate_size"], bias=False)
            expert.down_proj = nn.Linear(cfg["moe_intermediate_size"], cfg["hidden_size"], bias=False)
            layer.mlp.experts.append(expert)

        self.model.layers["1"] = layer


def _create_checkpoint(
    out_dir: Path, num_experts: int, num_shards: int, dtype: torch.dtype, cfg: dict[str, int], seed: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    names = _tensor_names(num_experts)

    # Distribute tensors round-robin across shards.
    shards: list[dict[str, torch.Tensor]] = [{} for _ in range(num_shards)]
    weight_map: dict[str, str] = {}
    for idx, name in enumerate(names):
        shard_idx = idx % num_shards
        shape = _shape_for_name(name, cfg)
        shard_name = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
        # Real checkpoints store Linear weights as (out_features, in_features); that
        # already matches nn.Linear.weight.shape, so no LazyTurtle transpose is needed.
        t = torch.randn(shape, dtype=dtype).contiguous()
        shards[shard_idx][name] = t
        weight_map[name] = shard_name

    for shard_idx, shard_tensors in enumerate(shards):
        shard_name = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
        save_file(shard_tensors, str(out_dir / shard_name), metadata={"format": "pt"})

    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for s in shards for t in s.values())}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def _build_shell(num_experts: int, cfg: dict[str, int]) -> nn.Module:
    # Scope the meta default device to shell construction so the rest of the process
    # does not accidentally create empty meta tensors.
    prev_device = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        shell = _ShellModel(num_experts, cfg)
    finally:
        torch.set_default_device(prev_device)
    for p in shell.parameters():
        p.requires_grad = False
    return shell


def _module_path_to_submodule(shell: nn.Module, path: str) -> nn.Module:
    return shell.get_submodule(path)


def _verify_loaded(shell: nn.Module, out_dir: Path) -> bool:
    """Compare every loaded parameter to the checkpoint source to detect H2D corruption."""
    from safetensors.torch import load_file

    source: dict[str, torch.Tensor] = {}
    for shard_path in sorted(out_dir.glob("model-*-of-*.safetensors")):
        source.update(load_file(str(shard_path), device="cpu"))

    ok = True
    for name, param in shell.named_parameters():
        if name not in source:
            continue
        expected = source[name].to(device=param.device, dtype=param.dtype)
        if not torch.equal(param, expected):
            diff = (param - expected).abs().max().item()
            print(f"[verify] MISMATCH {name} max_abs_diff={diff} device={param.device}")
            ok = False
    if ok:
        print(f"[verify] all {len(list(shell.named_parameters()))} parameters matched checkpoint source")
    return ok


def _build_submodules(shell: nn.Module, names: list[str], devices: list[torch.device]) -> list[tuple[nn.Module, str, torch.device]]:
    submodules: list[tuple[nn.Module, str, torch.device]] = []
    for idx, name in enumerate(names):
        path = name.replace(".weight", "")
        mod = _module_path_to_submodule(shell, path)
        submodules.append((mod, path, devices[idx % len(devices)]))
    return submodules


def _benchmark_once(
    workers: int,
    turtle: LazyTurtle,
    out_dir: Path,
    num_experts: int,
    cfg: dict[str, int],
    devices: list[torch.device],
    args: argparse.Namespace,
) -> dict[str, Any]:
    os.environ["GPTQMODEL_LAZY_TURTLE_PARALLEL_LOAD_WORKERS"] = str(workers)

    # Recreate the shell each iteration so tensors start on meta and must be copied again,
    # while reusing the same LazyTurtle so the page-locked mmap registration is amortized.
    shell = _build_shell(num_experts, cfg)
    names = _tensor_names(num_experts)
    submodules = _build_submodules(shell, names, devices)

    for dev in devices:
        torch.cuda.synchronize(dev)
    wall_start = time.perf_counter()
    turtle.materialize_submodules(
        target_model=shell,
        submodules=submodules,
        non_blocking=False,
        show_progress=False,
    )
    for dev in devices:
        torch.cuda.synchronize(dev)
    wall_elapsed = time.perf_counter() - wall_start

    total_bytes = sum(p.numel() * p.element_size() for p in shell.parameters() if p.device.type == "cuda")
    total_mb = total_bytes / (1024 * 1024)

    verified = False
    if getattr(args, "verify", False):
        verified = _verify_loaded(shell, out_dir)

    del shell
    for dev in devices:
        torch.cuda.empty_cache()

    return {
        "wall_s": wall_elapsed,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "verified": verified,
    }


def _run_configuration(
    workers: int,
    turtle: LazyTurtle,
    out_dir: Path,
    num_experts: int,
    cfg: dict[str, int],
    devices: list[torch.device],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cold_times: list[float] = []
    for i in range(args.warmup):
        print(f"[workers={workers}] cold/warmup run {i + 1}/{args.warmup}")
        result = _benchmark_once(workers, turtle, out_dir, num_experts, cfg, devices, args)
        cold_times.append(result["wall_s"])

    results: list[dict[str, Any]] = []
    for i in range(args.iters):
        print(f"[workers={workers}] timed run {i + 1}/{args.iters}")
        results.append(_benchmark_once(workers, turtle, out_dir, num_experts, cfg, devices, args))

    steady_times = [r["wall_s"] for r in results]
    total_bytes = results[0]["total_bytes"]
    total_mb = results[0]["total_mb"]
    verified = all(r.get("verified", True) for r in results)
    return {
        "workers": workers,
        "total_mb": total_mb,
        "total_gb": total_mb / 1024,
        "cold_mean_s": statistics.mean(cold_times) if cold_times else 0.0,
        "steady_mean_s": statistics.mean(steady_times),
        "steady_min_s": min(steady_times),
        "throughput_mean_gbps": (total_bytes / statistics.mean(steady_times)) / 1e9,
        "verified": verified,
    }


def _print_results_table(rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 110)
    print(
        f"{'workers':>8} {'total_gb':>10} {'cold_s':>12} {'steady_s':>12} {'steady_min_s':>14} "
        f"{'throughput_gbps':>16} {'verified':>10}"
    )
    print("-" * 110)
    for r in rows:
        verified = "yes" if r.get("verified", True) else "NO"
        print(
            f"{r['workers']:>8} {r['total_gb']:>10.2f} {r['cold_mean_s']:>12.3f} "
            f"{r['steady_mean_s']:>12.3f} {r['steady_min_s']:>14.3f} "
            f"{r['throughput_mean_gbps']:>16.2f} {verified:>10}"
        )
    print("=" * 110)


def main() -> int:
    args = _parse_args()
    cfg = _laguna_config()

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    torch.set_default_dtype(dtype)

    devices = [torch.device(d) for d in args.devices]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    gpu_infos = _idle_gpu_gate(args.physical_gpus, args)
    print(
        f"[setup] physical GPUs {[g['index'] for g in gpu_infos]} mapped to process devices {args.devices}; "
        f"gil_enabled={sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else 'unknown'}"
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="laguna-774-"))
    print(f"[setup] writing synthetic checkpoint to {out_dir}")
    _create_checkpoint(out_dir, args.num_experts, args.num_shards, dtype, cfg, args.seed)

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(out_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
        target_model=_build_shell(args.num_experts, cfg),
    )
    if turtle is None:
        raise RuntimeError("LazyTurtle could not be created for the synthetic checkpoint")

    rows: list[dict[str, Any]] = []
    for workers in args.workers:
        print(f"\n[benchmark] worker count = {workers}")
        row = _run_configuration(workers, turtle, out_dir, args.num_experts, cfg, devices, args)
        rows.append(row)
        print(
            f"[benchmark] workers={workers}: cold={row['cold_mean_s']:.3f}s, steady={row['steady_mean_s']:.3f}s, "
            f"throughput={row['throughput_mean_gbps']:.2f} GB/s, verified={row.get('verified', True)}"
        )

    _print_results_table(rows)

    turtle.close_shard_handlers()
    del turtle
    for dev in devices:
        torch.cuda.empty_cache()

    if not args.keep_checkpoint and not args.out_dir:
        print(f"[cleanup] removing {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
