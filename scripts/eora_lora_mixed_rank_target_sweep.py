#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel.adapter.adapter import HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_FILE_NAME
from gptqmodel.adapter.peft import LoraConfig


DEFAULT_RANKS = [8, 32, 64, 128]
DEFAULT_DEFAULT_RANK = 256


@dataclass(frozen=True)
class Target:
    """Identifies one LoRA module whose rank can be varied independently."""

    key: str
    runtime_key: str
    layer: int
    module: str
    safe_name: str


def parse_csv_ints(value: str) -> list[int]:
    """Parse comma-separated integer values while preserving user order."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_rank_adapter(value: str) -> tuple[int, Path]:
    """Parse a rank=path adapter-bank argument."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("--rank-adapter must use rank=/path/to/adapter")
    rank_text, path_text = value.split("=", 1)
    try:
        rank = int(rank_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid rank in --rank-adapter: {rank_text}") from exc
    return rank, Path(path_text).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    """Build the CLI for distributed mixed-rank target evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build mixed EoRA/LoRA int4 adapters where one target module uses a lower rank and "
            "all other modules stay at rank256, then evaluate targets across multiple GPUs."
        )
    )
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--rank-adapter",
        action="append",
        default=[],
        type=parse_rank_adapter,
        help="Rank adapter source as rank=/path. Repeat for 8,32,64,128,256.",
    )
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in DEFAULT_RANKS))
    parser.add_argument("--default-rank", type=int, default=DEFAULT_DEFAULT_RANK)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--backend", default="marlin")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--target-regex", default=None)
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--baseline-variant", default="int4_g128")
    parser.add_argument("--copy-rank-bank", action="store_true", default=True)
    parser.add_argument("--no-copy-rank-bank", action="store_false", dest="copy_rank_bank")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-indexes", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def adapter_complete(path: Path) -> bool:
    """Return whether an adapter directory has the expected config and weights."""

    return (path / HF_ADAPTER_CONFIG_FILE_NAME).is_file() and (path / HF_ADAPTER_FILE_NAME).is_file()


def ensure_adapter_bank(args: argparse.Namespace, ranks: list[int]) -> dict[int, Path]:
    """Validate and optionally copy rank adapters into the sweep work directory."""

    supplied = dict(args.rank_adapter)
    needed = set(ranks) | {int(args.default_rank)}
    missing = sorted(rank for rank in needed if rank not in supplied)
    if missing:
        raise ValueError(f"Missing --rank-adapter entries for ranks: {missing}")

    bank: dict[int, Path] = {}
    bank_root = args.work_dir / "rank_bank"
    for rank in sorted(needed):
        source = supplied[rank]
        if not adapter_complete(source):
            raise FileNotFoundError(f"Rank {rank} adapter is incomplete: {source}")
        if args.copy_rank_bank:
            dest = bank_root / f"rank{rank}_int4_g128"
            dest.mkdir(parents=True, exist_ok=True)
            for filename in (HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_FILE_NAME):
                source_file = source / filename
                dest_file = dest / filename
                if not dest_file.exists() or source_file.stat().st_size != dest_file.stat().st_size:
                    shutil.copy2(source_file, dest_file)
            bank[rank] = dest
        else:
            bank[rank] = source
    return bank


def target_from_key(key: str) -> Target | None:
    """Parse one safetensors key into a target descriptor."""

    suffix = ".lora_A.weight.qweight"
    if not key.endswith(suffix):
        return None
    target_key = key[: -len(suffix)]
    runtime_key = target_key
    for prefix in ("base_model.model.", "base_model."):
        if runtime_key.startswith(prefix):
            runtime_key = runtime_key[len(prefix) :]
            break
    match = re.search(r"model\.layers\.(\d+)\.(.+)$", runtime_key)
    if not match:
        return None
    layer = int(match.group(1))
    module = match.group(2)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", f"layer{layer:02d}_{module}").strip("_")
    return Target(
        key=target_key,
        runtime_key=runtime_key,
        layer=layer,
        module=module,
        safe_name=safe_name,
    )


def discover_targets(default_adapter: Path, target_regex: str | None, target_limit: int | None) -> list[Target]:
    """Discover all LoRA target modules from the default-rank adapter file."""

    weights = load_file(default_adapter / HF_ADAPTER_FILE_NAME)
    targets = []
    for key in sorted(weights):
        target = target_from_key(key)
        if target is None:
            continue
        if target_regex and not re.search(target_regex, target.runtime_key):
            continue
        targets.append(target)
    targets.sort(key=lambda item: (item.layer, item.module))
    if target_limit is not None:
        targets = targets[:target_limit]
    return targets


def write_targets(work_dir: Path, targets: list[Target]) -> None:
    """Persist the target list so worker processes agree on indexes."""

    payload = [target.__dict__ for target in targets]
    (work_dir / "targets.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_targets(work_dir: Path) -> list[Target]:
    """Read target descriptors written by the coordinator."""

    payload = json.loads((work_dir / "targets.json").read_text(encoding="utf-8"))
    return [Target(**item) for item in payload]


def adapter_file_mb(path: Path) -> float:
    """Return the adapter safetensors file size in MiB."""

    return (path / HF_ADAPTER_FILE_NAME).stat().st_size / 1024**2


def compressed_tensor_keys(target_key: str, side: str) -> tuple[str, str, str]:
    """Build grouped-LoRA tensor keys for a target and LoRA side."""

    base = f"{target_key}.{side}.weight"
    return (f"{base}.qweight", f"{base}.scales", f"{base}.shape")


def load_rank_weights(rank_dirs: dict[int, Path]) -> dict[int, dict[str, Any]]:
    """Load safetensor dictionaries for all ranks used by one worker."""

    return {rank: load_file(path / HF_ADAPTER_FILE_NAME) for rank, path in rank_dirs.items()}


def build_mixed_adapter(
    *,
    rank_dirs: dict[int, Path],
    rank_weights: dict[int, dict[str, Any]],
    target: Target,
    target_rank: int,
    default_rank: int,
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Save one mixed adapter with a lower rank for exactly one target module."""

    if skip_existing and adapter_complete(output_dir):
        return {
            "adapter_dir": str(output_dir),
            "adapter_file_mb": adapter_file_mb(output_dir),
            "adapter_status": "reused",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    default_weights = rank_weights[default_rank]
    target_weights = rank_weights[target_rank]
    mixed = {key: tensor for key, tensor in default_weights.items()}
    replaced = []
    for side in ("lora_A", "lora_B"):
        for key in compressed_tensor_keys(target.key, side):
            if key not in target_weights:
                raise KeyError(f"Missing {key} in rank {target_rank} adapter")
            mixed[key] = target_weights[key]
            replaced.append(key)

    config_payload = json.loads((rank_dirs[default_rank] / HF_ADAPTER_CONFIG_FILE_NAME).read_text(encoding="utf-8"))
    config_payload["r"] = default_rank
    config_payload["lora_alpha"] = default_rank
    config_payload["rank_pattern"] = {target.runtime_key: target_rank}
    LoraConfig(**{k: v for k, v in config_payload.items() if k in LoraConfig.__dataclass_fields__}).save_pretrained(
        str(output_dir)
    )
    save_file(mixed, output_dir / HF_ADAPTER_FILE_NAME, metadata={"format": "pt"})
    return {
        "adapter_dir": str(output_dir),
        "adapter_file_mb": adapter_file_mb(output_dir),
        "adapter_status": "built",
        "replaced_tensor_count": len(replaced),
    }


def extract_baseline(path: Path | None, variant: str) -> dict[str, Any] | None:
    """Extract a prior GSM8K baseline result if one is supplied."""

    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        summary = row.get("summary", {})
        if row.get("variant") == variant and summary.get("task") == "gsm8k_platinum_cot":
            return {
                "variant": variant,
                "score": summary.get("score") or summary.get("acc_num"),
                "correct": summary.get("correct"),
                "sample_count": summary.get("sample_count"),
                "source": str(path),
            }
    for row in payload.get("cases", []):
        summary = (row.get("eval") or {}).get("summary", {})
        if row.get("bits") == 4 and row.get("group_size") == 128 and summary:
            return {
                "variant": variant,
                "score": summary.get("score") or summary.get("acc_num"),
                "correct": summary.get("correct"),
                "sample_count": summary.get("sample_count"),
                "source": str(path),
            }
    return None


def load_case_rows(work_dir: Path) -> list[dict[str, Any]]:
    """Load completed worker JSONL rows."""

    rows: list[dict[str, Any]] = []
    for path in sorted((work_dir / "results").glob("gpu*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A worker may be flushing while the coordinator is merging.
                        continue
    return rows


def completed_case_keys(work_dir: Path) -> set[tuple[str, int]]:
    """Return target/rank pairs already present in worker result files."""

    return {(row["target"], int(row["target_rank"])) for row in load_case_rows(work_dir)}


def table_rows(rows: list[dict[str, Any]], baseline: dict[str, Any] | None) -> list[list[Any]]:
    """Render worker rows into stable table data."""

    baseline_score = baseline.get("score") if baseline else None
    baseline_correct = baseline.get("correct") if baseline else None
    baseline_rows = baseline.get("sample_count") if baseline else None
    table = []
    for row in rows:
        summary = (row.get("eval") or {}).get("summary", {})
        score = summary.get("acc_num") or summary.get("score")
        correct = summary.get("correct")
        sample_count = summary.get("sample_count")
        comparable = baseline_rows is None or sample_count == baseline_rows
        table.append(
            [
                row.get("target_index"),
                row.get("layer"),
                row.get("module"),
                row.get("target_rank"),
                sample_count or "",
                "" if score is None else f"{float(score):.4f}",
                "" if correct is None else correct,
                "" if score is None or baseline_score is None or not comparable else f"{float(score) - float(baseline_score):.4f}",
                "" if correct is None or baseline_correct is None or not comparable else int(correct) - int(baseline_correct),
                "" if (row.get("eval") or {}).get("rows_per_s") is None else f"{float(row['eval']['rows_per_s']):.3f}",
                "" if (row.get("eval") or {}).get("eval_seconds") is None else f"{float(row['eval']['eval_seconds']):.1f}",
                "" if (row.get("eval") or {}).get("peak_allocated_gb") is None else f"{float(row['eval']['peak_allocated_gb']):.2f}",
                "" if row.get("adapter_file_mb") is None else f"{float(row['adapter_file_mb']):.1f}",
                row.get("gpu"),
                row.get("adapter_status", ""),
            ]
        )
    return sorted(table, key=lambda item: (int(item[0]), int(item[3])))


def write_merged_outputs(
    *,
    work_dir: Path,
    context: dict[str, Any],
    baseline: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> str:
    """Write merged JSON/markdown results and return the rendered table."""

    headers = [
        "target",
        "layer",
        "module",
        "rank",
        "rows",
        "acc,num",
        "correct",
        "delta256",
        "delta correct",
        "rows/s",
        "eval s",
        "peak GB",
        "adapter MB",
        "gpu",
        "status",
    ]
    table = tabulate(
        table_rows(rows, baseline),
        headers=headers,
        tablefmt="github",
        stralign="right",
        numalign="right",
        disable_numparse=True,
    )
    payload = {
        "context": context,
        "baseline": baseline,
        "results": sorted(rows, key=lambda row: (row.get("target_index", -1), row.get("target_rank", -1))),
        "table": table,
    }
    (work_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# EoRA mixed-rank target sweep",
        "",
        f"- quantized model: `{context['quantized_model']}`",
        f"- default rank: `{context['default_rank']}`",
        f"- target ranks: `{context['ranks']}`",
        f"- GPUs: `{context['gpus']}`",
        f"- targets: `{context['target_count']}`",
        f"- full rows requested: `{context['max_rows'] is None}`",
    ]
    if baseline:
        md.append(
            f"- rank{context['default_rank']} baseline: `{float(baseline['score']):.4f}` "
            f"({baseline.get('correct')}/{baseline.get('sample_count')}) from `{baseline.get('source')}`"
        )
    md.extend(["", table, ""])
    (work_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    return table


def worker_main(args: argparse.Namespace, ranks: list[int]) -> int:
    """Run the target/rank cases assigned to one GPU worker process."""

    import torch

    from scripts.eora_lora_quantized_adapter_sweep import _evaluate_adapter

    assert args.worker_gpu is not None
    target_indexes = {int(item) for item in args.target_indexes.split(",") if item.strip()}
    targets = read_targets(args.work_dir)
    rank_dirs = {int(rank): path for rank, path in json.loads((args.work_dir / "rank_bank.json").read_text()).items()}
    rank_dirs = {rank: Path(path) for rank, path in rank_dirs.items()}
    rank_weights = load_rank_weights(rank_dirs)
    result_dir = args.work_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"gpu{args.worker_gpu}.jsonl"
    completed = completed_case_keys(args.work_dir) if args.skip_existing else set()

    with result_path.open("a", encoding="utf-8") as result_file:
        for index, target in enumerate(targets):
            if index not in target_indexes:
                continue
            for rank in ranks:
                case_key = (target.runtime_key, rank)
                if case_key in completed:
                    print(f"gpu{args.worker_gpu}: skip target={index} rank={rank}", flush=True)
                    continue
                adapter_dir = args.work_dir / "mixed_adapters" / target.safe_name / f"rank{rank}"
                print(f"gpu{args.worker_gpu}: target={index} {target.runtime_key} rank={rank}", flush=True)
                start = time.perf_counter()
                case = build_mixed_adapter(
                    rank_dirs=rank_dirs,
                    rank_weights=rank_weights,
                    target=target,
                    target_rank=rank,
                    default_rank=args.default_rank,
                    output_dir=adapter_dir,
                    skip_existing=args.skip_existing,
                )
                case.update(
                    {
                        "target_index": index,
                        "target": target.runtime_key,
                        "target_full_key": target.key,
                        "target_rank": rank,
                        "default_rank": args.default_rank,
                        "layer": target.layer,
                        "module": target.module,
                        "gpu": args.worker_gpu,
                        "build_seconds": time.perf_counter() - start,
                    }
                )
                if not args.skip_eval:
                    torch.cuda.empty_cache()
                    case["eval"] = _evaluate_adapter(
                        quantized_model=args.quantized_model,
                        adapter_dir=adapter_dir,
                        rank=args.default_rank,
                        backend=args.backend,
                        eval_batch_size=args.eval_batch_size,
                        max_new_tokens=args.max_new_tokens,
                        max_rows=args.max_rows,
                    )
                case["status"] = "completed"
                result_file.write(json.dumps(case) + "\n")
                result_file.flush()
                (adapter_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return 0


def spawn_workers(args: argparse.Namespace, ranks: list[int], targets: list[Target], gpus: list[int]) -> int:
    """Start one subprocess per GPU and monitor completion."""

    logs_dir = args.work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for shard, gpu in enumerate(gpus):
        indexes = [str(index) for index in range(len(targets)) if index % len(gpus) == shard]
        if not indexes:
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-gpu",
            str(gpu),
            "--target-indexes",
            ",".join(indexes),
            "--quantized-model",
            str(args.quantized_model),
            "--work-dir",
            str(args.work_dir),
            "--ranks",
            ",".join(str(rank) for rank in ranks),
            "--default-rank",
            str(args.default_rank),
            "--backend",
            str(args.backend),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.max_rows is not None:
            cmd.extend(["--max-rows", str(args.max_rows)])
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.skip_eval:
            cmd.append("--skip-eval")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_file = (logs_dir / f"gpu{gpu}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log_file, stderr=subprocess.STDOUT)
        workers.append((gpu, process, log_file))
        print(f"started gpu{gpu}: {len(indexes)} targets, log={logs_dir / f'gpu{gpu}.log'}", flush=True)

    baseline = extract_baseline(args.baseline_json, args.baseline_variant)
    context = {
        "quantized_model": str(args.quantized_model),
        "work_dir": str(args.work_dir),
        "ranks": ranks,
        "default_rank": args.default_rank,
        "gpus": gpus,
        "target_count": len(targets),
        "backend": args.backend,
        "eval_batch_size": args.eval_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_rows": args.max_rows,
        "skip_eval": args.skip_eval,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    total_cases = len(targets) * len(ranks)
    last_completed = -1
    last_print = 0.0
    try:
        while True:
            rows = load_case_rows(args.work_dir)
            completed = len(rows)
            now = time.time()
            if completed != last_completed or now - last_print > 60:
                table = write_merged_outputs(work_dir=args.work_dir, context=context, baseline=baseline, rows=rows)
                print(f"progress: {completed}/{total_cases} cases complete", flush=True)
                if rows:
                    print("\n".join(table.splitlines()[-min(12, len(table.splitlines())) :]), flush=True)
                print(f"merged results: {args.work_dir / 'results.md'}", flush=True)
                last_completed = completed
                last_print = now

            running = [(gpu, proc) for gpu, proc, _ in workers if proc.poll() is None]
            if not running:
                break
            time.sleep(30)
    finally:
        for _, _, log_file in workers:
            log_file.close()

    failed = [(gpu, proc.returncode) for gpu, proc, _ in workers if proc.returncode != 0]
    rows = load_case_rows(args.work_dir)
    write_merged_outputs(work_dir=args.work_dir, context=context, baseline=baseline, rows=rows)
    if failed:
        print(f"failed workers: {failed}", flush=True)
        return 1
    return 0


def main() -> int:
    """Coordinate or execute a mixed-rank target sweep."""

    args = parse_args()
    ranks = parse_csv_ints(args.ranks)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        return worker_main(args, ranks)

    gpus = parse_csv_ints(args.gpus)
    rank_dirs = ensure_adapter_bank(args, ranks)
    (args.work_dir / "rank_bank.json").write_text(
        json.dumps({rank: str(path) for rank, path in rank_dirs.items()}, indent=2),
        encoding="utf-8",
    )
    targets = discover_targets(rank_dirs[args.default_rank], args.target_regex, args.target_limit)
    if not targets:
        raise ValueError("No EoRA targets discovered.")
    write_targets(args.work_dir, targets)
    (args.work_dir / "target_table.md").write_text(
        tabulate(
            [[idx, target.layer, target.module, target.runtime_key] for idx, target in enumerate(targets)],
            headers=["target", "layer", "module", "runtime_key"],
            tablefmt="github",
            disable_numparse=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"targets: {len(targets)} saved to {args.work_dir / 'target_table.md'}", flush=True)
    return spawn_workers(args, ranks, targets, gpus)


if __name__ == "__main__":
    raise SystemExit(main())
