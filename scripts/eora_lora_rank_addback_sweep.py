#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEFAULT_RANK = 256


@dataclass(frozen=True)
class Candidate:
    """Identifies one reduced-rank module that can be restored to the default rank."""

    runtime_key: str
    source_rank: int
    safe_name: str


def parse_csv_ints(value: str) -> list[int]:
    """Parse comma-separated GPU ids while preserving user order."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Build the CLI for greedy EoRA mixed-rank add-back evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mixed-rank EoRA maps where reduced-rank targets are restored to rank256, "
            "one greedy round at a time."
        )
    )
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--rank-map", required=True, type=Path)
    parser.add_argument("--rank-bank-json", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--current-results-json", type=Path, default=None)
    parser.add_argument("--default-rank", type=int, default=DEFAULT_DEFAULT_RANK)
    parser.add_argument("--backend", default="marlin")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--target-correct", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--launch-cooldown-seconds", type=float, default=8.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going-on-error", action="store_true")
    parser.add_argument("--no-stop-at-target", action="store_false", dest="stop_at_target")
    parser.set_defaults(stop_at_target=True)
    return parser.parse_args()


def safe_target_name(runtime_key: str) -> str:
    """Convert a module runtime key into a filesystem-safe case name."""

    return re.sub(r"[^a-zA-Z0-9]+", "_", runtime_key).strip("_")


def load_rank_map(path: Path) -> tuple[dict[str, int], dict[str, Any] | None, dict[str, Any]]:
    """Load a rank-map payload while preserving non-map metadata for derived maps."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rank_map = {str(key): int(value) for key, value in payload["rank_map"].items()}
    return rank_map, payload.get("baseline"), payload


def load_current_summary(path: Path | None) -> dict[str, Any] | None:
    """Load the current combined-map score used as the greedy baseline."""

    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = ((payload.get("eval") or {}).get("summary") or {})
    return {
        "adapter": payload.get("adapter_name"),
        "correct": summary.get("correct"),
        "score": summary.get("acc_num") or summary.get("score"),
        "sample_count": summary.get("sample_count"),
        "source": str(path),
    }


def restored_targets_from_payload(payload: dict[str, Any]) -> list[str]:
    """Return restored-target history from a previously generated rank map."""

    greedy = payload.get("greedy_addback")
    if isinstance(greedy, dict) and isinstance(greedy.get("restored_targets"), list):
        return [str(item) for item in greedy["restored_targets"]]
    addback = payload.get("addback")
    if isinstance(addback, dict) and isinstance(addback.get("restored_targets"), list):
        return [str(item) for item in addback["restored_targets"]]
    return []


def candidate_list(rank_map: dict[str, int], default_rank: int, limit: int | None) -> list[Candidate]:
    """Return reduced-rank targets that can be restored to the default rank."""

    candidates = [
        Candidate(runtime_key=key, source_rank=rank, safe_name=safe_target_name(key))
        for key, rank in sorted(rank_map.items())
        if rank != default_rank
    ]
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def selected_rank_counts(rank_map: dict[str, int]) -> dict[int, int]:
    """Count target ranks for output metadata."""

    return dict(sorted(Counter(rank_map.values()).items()))


def write_case_rank_map(
    *,
    path: Path,
    source_payload: dict[str, Any],
    rank_map: dict[str, int],
    baseline: dict[str, Any] | None,
    round_index: int,
    candidate: Candidate,
    restored_targets: list[str],
) -> None:
    """Write one derived rank map for a single add-back case."""

    payload = dict(source_payload)
    payload["baseline"] = baseline
    payload["rank_map"] = rank_map
    payload["selected_rank_counts"] = selected_rank_counts(rank_map)
    payload["addback"] = {
        "round": round_index,
        "candidate": candidate.runtime_key,
        "candidate_source_rank": candidate.source_rank,
        "restored_targets": restored_targets + [candidate.runtime_key],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def result_summary(result_json: Path, candidate: Candidate, current_correct: int | None, baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Extract score and runtime metrics from one combined-rank evaluation result."""

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    eval_payload = payload.get("eval") or {}
    summary = eval_payload.get("summary") or {}
    correct = summary.get("correct")
    score = summary.get("acc_num") or summary.get("score")
    baseline_correct = (baseline or {}).get("correct")
    baseline_score = (baseline or {}).get("score")
    return {
        "target": candidate.runtime_key,
        "source_rank": candidate.source_rank,
        "rows": summary.get("sample_count"),
        "score": score,
        "correct": correct,
        "delta_current_correct": "" if correct is None or current_correct is None else int(correct) - int(current_correct),
        "delta_rank256_correct": "" if correct is None or baseline_correct is None else int(correct) - int(baseline_correct),
        "delta_rank256_score": "" if score is None or baseline_score is None else float(score) - float(baseline_score),
        "rows_per_s": eval_payload.get("rows_per_s"),
        "eval_seconds": eval_payload.get("eval_seconds"),
        "peak_allocated_gb": eval_payload.get("peak_allocated_gb"),
        "adapter_mb": (payload.get("adapter") or {}).get("adapter_file_mb"),
        "non_default_targets": payload.get("non_default_target_count"),
        "result_json": str(result_json),
    }


def table_for_results(results: list[dict[str, Any]]) -> str:
    """Render completed case results in a fixed-column table."""

    headers = [
        "restored target",
        "from rank",
        "rows",
        "acc,num",
        "correct",
        "delta current",
        "delta rank256",
        "rows/s",
        "eval s",
        "peak GB",
        "adapter MB",
        "non256",
    ]
    rows = []
    for item in sorted(results, key=lambda row: (row.get("correct") or -1, row.get("score") or -1), reverse=True):
        rows.append(
            [
                item["target"],
                item["source_rank"],
                item.get("rows", ""),
                "" if item.get("score") is None else f"{float(item['score']):.6f}",
                "" if item.get("correct") is None else item["correct"],
                item.get("delta_current_correct", ""),
                item.get("delta_rank256_correct", ""),
                "" if item.get("rows_per_s") is None else f"{float(item['rows_per_s']):.3f}",
                "" if item.get("eval_seconds") is None else f"{float(item['eval_seconds']):.1f}",
                "" if item.get("peak_allocated_gb") is None else f"{float(item['peak_allocated_gb']):.2f}",
                "" if item.get("adapter_mb") is None else f"{float(item['adapter_mb']):.1f}",
                item.get("non_default_targets", ""),
            ]
        )
    return tabulate(rows, headers=headers, tablefmt="github", stralign="right", numalign="right")


def write_outputs(work_dir: Path, payload: dict[str, Any]) -> None:
    """Persist live JSON and Markdown summaries after each completed case."""

    results = payload.get("results") or []
    errors = payload.get("errors") or []
    table = table_for_results(results) if results else ""
    payload["table"] = table
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# EoRA mixed-rank add-back sweep",
        "",
        f"- quantized model: `{payload['context']['quantized_model']}`",
        f"- source rank map: `{payload['context']['rank_map']}`",
        f"- gpus: `{payload['context']['gpus']}`",
        f"- baseline correct: `{(payload.get('baseline') or {}).get('correct')}`",
        f"- current correct: `{(payload.get('current') or {}).get('correct')}`",
        "",
    ]
    if table:
        md.extend([table, ""])
    if errors:
        md.extend(["## Errors", ""])
        for error in errors:
            md.append(f"- `{error['target']}` gpu={error.get('gpu')} rc={error.get('returncode')} log=`{error.get('log')}`")
        md.append("")
    (work_dir / "results.md").write_text("\n".join(md), encoding="utf-8")


def launch_case(
    *,
    gpu: int,
    case_dir: Path,
    map_path: Path,
    adapter_name: str,
    args: argparse.Namespace,
    log_path: Path,
) -> subprocess.Popen:
    """Start one combined-rank evaluation subprocess pinned to a single physical GPU."""

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eora_lora_combined_rank_eval.py"),
        "--quantized-model",
        str(args.quantized_model),
        "--rank-map",
        str(map_path),
        "--rank-bank-json",
        str(args.rank_bank_json),
        "--work-dir",
        str(case_dir),
        "--adapter-name",
        adapter_name,
        "--gpu",
        str(gpu),
        "--backend",
        args.backend,
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])
    if args.skip_existing:
        cmd.append("--skip-existing")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT, env=env)


def run_round(
    *,
    round_index: int,
    current_map: dict[str, int],
    source_payload: dict[str, Any],
    baseline: dict[str, Any] | None,
    current_correct: int | None,
    restored_targets: list[str],
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate one greedy round of single-target rank256 add-back cases."""

    candidates = candidate_list(current_map, args.default_rank, args.candidate_limit)
    gpus = parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must include at least one GPU id")
    if not candidates:
        return []

    pending = list(candidates)
    running: dict[subprocess.Popen, dict[str, Any]] = {}
    gpu_ready_at = {gpu: 0.0 for gpu in gpus}
    round_results: list[dict[str, Any]] = []
    print(f"round {round_index}: evaluating {len(candidates)} add-back candidates across GPUs {gpus}", flush=True)

    while pending or running:
        used_gpus = {meta["gpu"] for meta in running.values()}
        now = time.monotonic()
        free_gpus = [gpu for gpu in gpus if gpu not in used_gpus and now >= gpu_ready_at[gpu]]
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            candidate = pending.pop(0)
            case_name = f"round{round_index:02d}_{candidate.safe_name}_from{candidate.source_rank}_to{args.default_rank}"
            case_dir = args.work_dir / "cases" / case_name
            result_json = case_dir / "results.json"
            map_path = args.work_dir / "rank_maps" / f"{case_name}.json"
            log_path = args.work_dir / "logs" / f"{case_name}.log"

            case_map = dict(current_map)
            case_map[candidate.runtime_key] = args.default_rank
            write_case_rank_map(
                path=map_path,
                source_payload=source_payload,
                rank_map=case_map,
                baseline=baseline,
                round_index=round_index,
                candidate=candidate,
                restored_targets=restored_targets,
            )

            if args.skip_existing and result_json.is_file():
                result = result_summary(result_json, candidate, current_correct, baseline)
                round_results.append(result)
                payload["results"].append(result)
                write_outputs(args.work_dir, payload)
                print(f"reused {candidate.runtime_key}: correct={result.get('correct')}", flush=True)
                continue

            process = launch_case(
                gpu=gpu,
                case_dir=case_dir,
                map_path=map_path,
                adapter_name=case_name,
                args=args,
                log_path=log_path,
            )
            running[process] = {
                "candidate": candidate,
                "gpu": gpu,
                "case_dir": case_dir,
                "result_json": result_json,
                "log": str(log_path),
            }
            print(f"launched gpu{gpu}: {candidate.runtime_key} rank{candidate.source_rank}->rank{args.default_rank}", flush=True)

        time.sleep(5)
        for process, meta in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            del running[process]
            candidate = meta["candidate"]
            gpu_ready_at[meta["gpu"]] = time.monotonic() + max(0.0, float(args.launch_cooldown_seconds))
            if returncode == 0 and meta["result_json"].is_file():
                result = result_summary(meta["result_json"], candidate, current_correct, baseline)
                round_results.append(result)
                payload["results"].append(result)
                print(
                    f"finished gpu{meta['gpu']}: {candidate.runtime_key} correct={result.get('correct')} "
                    f"delta_current={result.get('delta_current_correct')}",
                    flush=True,
                )
            else:
                error = {
                    "target": candidate.runtime_key,
                    "gpu": meta["gpu"],
                    "returncode": returncode,
                    "log": meta["log"],
                }
                payload["errors"].append(error)
                print(f"error gpu{meta['gpu']}: {candidate.runtime_key} rc={returncode} log={meta['log']}", flush=True)
                if not args.keep_going_on_error:
                    raise RuntimeError(f"Case failed: {candidate.runtime_key}; see {meta['log']}")
            write_outputs(args.work_dir, payload)

    return round_results


def best_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-correct result from one add-back round."""

    if not results:
        return None
    return max(results, key=lambda item: (item.get("correct") or -1, item.get("score") or -1))


def main() -> int:
    """Run one or more greedy add-back rounds and save live outputs."""

    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rank_map, baseline, source_payload = load_rank_map(args.rank_map)
    current = load_current_summary(args.current_results_json)
    baseline_correct = (baseline or {}).get("correct")
    target_correct = args.target_correct if args.target_correct is not None else baseline_correct
    current_correct = None if current is None else current.get("correct")
    restored_targets = restored_targets_from_payload(source_payload)

    payload: dict[str, Any] = {
        "context": {
            "quantized_model": str(args.quantized_model),
            "rank_map": str(args.rank_map),
            "rank_bank_json": str(args.rank_bank_json),
            "default_rank": args.default_rank,
            "backend": args.backend,
            "eval_batch_size": args.eval_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_rows": args.max_rows,
            "gpus": parse_csv_ints(args.gpus),
            "rounds": args.rounds,
            "target_correct": target_correct,
            "launch_cooldown_seconds": args.launch_cooldown_seconds,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "baseline": baseline,
        "current": current,
        "initial_rank_counts": selected_rank_counts(rank_map),
        "selected": [],
        "results": [],
        "errors": [],
    }
    write_outputs(args.work_dir, payload)

    for round_index in range(1, args.rounds + 1):
        round_results = run_round(
            round_index=round_index,
            current_map=rank_map,
            source_payload=source_payload,
            baseline=baseline,
            current_correct=current_correct,
            restored_targets=restored_targets,
            args=args,
            payload=payload,
        )
        best = best_result(round_results)
        if best is None:
            break

        payload["selected"].append(best)
        best_correct = best.get("correct")
        improved = current_correct is None or (best_correct is not None and int(best_correct) > int(current_correct))
        if not improved:
            print(f"round {round_index}: no improving add-back; stopping", flush=True)
            break

        rank_map[best["target"]] = args.default_rank
        restored_targets.append(best["target"])
        current_correct = int(best_correct)
        payload["current"] = {
            "adapter": f"round{round_index:02d}_selected",
            "correct": best_correct,
            "score": best.get("score"),
            "sample_count": best.get("rows"),
            "source": best.get("result_json"),
        }
        selected_map_payload = dict(source_payload)
        selected_map_payload["baseline"] = baseline
        selected_map_payload["rank_map"] = rank_map
        selected_map_payload["selected_rank_counts"] = selected_rank_counts(rank_map)
        selected_map_payload["greedy_addback"] = {
            "round": round_index,
            "restored_targets": restored_targets,
            "current_correct": current_correct,
            "target_correct": target_correct,
        }
        selected_path = args.work_dir / "rank_maps" / f"round{round_index:02d}_selected_rank_map.json"
        selected_path.write_text(json.dumps(selected_map_payload, indent=2), encoding="utf-8")
        payload["selected"][-1]["selected_rank_map"] = str(selected_path)
        write_outputs(args.work_dir, payload)
        print(f"round {round_index}: selected {best['target']} -> correct={best_correct}", flush=True)

        if args.stop_at_target and target_correct is not None and current_correct >= int(target_correct):
            print(f"target reached: current_correct={current_correct} target_correct={target_correct}", flush=True)
            break

    write_outputs(args.work_dir, payload)
    if payload["table"]:
        print(payload["table"])
    print(f"\nresults: {args.work_dir / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
