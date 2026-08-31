#!/usr/bin/env python3
"""Durable GPU queue for held-out QVQ evaluations.

The monitor is deliberately conservative: it adopts evaluations already visible
in the process table, never schedules an excluded calibration artifact, and
publishes a ledger line only after an evaluator exits successfully and its JSON
report is complete.  ``--once`` is a non-mutating inventory/dry run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:  # Python 3.10 compatibility
    UTC = timezone.utc
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("/root/qvq-results")
REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DEFAULT = REPO_ROOT / "docs/experiments/qvq_eval_monitor_state.json"
LEDGER_DEFAULT = REPO_ROOT / "docs/qvq_llama32_1b_experiment_log_2026-08-25.md"
DENSE_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
D300_DATASET = "/root/qvq-data/divergence300-v1/divergence300-development.jsonl"
MICRO_MATH_DATASET = REPO_ROOT / "dataset/micro_math_llama3.2_1b.jsonl"
MICRO_MATH_MANIFEST = REPO_ROOT / "docs/experiments/micro-math-disjointness.json"


@dataclass
class Job:
    checkpoint: str
    task: str
    output: str
    gpu: int | None = None
    pid: int | None = None
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    attempts: int = 0


def excluded_checkpoint(checkpoint: Path) -> bool:
    """Return true for calibration artifacts disqualified by contamination audit."""
    return "div300-sources-500k" in checkpoint.name


def complete_checkpoint(checkpoint: Path) -> bool:
    return checkpoint.is_dir() and (checkpoint / "qvq_quantize_run.json").is_file()


def discover_checkpoints(root: Path = RESULTS_ROOT) -> list[Path]:
    return [p for p in sorted(root.glob("*")) if complete_checkpoint(p) and not excluded_checkpoint(p)]


def report_path(checkpoint: Path, task: str) -> Path:
    if task == "micro_math":
        return Path(f"{checkpoint}-micro-math-v1.json")
    if task == "gsm8k_platinum_cot" and "full-reference-reg020" in checkpoint.name:
        return Path(f"{checkpoint}-gsm8k-platinum-reverify-v2.json")
    suffix = "gsm8k-platinum-v1" if task == "gsm8k_platinum_cot" else "div300-dev-v1"
    return Path(f"{checkpoint}-{suffix}.json")


def process_table() -> str:
    # Include the executable name so shell queue wrappers that merely contain
    # ``scripts/qvq_evaluate.py`` in their wait command are not mistaken for
    # live evaluator processes.
    return subprocess.run(["ps", "-eo", "pid=,comm=,args="], check=True, capture_output=True, text=True).stdout


def active_jobs() -> set[tuple[str, str]]:
    jobs: set[tuple[str, str]] = set()
    for line in process_table().splitlines():
        match = re.match(r"\s*(\d+)\s+(\S+)\s+(.*)$", line)
        if not match or not match.group(2).startswith("python"):
            continue
        args = match.group(3)
        if "scripts/qvq_evaluate.py" not in args:
            continue
        checkpoint = re.search(r"--checkpoint\s+(\S+)", args)
        if not checkpoint:
            continue
        task = (
            "micro_math" if " micro_math " in args
            else "gsm8k_platinum_cot" if " tasks " in args
            else "divergence300" if " divergence300 " in args
            else None
        )
        if task:
            jobs.add((str(Path(checkpoint.group(1)).resolve()), task))
    return jobs


def gpu_snapshot() -> dict[int, tuple[float, int]]:
    raw = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = {}
    for line in raw.splitlines():
        index, util, memory = (x.strip() for x in line.split(","))
        result[int(index)] = (float(util), int(memory))
    return result


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": 1, "updated_at": None, "jobs": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema", 1)
    payload.setdefault("jobs", {})
    return payload


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def requested_jobs(state: dict[str, Any], checkpoints: list[Path]) -> list[Job]:
    jobs: list[Job] = []
    for checkpoint in checkpoints:
        for task in ("micro_math", "gsm8k_platinum_cot", "divergence300"):
            output = report_path(checkpoint, task)
            key = f"{checkpoint}|{task}"
            existing = state["jobs"].get(key)
            if output.is_file() or (existing and existing.get("status") in {"queued", "running", "complete", "skipped_legacy", "skipped_no_contract"}):
                continue
            if task == "micro_math":
                run_manifest = checkpoint / "qvq_quantize_run.json"
                try:
                    run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                contract = run_payload.get("disjointness_manifest")
                if not isinstance(contract, dict) or not contract.get("strict_required") or contract.get("status") != "pass":
                    state["jobs"][key] = asdict(Job(
                        str(checkpoint), task, str(output), status="skipped_no_contract",
                        error="checkpoint lacks a passing strict disjointness contract",
                    ))
                    continue
            attempts = int(existing.get("attempts", 0)) if existing else 0
            if attempts >= 3:
                continue
            jobs.append(Job(str(checkpoint), task, str(output), attempts=attempts))
    def priority(job: Job) -> tuple[int, int, str]:
        name = Path(job.checkpoint).name
        checkpoint_priority = 0 if "full-reference-reg020" in name else 1 if "-w2-" in name else 2
        task_priority = {"micro_math": 0, "gsm8k_platinum_cot": 1, "divergence300": 2}[job.task]
        return checkpoint_priority, task_priority, name
    return sorted(jobs, key=priority)


def initialize_micro_math_state(state: dict[str, Any], checkpoints: list[Path], *, dry_run: bool) -> None:
    """Avoid flooding the queue with retroactive micro-math jobs for old arms."""

    if state.get("micro_math_migration_v1"):
        return
    if not dry_run:
        for checkpoint in checkpoints:
            key = f"{checkpoint}|micro_math"
            if key not in state["jobs"]:
                state["jobs"][key] = asdict(Job(
                    str(checkpoint), "micro_math", str(report_path(checkpoint, "micro_math")),
                    status="skipped_legacy", error="micro-math instrumentation added after this arm completed",
                ))
        state["micro_math_migration_v1"] = datetime.now(UTC).isoformat()


def promote_current_micro_math(state: dict[str, Any], checkpoints: list[Path], *, dry_run: bool) -> None:
    """Promote the arm family created with the new proxy contract.

    The first migration intentionally marked all pre-existing checkpoints as
    ``skipped_legacy`` so enabling the proxy could not flood the GPU queue.
    The W3-anchor campaign was still the active experiment batch at that point,
    so those checkpoints must be evaluated under the new protocol.  New
    checkpoints are not placed in state by the migration and are queued by
    :func:`requested_jobs` normally.
    """

    if state.get("micro_math_migration_v2"):
        return
    if not dry_run:
        promoted = 0
        for checkpoint in checkpoints:
            if "w3anchor" not in checkpoint.name:
                continue
            key = f"{checkpoint}|micro_math"
            raw = state["jobs"].get(key)
            if isinstance(raw, dict) and raw.get("status") == "skipped_legacy":
                state["jobs"].pop(key, None)
                promoted += 1
        state["micro_math_migration_v2"] = {
            "at": datetime.now(UTC).isoformat(),
            "promoted_w3anchor_jobs": promoted,
        }


def command(job: Job) -> list[str]:
    if job.task == "micro_math":
        return [
            "python", "scripts/qvq_evaluate.py", "micro_math", "--dense-model", DENSE_MODEL,
            "--checkpoint", job.checkpoint, "--dataset", str(MICRO_MATH_DATASET),
            "--manifest", str(MICRO_MATH_MANIFEST), "--rows", "64", "--rollout-tokens", "48",
            "--max-prompt-tokens", "2048", "--device", "cuda:0", "--dtype", "float16",
            "--attn-implementation", "sdpa", "--output", job.output,
        ]
    if job.task == "gsm8k_platinum_cot":
        return [
            "python", "scripts/qvq_evaluate.py", "tasks", "--checkpoint", job.checkpoint,
            "--output", job.output, "--task", job.task, "--batch-size", "64",
            "--device", "cuda:0", "--attn-implementation", "paged|flash_attention_2",
            "--use-cuda-graph",
        ]
    return [
        "python", "scripts/qvq_evaluate.py", "divergence300", "--dense-model", DENSE_MODEL,
        "--checkpoint", job.checkpoint, "--dataset", D300_DATASET, "--device", "cuda:0",
        "--output", job.output, "--max-prompt-tokens", "16384", "--dtype", "float16",
        "--attn-implementation", "sdpa",
    ]


def append_ledger(ledger: Path, job: Job, payload: dict[str, Any]) -> None:
    task_payload = payload.get("gsm8k_platinum_cot", payload.get("tasks", {}).get("gsm8k_platinum_cot", {}))
    metric = task_payload.get("metrics", {}).get("acc,num")
    if job.task == "micro_math":
        metrics = payload.get("metrics", {})
        metric = (
            f"mini_exact={metrics.get('mini_math_exact_answer_accuracy')!r}; "
            f"delta_ce={metrics.get('reasoning_delta_ce')!r}; "
            f"delta_kl={metrics.get('reasoning_delta_kl')!r}; "
            f"answer_logprob_delta={metrics.get('answer_token_logprob_delta')!r}; "
            f"answer_margin_delta={metrics.get('answer_token_margin_delta')!r}; "
            f"critical_top1={metrics.get('critical_token_top1_quantized_vs_dense')!r}"
        )
    elif job.task == "divergence300":
        metric = payload.get("divergence_300_at_32", {}).get("independent_token_top1_agreement_at_32")
    line = (
        f"\n| monitor | `{Path(job.checkpoint).name}` | `{job.task}` | "
        f"complete; metric={metric!r}; report `{job.output}` |\n"
    )
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(line)


def adopt_recent_reports(state: dict[str, Any], ledger: Path, checkpoints: list[Path], dry_run: bool) -> int:
    """Adopt reports published by external wrappers during the current run window."""
    if dry_run:
        return 0
    cutoff = time.time() - 6 * 3600
    adopted = 0
    for checkpoint in checkpoints:
        for task in ("gsm8k_platinum_cot", "divergence300"):
            output = report_path(checkpoint, task)
            key = f"{checkpoint}|{task}"
            if key in state["jobs"] or not output.is_file() or output.stat().st_mtime < cutoff:
                continue
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            job = Job(str(checkpoint), task, str(output), status="complete", finished_at=datetime.now(UTC).isoformat())
            state["jobs"][key] = asdict(job)
            append_ledger(ledger, job, payload)
            adopted += 1
    return adopted


def run_once(state_path: Path, ledger: Path, gpus: list[int], dry_run: bool = False) -> dict[str, Any]:
    state = load_state(state_path)
    checkpoints = discover_checkpoints()
    initialize_micro_math_state(state, checkpoints, dry_run=dry_run)
    promote_current_micro_math(state, checkpoints, dry_run=dry_run)
    active = active_jobs()
    adopted = adopt_recent_reports(state, ledger, checkpoints, dry_run)
    # Adopt the separately-run fresh verification of the historically surprising
    # full-reference arm so it is durable and cannot later be queued again.
    verify_checkpoint = next((p for p in checkpoints if "full-reference-reg020" in p.name), None)
    if verify_checkpoint is not None:
        verify_job = Job(str(verify_checkpoint), "gsm8k_platinum_cot", str(report_path(verify_checkpoint, "gsm8k_platinum_cot")), status="complete")
        verify_key = f"{verify_checkpoint}|gsm8k_platinum_cot"
        if verify_key not in state["jobs"] and Path(verify_job.output).is_file() and not dry_run:
            payload = json.loads(Path(verify_job.output).read_text(encoding="utf-8"))
            verify_job.finished_at = datetime.now(UTC).isoformat()
            state["jobs"][verify_key] = asdict(verify_job)
            append_ledger(ledger, verify_job, payload)
    jobs = requested_jobs(state, checkpoints)
    for key, raw in state["jobs"].items():
        running_key = (str(Path(raw["checkpoint"]).resolve()), raw["task"])
        if raw.get("status") == "running" and running_key not in active and not Path(raw["output"]).is_file():
            raw["status"] = "failed"
            raw["error"] = "process disappeared before report publication"
    snapshots = gpu_snapshot()
    occupied = {raw.get("gpu") for raw in state["jobs"].values() if raw.get("status") == "running"}
    launched: list[Job] = []
    for job in jobs:
        if (str(Path(job.checkpoint).resolve()), job.task) in active:
            continue
        available = [g for g in gpus if g in snapshots and g not in occupied and snapshots[g][0] < 5 and snapshots[g][1] < 2000]
        if not available:
            break
        job.gpu = available[0]
        key = f"{job.checkpoint}|{job.task}"
        if dry_run:
            launched.append(job)
            occupied.add(job.gpu)
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        proc = subprocess.Popen(command(job), cwd=REPO_ROOT, env=env)
        job.pid = proc.pid
        job.attempts += 1
        job.status = "running"
        job.started_at = datetime.now(UTC).isoformat()
        state["jobs"][key] = asdict(job)
        occupied.add(job.gpu)
        launched.append(job)
    if not dry_run:
        for key, raw in list(state["jobs"].items()):
            if raw.get("status") != "running" or not Path(raw["output"]).is_file():
                continue
            payload = json.loads(Path(raw["output"]).read_text(encoding="utf-8"))
            raw["status"] = "complete"
            raw["finished_at"] = datetime.now(UTC).isoformat()
            append_ledger(ledger, Job(**{k: raw[k] for k in Job.__dataclass_fields__}), payload)
        write_state(state_path, state)
    return {"checkpoints": len(checkpoints), "pending": len(jobs), "adopted": adopted, "launched": [asdict(j) for j in launched], "active_jobs": len(active)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE_DEFAULT)
    parser.add_argument("--ledger", type=Path, default=LEDGER_DEFAULT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    lock_path = args.state.with_suffix(args.state.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another qvq_eval_monitor instance owns the state lock")
    while True:
        summary = run_once(args.state, args.ledger, gpus, dry_run=args.dry_run)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
