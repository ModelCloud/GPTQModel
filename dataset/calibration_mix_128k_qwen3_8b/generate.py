"""Reproduce the 128K-token calibration mix for Qwen/Qwen3-8B.

This script:
1. Downloads/loads `lemon07r/bartowski-imatrix-v5-semantic` and
   `neuralmagic/calibration` (LLM config).
2. Tokenizes the rows with the Qwen3-8B tokenizer and builds ~32K-token
   candidate shards.
3. Builds a held-out reference from the rows immediately following each
   candidate block.
4. Runs `optimize/calibration_coverage.py` with `--target-tokens 131072`
   and `--target-tokens-mode gain_per_token`.
5. Assembles the greedy-selected best-score mix into this folder as
   `calibration.parquet` plus `dataset_info.json` and a `report.md` copy.

Run with the free-threaded GPT-QModel venv and `PYTHON_GIL=0` for
multi-threaded search, e.g.:

    PYTHON_GIL=0 /home/ubuntu/.venv-gptq-gil0/bin/python generate.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
DTYPE = "float16"  # AVX-512 FP16 gives faster CPU inference than bfloat16 here
CONCAT_SIZE = 1024
SHARD_TOKENS = 32_768
REF_TOKENS = 8_192

IMAX_ROWS = 250
NM_ROWS = 500

IMATRIX_DS = "lemon07r/bartowski-imatrix-v5-semantic"
NM_DS = "neuralmagic/calibration"
NM_CONFIG = "LLM"


def to_messages(sample: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(sample, list):
        return sample
    return [{"role": "user", "content": str(sample)}]


def count_messages(tok, messages: list[dict[str, str]]) -> int:
    enc = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    return len(enc["input_ids"])


def group_messages(
    rows: list[tuple[list[dict[str, str]], int]],
    concat_size: int,
) -> list[tuple[list[dict[str, str]], int]]:
    """Group short calibration rows so each returned row is close to ``concat_size`` tokens.

    This reduces the number of forward chunks the scanner has to run for short
    conversation-style rows, without changing the total token budget.
    """

    grouped: list[tuple[list[dict[str, str]], int]] = []
    batch: list[dict[str, str]] = []
    batch_tokens = 0
    for messages, n in rows:
        if batch and batch_tokens + n > concat_size:
            grouped.append((batch, batch_tokens))
            batch = list(messages)
            batch_tokens = n
        else:
            batch.extend(messages)
            batch_tokens += n
    if batch:
        grouped.append((batch, batch_tokens))
    return grouped


def build_shards(
    collected_rows: list[tuple[list[dict[str, str]], int]],
    name: str,
    num_shards: int,
    shard_tokens: int,
    ref_tokens: int,
    out_dir: Path,
) -> tuple[list[Path], list[list[dict[str, str]]]]:
    """Build `num_shards` candidate shards plus a held-out reference chunk."""

    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    current: list[list[dict[str, str]]] = []
    current_tokens = 0
    idx = 0

    while idx < len(collected_rows) and len(shards) < num_shards:
        messages, n = collected_rows[idx]
        if (
            current
            and current_tokens + n > shard_tokens
            and current_tokens >= shard_tokens * 0.8
        ):
            path = out_dir / f"{name}_{len(shards):02d}.parquet"
            pd.DataFrame({"messages": current}).to_parquet(path, index=False)
            print(f"[prep] {path.name}: {len(current)} rows, ~{current_tokens} tokens")
            shards.append(path)
            current = []
            current_tokens = 0
        else:
            current.append(messages)
            current_tokens += n
            idx += 1

    ref_rows: list[list[dict[str, str]]] = []
    ref_tokens_count = 0
    if current:
        ref_rows.extend(current)
        ref_tokens_count += current_tokens

    while idx < len(collected_rows):
        messages, n = collected_rows[idx]
        if ref_tokens_count + n > ref_tokens and ref_tokens_count >= ref_tokens * 0.5:
            break
        ref_rows.append(messages)
        ref_tokens_count += n
        idx += 1

    print(f"[prep] {name} reference: {len(ref_rows)} rows, ~{ref_tokens_count} tokens")
    return shards, ref_rows


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    shards_dir = out_dir / "_shards"
    scanner_dir = out_dir / "_scanner"
    shards_dir.mkdir(exist_ok=True)
    scanner_dir.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)

    imatrix = load_dataset(IMATRIX_DS, split="train")
    nm_llm = load_dataset(NM_DS, NM_CONFIG, split="train")

    imatrix_rows = group_messages(
        [
            (to_messages(imatrix[i]["text"]), count_messages(tok, to_messages(imatrix[i]["text"])))
            for i in range(IMAX_ROWS)
        ],
        CONCAT_SIZE,
    )
    nm_rows = group_messages(
        [
            (to_messages(nm_llm[i]["messages"]), count_messages(tok, to_messages(nm_llm[i]["messages"])))
            for i in range(NM_ROWS)
        ],
        CONCAT_SIZE,
    )

    _, imatrix_ref = build_shards(
        imatrix_rows,
        "imatrix",
        num_shards=2,
        shard_tokens=SHARD_TOKENS,
        ref_tokens=REF_TOKENS,
        out_dir=shards_dir,
    )

    _, nm_ref = build_shards(
        nm_rows,
        "nm_llm",
        num_shards=4,
        shard_tokens=SHARD_TOKENS,
        ref_tokens=REF_TOKENS,
        out_dir=shards_dir,
    )

    combined_ref = imatrix_ref + nm_ref
    ref_path = shards_dir / "reference.parquet"
    pd.DataFrame({"messages": combined_ref}).to_parquet(ref_path, index=False)
    print(f"[prep] combined reference -> {ref_path} ({len(combined_ref)} rows)")

    all_shards = sorted(shards_dir.glob("*.parquet"))
    all_shards.remove(ref_path)
    dataset_args = []
    for p in all_shards:
        rel = p.relative_to(out_dir)
        dataset_args.extend(["--dataset", f"{rel}:{p.stem}"])

    scanner_script = repo_root / "optimize" / "calibration_coverage.py"
    ref_rel = ref_path.relative_to(out_dir)
    cmd = [
        sys.executable,
        str(scanner_script),
        "--model",
        MODEL,
        *dataset_args,
        "--reference",
        f"{ref_rel}:reference",
        "--output-dir",
        str(scanner_dir),
        "--concat-size",
        str(CONCAT_SIZE),
        "--sketch-samples",
        "128",
        "--torch-dtype",
        DTYPE,
        "--greedy-threads",
        "4",
        "--target-tokens",
        str(SHARD_TOKENS * 4),  # 131072
        "--target-tokens-mode",
        "gain_per_token",
        "--max-samples",
        "0",
    ]

    env = os.environ.copy()
    env["PYTHON_GIL"] = "0"
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    report_path = scanner_dir / "coverage_report.json"
    report = json.loads(report_path.read_text())

    order = [row["dataset"] for row in report["greedy_ranking"]]
    print("[mix] greedy order:", " -> ".join(order))

    rows: list[pd.DataFrame] = []
    for name in order:
        p = shards_dir / f"{name}.parquet"
        df = pd.read_parquet(p)
        df["shard"] = name
        df["selection_order"] = order.index(name) + 1
        rows.append(df[["messages", "shard", "selection_order"]])
    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(out_dir / "calibration.parquet", index=False)

    info = {
        "id": "calibration_mix_128k_qwen3_8b",
        "name": "128K-token calibration mix for Qwen3-8B (best-score floor)",
        "model": report["config"]["model"],
        "target_tokens_floor": report["config"]["target_tokens"],
        "selection_mode": report["config"]["target_tokens_mode"],
        "total_examples": len(combined),
        "total_tokens": report["selected_mix"]["total_tokens"],
        "final_score": report["selected_mix"]["score"],
        "score_start": report["selected_mix"]["score_start"],
        "cumulative_gain": report["selected_mix"]["cumulative_gain"],
        "sources": [
            {"dataset": IMATRIX_DS, "shards": [n for n in order if n.startswith("imatrix")]},
            {"dataset": NM_DS, "config": NM_CONFIG, "shards": [n for n in order if n.startswith("nm_llm")]},
        ],
        "greedy_ranking": report["greedy_ranking"],
        "reference": report["reference"],
        "timings_seconds": report["timings"],
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")

    shutil.copy(scanner_dir / "coverage_report.json", out_dir / "report.json")
    shutil.copy(scanner_dir / "coverage_report.md", out_dir / "report.md")

    print(f"[done] wrote {out_dir / 'calibration.parquet'} ({len(combined)} rows, {info['total_tokens']} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
