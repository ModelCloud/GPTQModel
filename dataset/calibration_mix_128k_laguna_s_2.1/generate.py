"""Reproduce the 128K-token calibration mix for poolside/Laguna-S-2.1.

This script:
1. Downloads/loads `lemon07r/bartowski-imatrix-v5-semantic`,
   `neuralmagic/calibration` (LLM config), plus domain-diverse candidates:
   code (`ise-uiuc/Magicoder-OSS-Instruct-75K`), math
   (`nvidia/OpenMathInstruct-2`), multilingual (`wikimedia/wikipedia`
   zh/ru) and long-form books (`emozilla/pg19`).
2. Tokenizes the rows with the Laguna-S-2.1 tokenizer (chat template applied
   to every row) and builds ~32K-token candidate shards.
3. Builds a held-out reference from the rows immediately following each
   candidate block.
4. Runs `optimize/calibration_coverage.py` with `--target-tokens 131072`
   and `--target-tokens-mode gain_per_token`.
5. Assembles the greedy-selected best-score mix into this folder as
   `calibration.parquet` plus `dataset_info.json` and a `report.md` copy.

The model is a 219GB bf16 MoE checkpoint; the scanner shards it across the
GPUs given in ``LAGUNA_PHYSICAL_GPUS`` (comma-separated physical nvidia-smi
indices, default ``0,1,2,3``) via ``device_map=auto``. Set it to a single index
for one GPU, or unset ``LAGUNA_PHYSICAL_GPUS`` and pass ``--cpu`` for the CPU
fallback.

Run with the free-threaded GPT-QModel venv and `PYTHON_GIL=0` for
multi-threaded search, e.g.:

    PYTHON_GIL=0 python generate.py
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tokenicer import Tokenicer

MODEL = os.environ.get("LAGUNA_MODEL_PATH", "/monster/data/model/Laguna-S-2.1-PER-LAYER")
DTYPE = "bfloat16"
CONCAT_SIZE = 1024
SHARD_TOKENS = 32_768
REF_TOKENS = 8_192

IMAX_ROWS = 250
NM_ROWS = 500

IMATRIX_DS = "lemon07r/bartowski-imatrix-v5-semantic"
NM_DS = "neuralmagic/calibration"
NM_CONFIG = "LLM"

CODE_DS = "ise-uiuc/Magicoder-OSS-Instruct-75K"
MATH_DS = "nvidia/OpenMathInstruct-2"
WIKI_DS = "wikimedia/wikipedia"
PG19_DS = "emozilla/pg19"
TULU_DS = "allenai/tulu-3-sft-mixture"
FINEWEB_DS = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"

# Cap raw text length before tokenization so single rows stay around 1-2K tokens.
MAX_TEXT_CHARS = 6000

# Tokenization threads (fast tokenizers are thread-safe; results are ordered by
# index so no shared mutable state needs locking).
TOKENIZE_THREADS = 32


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


def count_rows_parallel(
    tok,
    rows: list[list[dict[str, str]]],
    threads: int = TOKENIZE_THREADS,
) -> list[tuple[list[dict[str, str]], int]]:
    """Tokenize rows in parallel; results stay index-ordered so no locks are needed."""

    with ThreadPoolExecutor(max_workers=threads) as pool:
        counts = list(pool.map(lambda m: count_messages(tok, m), rows))
    return list(zip(rows, counts))


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


def collect_budget(
    texts,
    tok,
    budget_tokens: int,
    max_chars: int = MAX_TEXT_CHARS,
) -> list[tuple[list[dict[str, str]], int]]:
    """Collect rows from an iterable of texts/messages until a token budget is met."""

    rows: list[tuple[list[dict[str, str]], int]] = []
    total = 0
    it = iter(texts)
    while total < budget_tokens:
        batch: list[list[dict[str, str]]] = []
        for sample in itertools.islice(it, 256):
            if isinstance(sample, str):
                sample = sample[:max_chars]
                if not sample.strip():
                    continue
            batch.append(to_messages(sample))
        if not batch:
            break
        for messages, n in count_rows_parallel(tok, batch):
            rows.append((messages, n))
            total += n
            if total >= budget_tokens:
                break
    return rows


def iter_code():
    ds = load_dataset(CODE_DS, split="train", streaming=True)
    for row in ds:
        yield [
            {"role": "user", "content": str(row["problem"])[:MAX_TEXT_CHARS]},
            {"role": "assistant", "content": str(row["solution"])[:MAX_TEXT_CHARS]},
        ]


def iter_math():
    ds = load_dataset(MATH_DS, split="train", streaming=True)
    for row in ds:
        yield [
            {"role": "user", "content": str(row["problem"])[:MAX_TEXT_CHARS]},
            {"role": "assistant", "content": str(row["generated_solution"])[:MAX_TEXT_CHARS]},
        ]


def iter_wiki(config: str):
    ds = load_dataset(WIKI_DS, config, split="train", streaming=True)
    for row in ds:
        yield row["text"]


def iter_pg19():
    ds = load_dataset(PG19_DS, split="train", streaming=True)
    for row in ds:
        yield row["text"]


def iter_tulu():
    ds = load_dataset(TULU_DS, split="train", streaming=True)
    for row in ds:
        yield [
            {"role": m["role"], "content": str(m["content"])[:MAX_TEXT_CHARS]}
            for m in row["messages"]
        ]


def iter_fineweb():
    ds = load_dataset(FINEWEB_DS, FINEWEB_CONFIG, split="train", streaming=True)
    for row in ds:
        yield row["text"]


# name -> (iterator factory, number of ~32K-token candidate shards)
EXTRA_SOURCES = {
    "code": (iter_code, 2),
    "math": (iter_math, 2),
    "wiki_zh": (lambda: iter_wiki("20231101.zh"), 2),
    "wiki_ru": (lambda: iter_wiki("20231101.ru"), 1),
    "wiki_ja": (lambda: iter_wiki("20231101.ja"), 1),
    "wiki_ar": (lambda: iter_wiki("20231101.ar"), 1),
    "wiki_de": (lambda: iter_wiki("20231101.de"), 1),
    "wiki_ko": (lambda: iter_wiki("20231101.ko"), 1),
    "pg19": (iter_pg19, 2),
    "tulu": (iter_tulu, 2),
    "fineweb_edu": (iter_fineweb, 2),
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    shards_dir = out_dir / "_shards"
    scanner_dir = out_dir / "_scanner"
    shutil.rmtree(shards_dir, ignore_errors=True)
    shards_dir.mkdir(exist_ok=True)
    scanner_dir.mkdir(exist_ok=True)

    tok = Tokenicer.load(MODEL, trust_remote_code=True).tokenizer

    imatrix = load_dataset(IMATRIX_DS, split="train")
    nm_llm = load_dataset(NM_DS, NM_CONFIG, split="train")

    imatrix_rows = group_messages(
        count_rows_parallel(tok, [to_messages(imatrix[i]["text"]) for i in range(IMAX_ROWS)]),
        CONCAT_SIZE,
    )
    nm_rows = group_messages(
        count_rows_parallel(tok, [to_messages(nm_llm[i]["messages"]) for i in range(NM_ROWS)]),
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

    extra_refs: list[list[dict[str, str]]] = []
    for name, (factory, num_shards) in EXTRA_SOURCES.items():
        print(f"[prep] collecting `{name}` ...")
        budget = int(num_shards * SHARD_TOKENS * 1.15) + 4096
        collected = collect_budget(factory(), tok, budget)
        grouped = group_messages(collected, CONCAT_SIZE)
        _, src_ref = build_shards(
            grouped,
            name,
            num_shards=num_shards,
            shard_tokens=SHARD_TOKENS,
            ref_tokens=2048,
            out_dir=shards_dir,
        )
        extra_refs.extend(src_ref)

    combined_ref = imatrix_ref + nm_ref + extra_refs
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
        "--trust-remote-code",
        "--greedy-threads",
        "32",
        "--target-tokens",
        str(SHARD_TOKENS * 4),  # 131072
        "--target-tokens-mode",
        "gain_per_token",
        "--max-samples",
        "0",
    ]
    physical_gpus = os.environ.get("LAGUNA_PHYSICAL_GPUS", "0,1,2,3")
    if physical_gpus and "--cpu" not in sys.argv:
        cmd.extend(["--physical-gpu", physical_gpus])

    moe_coverage = os.environ.get("LAGUNA_MOE_COVERAGE", "1") == "1"
    moe_bypass = os.environ.get("LAGUNA_MOE_ROUTING_BYPASS", "0") == "1"
    if moe_coverage:
        cmd.append("--moe-expert-coverage")
    if moe_bypass:
        cmd.append("--moe-routing-bypass")
    elif moe_coverage:
        # With bypass the MoE floor equals --target-tokens, so only pass it
        # when real top-k routing is active.
        moe_floor = os.environ.get("LAGUNA_TARGET_MOE_EXPERT_TOKENS", "")
        if moe_floor:
            cmd.extend(["--target-moe-expert-tokens", moe_floor])
    if os.environ.get("LAGUNA_DEFUSE_EXPERTS", "0") == "1":
        cmd.append("--defuse-experts")

    env = os.environ.copy()
    env["PYTHON_GIL"] = "0"
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, env=env, cwd=str(out_dir), check=True)

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
        "id": "calibration_mix_128k_laguna_s_2.1",
        "name": "128K-token calibration mix for poolside/Laguna-S-2.1 (best-score floor)",
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
            {"dataset": CODE_DS, "shards": [n for n in order if n.startswith("code")]},
            {"dataset": MATH_DS, "shards": [n for n in order if n.startswith("math")]},
            {"dataset": WIKI_DS, "shards": [n for n in order if n.startswith("wiki_")]},
            {"dataset": PG19_DS, "shards": [n for n in order if n.startswith("pg19")]},
            {"dataset": TULU_DS, "shards": [n for n in order if n.startswith("tulu")]},
            {"dataset": FINEWEB_DS, "config": FINEWEB_CONFIG, "shards": [n for n in order if n.startswith("fineweb")]},
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
