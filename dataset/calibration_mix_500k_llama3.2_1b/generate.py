"""Build a leakage-safe 500K-token activation-coverage mix for Llama 3.2 1B Instruct."""

from __future__ import annotations

import hashlib
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

MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
NM_LOCAL = "/monster/data/model/dataset/nm-calibration/llm.parquet"
IMATRIX_DS = "lemon07r/bartowski-imatrix-v5-semantic"
CODE_DS = "ise-uiuc/Magicoder-OSS-Instruct-75K"
MATH_DS = "nvidia/OpenMathInstruct-2"
WIKI_DS = "wikimedia/wikipedia"
PG19_DS = "emozilla/pg19"
TULU_DS = "allenai/tulu-3-sft-mixture"
FINEWEB_DS = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"

CONCAT_SIZE = 2048
SHARD_TOKENS = 65_536
TARGET_TOKENS = 500_000
MIN_TARGET_TOKENS = 256_000
REFERENCE_TOKENS_PER_SOURCE = 4096
MAX_TEXT_CHARS = 8000
TOKENIZE_THREADS = 32

BENCHMARK_ROWS = range(128, 428)
YAQA_ROWS = range(512, 640)


def assemble_existing(repo_root: Path, out_dir: Path) -> int:
    """Assemble the saved scan, extending positive gain to the requested floor."""
    assembly_floor = int(
        os.environ.get("LLAMA_CALIBRATION_ASSEMBLE_FLOOR", MIN_TARGET_TOKENS)
    )
    if assembly_floor < MIN_TARGET_TOKENS or assembly_floor > TARGET_TOKENS:
        raise ValueError(
            "LLAMA_CALIBRATION_ASSEMBLE_FLOOR must be between "
            f"{MIN_TARGET_TOKENS} and {TARGET_TOKENS}"
        )
    scanner_dir = out_dir / "_scanner"
    preparation = json.loads((out_dir / "preparation.json").read_text())
    report = json.loads((scanner_dir / "coverage_report.json").read_text())
    records = {row["name"]: row for row in preparation["shards"]}

    order = [row["dataset"] for row in report["greedy_ranking"]]
    complementarity = {
        row["name"]: row["conditional_gain"] for row in report["complementarity"]
    }
    effective_tokens = {
        row["name"]: row["total_tokens"] for row in report["per_dataset"]
    }
    # Retain the positive-gain prefix and add the least-redundant remaining
    # shards only until the explicit minimum is satisfied. The desired target
    # remains soft.
    remaining = sorted(
        (name for name in records if name not in order),
        key=lambda name: complementarity[name],
        reverse=True,
    )
    selected_effective_tokens = sum(effective_tokens[name] for name in order)
    for name in remaining:
        if selected_effective_tokens >= assembly_floor:
            break
        order.append(name)
        selected_effective_tokens += effective_tokens[name]
    selected_tokens = sum(records[name]["tokens"] for name in order)

    frames = []
    selected_hashes = set()
    for selection_order, name in enumerate(order, 1):
        frame = pd.read_parquet(records[name]["path"])
        frame["shard"] = name
        frame["selection_order"] = selection_order
        frames.append(frame[["messages", "shard", "selection_order"]])
        selected_hashes.update(
            content_hash(to_messages(row)) for row in frame["messages"]
        )
    mix = pd.concat(frames, ignore_index=True)
    mix_path = out_dir / "calibration.parquet"
    mix.to_parquet(mix_path, index=False)

    nm = load_dataset("parquet", data_files=NM_LOCAL, split="train")
    forbidden = {
        content_hash(to_messages(nm[i]["messages"]))
        for i in itertools.chain(BENCHMARK_ROWS, YAQA_ROWS)
    }
    reference_frame = pd.read_parquet(preparation["reference"])
    reference_hashes = {
        content_hash(to_messages(row)) for row in reference_frame["messages"]
    }
    evidence = {
        "model": MODEL,
        "target_tokens_floor": TARGET_TOKENS,
        "min_target_tokens": MIN_TARGET_TOKENS,
        "assembly_floor": assembly_floor,
        "selected_tokens": selected_tokens,
        "selected_effective_tokens": selected_effective_tokens,
        "selected_rows": len(mix),
        "selected_order": order,
        "selection_policy": (
            "positive conditional-gain prefix, then least-redundant remaining "
            "shards by conditional gain until the minimum token floor"
        ),
        "benchmark": preparation["benchmark"],
        "yaqa": preparation["yaqa"],
        "coverage_reference_rows": preparation["reference_rows"],
        "content_hash_intersections": {
            "selected_vs_benchmark_or_yaqa": len(selected_hashes & forbidden),
            "selected_vs_reference": len(selected_hashes & reference_hashes),
            "reference_vs_benchmark_or_yaqa": len(reference_hashes & forbidden),
        },
        "scanner_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "sources": preparation["shards"],
        "report": report,
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy(scanner_dir / "coverage_report.json", out_dir / "report.json")
    shutil.copy(scanner_dir / "coverage_report.md", out_dir / "report.md")
    print(f"[done] {mix_path}: rows={len(mix)} tokens={selected_tokens}", flush=True)
    return 0


def to_messages(sample) -> list[dict[str, str]]:
    if not isinstance(sample, (str, bytes)) and hasattr(sample, "__iter__"):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in sample]
    return [{"role": "user", "content": str(sample)}]


def content_hash(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def count_messages(tokenizer, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True
    )
    return len(encoded["input_ids"])


def count_rows(tokenizer, rows: list[list[dict[str, str]]]):
    with ThreadPoolExecutor(max_workers=TOKENIZE_THREADS) as pool:
        counts = list(pool.map(lambda row: count_messages(tokenizer, row), rows))
    return list(zip(rows, counts))


def group_rows(rows):
    grouped = []
    current = []
    tokens = 0
    for messages, count in rows:
        if current and tokens + count > CONCAT_SIZE:
            grouped.append((current, tokens))
            current = list(messages)
            tokens = count
        else:
            current.extend(messages)
            tokens += count
    if current:
        grouped.append((current, tokens))
    return grouped


def collect_budget(factory, tokenizer, budget_tokens: int, forbidden: set[str]):
    rows = []
    total = 0
    iterator = iter(factory())
    while total < budget_tokens:
        batch = []
        for sample in itertools.islice(iterator, 256):
            messages = to_messages(sample)
            if content_hash(messages) in forbidden:
                continue
            batch.append(messages)
        if not batch:
            break
        for messages, count in count_rows(tokenizer, batch):
            rows.append((messages, count))
            total += count
            if total >= budget_tokens:
                break
    return rows


def write_shards(rows, name: str, count: int, out_dir: Path):
    shards = []
    index = 0
    for shard_index in range(count):
        selected = []
        tokens = 0
        while index < len(rows):
            messages, row_tokens = rows[index]
            if (
                selected
                and tokens + row_tokens > SHARD_TOKENS
                and tokens >= SHARD_TOKENS * 0.8
            ):
                break
            selected.append(messages)
            tokens += row_tokens
            index += 1
        if not selected:
            break
        path = out_dir / f"{name}_{shard_index:02d}.parquet"
        pd.DataFrame({"messages": selected}).to_parquet(path, index=False)
        shards.append(
            {"path": path, "name": path.stem, "tokens": tokens, "rows": len(selected)}
        )
        print(f"[prep] {path.name}: rows={len(selected)} tokens={tokens}", flush=True)

    reference = []
    reference_tokens = 0
    while index < len(rows) and reference_tokens < REFERENCE_TOKENS_PER_SOURCE:
        messages, row_tokens = rows[index]
        reference.append(messages)
        reference_tokens += row_tokens
        index += 1
    print(
        f"[prep] {name} reference: rows={len(reference)} tokens={reference_tokens}",
        flush=True,
    )
    return shards, reference


def iter_imatrix():
    for row in load_dataset(IMATRIX_DS, split="train", streaming=True):
        yield str(row["text"])[:MAX_TEXT_CHARS]


def iter_code():
    for row in load_dataset(CODE_DS, split="train", streaming=True):
        yield [
            {"role": "user", "content": str(row["problem"])[:MAX_TEXT_CHARS]},
            {"role": "assistant", "content": str(row["solution"])[:MAX_TEXT_CHARS]},
        ]


def iter_math():
    for row in load_dataset(MATH_DS, split="train", streaming=True):
        yield [
            {"role": "user", "content": str(row["problem"])[:MAX_TEXT_CHARS]},
            {
                "role": "assistant",
                "content": str(row["generated_solution"])[:MAX_TEXT_CHARS],
            },
        ]


def iter_wiki(config: str):
    for row in load_dataset(WIKI_DS, config, split="train", streaming=True):
        yield str(row["text"])[:MAX_TEXT_CHARS]


def iter_pg19():
    for row in load_dataset(PG19_DS, split="train", streaming=True):
        yield str(row["text"])[:MAX_TEXT_CHARS]


def iter_tulu():
    for row in load_dataset(TULU_DS, split="train", streaming=True):
        yield [
            {"role": str(m["role"]), "content": str(m["content"])[:MAX_TEXT_CHARS]}
            for m in row["messages"]
        ]


def iter_fineweb():
    for row in load_dataset(FINEWEB_DS, FINEWEB_CONFIG, split="train", streaming=True):
        yield str(row["text"])[:MAX_TEXT_CHARS]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    if os.environ.get("LLAMA_CALIBRATION_ASSEMBLE_ONLY") == "1":
        return assemble_existing(repo_root, out_dir)
    shards_dir = out_dir / "_shards"
    scanner_dir = out_dir / "_scanner"
    shutil.rmtree(shards_dir, ignore_errors=True)
    shutil.rmtree(scanner_dir, ignore_errors=True)
    shards_dir.mkdir(parents=True)
    scanner_dir.mkdir(parents=True)

    tokenizer = Tokenicer.load(MODEL, trust_remote_code=False).tokenizer
    nm = load_dataset("parquet", data_files=NM_LOCAL, split="train")
    benchmark_messages = [to_messages(nm[i]["messages"]) for i in BENCHMARK_ROWS]
    yaqa_messages = [to_messages(nm[i]["messages"]) for i in YAQA_ROWS]
    forbidden = {content_hash(row) for row in benchmark_messages + yaqa_messages}

    # NeuralMagic candidates deliberately begin after the locked benchmark and YAQA ranges.
    nm_candidate_indices = list(range(640, len(nm)))
    nm_rows = []
    for index in nm_candidate_indices:
        messages = to_messages(nm[index]["messages"])
        if content_hash(messages) not in forbidden:
            nm_rows.append(messages)
    nm_counted = group_rows(count_rows(tokenizer, nm_rows))

    sources = [
        ("nm_llm", lambda: (), 3, nm_counted),
        ("imatrix", iter_imatrix, 2, None),
        ("code", iter_code, 1, None),
        ("math", iter_math, 1, None),
        ("wiki_zh", lambda: iter_wiki("20231101.zh"), 1, None),
        ("wiki_ja", lambda: iter_wiki("20231101.ja"), 1, None),
        ("wiki_de", lambda: iter_wiki("20231101.de"), 1, None),
        ("pg19", iter_pg19, 1, None),
        ("tulu", iter_tulu, 1, None),
        ("fineweb_edu", iter_fineweb, 1, None),
    ]

    shard_records = []
    reference_rows = []
    for name, factory, shard_count, prepared in sources:
        print(f"[prep] collecting {name}", flush=True)
        if prepared is None:
            budget = (
                int(shard_count * SHARD_TOKENS * 1.15) + REFERENCE_TOKENS_PER_SOURCE
            )
            prepared = group_rows(collect_budget(factory, tokenizer, budget, forbidden))
        shards, reference = write_shards(prepared, name, shard_count, shards_dir)
        shard_records.extend(shards)
        reference_rows.extend(reference)

    # Add a local-domain reference that is strictly between benchmark and YAQA ranges.
    local_reference = [to_messages(nm[i]["messages"]) for i in range(428, 512)]
    reference_rows.extend(
        row for row in local_reference if content_hash(row) not in forbidden
    )
    reference_hashes = {content_hash(row) for row in reference_rows}
    if reference_hashes & forbidden:
        raise RuntimeError(
            "Reference content overlaps locked benchmark or YAQA content"
        )

    candidate_hashes = set()
    for record in shard_records:
        frame = pd.read_parquet(record["path"])
        for messages in frame["messages"]:
            candidate_hashes.add(content_hash(to_messages(messages)))
    if candidate_hashes & forbidden:
        raise RuntimeError(
            "Candidate content overlaps locked benchmark or YAQA content"
        )
    if candidate_hashes & reference_hashes:
        raise RuntimeError("Candidate content overlaps coverage-reference content")

    reference_path = shards_dir / "reference.parquet"
    pd.DataFrame({"messages": reference_rows}).to_parquet(reference_path, index=False)

    preparation = {
        "model": MODEL,
        "target_tokens_floor": TARGET_TOKENS,
        "benchmark": {"source": NM_LOCAL, "row_start": 128, "rows": 300},
        "yaqa": {"source": NM_LOCAL, "row_start": 512, "rows": 128},
        "reference": str(reference_path),
        "reference_rows": len(reference_rows),
        "content_hash_intersections": {
            "candidate_vs_benchmark_or_yaqa": len(candidate_hashes & forbidden),
            "candidate_vs_reference": len(candidate_hashes & reference_hashes),
            "reference_vs_benchmark_or_yaqa": len(reference_hashes & forbidden),
        },
        "shards": [
            {k: str(v) if isinstance(v, Path) else v for k, v in row.items()}
            for row in shard_records
        ],
    }
    (out_dir / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n"
    )
    if os.environ.get("LLAMA_CALIBRATION_PREPARE_ONLY") == "1":
        print(f"[done] prepared {len(shard_records)} candidate shards", flush=True)
        return 0

    command = [
        sys.executable,
        str(repo_root / "optimize/calibration_coverage.py"),
        "--model",
        MODEL,
    ]
    for record in shard_records:
        command.extend(["--dataset", f"{record['path']}:{record['name']}"])
    command.extend(
        [
            "--reference",
            f"{reference_path}:reference",
            "--output-dir",
            str(scanner_dir),
            "--physical-gpu",
            os.environ.get("LLAMA_COVERAGE_PHYSICAL_GPU", "0"),
            "--concat-size",
            str(CONCAT_SIZE),
            "--sketch-samples",
            "128",
            "--torch-dtype",
            "float16",
            "--greedy-threads",
            "32",
            "--target-tokens",
            str(TARGET_TOKENS),
            "--min-target-tokens",
            str(MIN_TARGET_TOKENS),
            "--target-tokens-mode",
            "gain_per_token",
            "--max-samples",
            "0",
        ]
    )
    env = os.environ.copy()
    env["PYTHON_GIL"] = "0"
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, env=env, cwd=repo_root, check=True)

    report = json.loads((scanner_dir / "coverage_report.json").read_text())
    order = [row["dataset"] for row in report["greedy_ranking"]]
    frames = []
    for selection_order, name in enumerate(order, 1):
        frame = pd.read_parquet(shards_dir / f"{name}.parquet")
        frame["shard"] = name
        frame["selection_order"] = selection_order
        frames.append(frame[["messages", "shard", "selection_order"]])
    mix = pd.concat(frames, ignore_index=True)
    mix_path = out_dir / "calibration.parquet"
    mix.to_parquet(mix_path, index=False)

    selected_hashes = {content_hash(to_messages(row)) for row in mix["messages"]}
    evidence = {
        "model": MODEL,
        "target_tokens_floor": TARGET_TOKENS,
        "selected_tokens": report["selected_mix"]["total_tokens"],
        "selected_rows": len(mix),
        "selected_order": order,
        "benchmark": {"source": NM_LOCAL, "row_start": 128, "rows": 300},
        "yaqa": {"source": NM_LOCAL, "row_start": 512, "rows": 128},
        "coverage_reference_rows": len(reference_rows),
        "content_hash_intersections": {
            "selected_vs_benchmark_or_yaqa": len(selected_hashes & forbidden),
            "selected_vs_reference": len(selected_hashes & reference_hashes),
            "reference_vs_benchmark_or_yaqa": len(reference_hashes & forbidden),
        },
        "scanner_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "sources": [
            {k: str(v) if isinstance(v, Path) else v for k, v in row.items()}
            for row in shard_records
        ],
        "report": report,
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy(scanner_dir / "coverage_report.json", out_dir / "report.json")
    shutil.copy(scanner_dir / "coverage_report.md", out_dir / "report.md")
    print(
        f"[done] {mix_path}: rows={len(mix)} tokens={evidence['selected_tokens']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
