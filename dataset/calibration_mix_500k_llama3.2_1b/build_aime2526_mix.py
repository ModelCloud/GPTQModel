"""Append leakage-safe AIME 2025/2026 prompts to the YAQA calibration mix.

The D300 development math prompts are excluded by canonical message hash.  The
result is intentionally an experiment artifact, not a benchmark manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "calibration.parquet"
OUT = ROOT / "calibration_aime2526.parquet"
D300 = Path("/root/qvq-data/divergence300-v1/divergence300-development.jsonl")
CACHE = ROOT.parent / "hf_cache"


def h(messages):
    messages = [{"role": str(x["role"]), "content": str(x["content"])} for x in messages]
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    base = pd.read_parquet(BASE)
    base["source"] = base.get("shard")
    base["split"] = "base"

    forbidden = set()
    # The manifest is JSONL, but some prompt payloads contain literal newlines
    # in older snapshots; hashes are sufficient and can be read robustly.
    import re
    forbidden.update(re.findall(r'"prompt_sha256"\s*:\s*"([0-9a-f]{64})"', D300.read_text()))
    # D300 stores the canonical hash, while compute it here for independent
    # verification against the exact messages we append.
    selected = []
    excluded = []
    seen = {h(x.tolist() if hasattr(x, "tolist") else x) for x in base["messages"]}
    for ds_name in ("MathArena/aime_2025", "MathArena/aime_2026"):
        ds = load_dataset(ds_name, split="train", cache_dir=str(CACHE))
        for row in ds:
            messages = [{"role": "user", "content": "Solve the following competition mathematics problem step by step, then give the final answer clearly.\n\n" + row["problem"].strip()}]
            digest = h(messages)
            if digest in forbidden:
                excluded.append({"dataset": ds_name, "problem_idx": row["problem_idx"], "hash": digest})
            elif digest not in seen:
                seen.add(digest)
                selected.append({
                    "messages": messages,
                    "source": ds_name,
                    "split": "aime2526",
                    "problem_idx": row["problem_idx"],
                    "hash": digest,
                })
    extra = pd.DataFrame(selected)
    combined = pd.concat([base, extra], ignore_index=True)
    combined.to_parquet(OUT, index=False)
    info = {
        "base": str(BASE),
        "output": str(OUT),
        "base_rows": len(base),
        "added_rows": len(selected),
        "output_rows": len(combined),
        "sources": ["MathArena/aime_2025", "MathArena/aime_2026"],
        "excluded_d300_overlap": excluded,
        "excluded_count": len(excluded),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "prompt_hash_policy": "sha256(canonical JSON messages), exact D300 development prompt hashes excluded",
    }
    (ROOT / "calibration_aime2526.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
