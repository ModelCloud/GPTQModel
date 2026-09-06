"""Bounded recovery calibration from the verified historical Fisher dataset."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCE = Path("/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet")
EXPECTED = "5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39"
OUTPUT = Path("/root/p32-recovery-calibration/inputs.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--minimum-tokens", type=int, default=0)
    args = parser.parse_args()
    if args.rows < 1 or args.minimum_tokens < 0 or args.minimum_tokens > 2048:
        parser.error("Invalid row count or minimum token length")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing calibration manifest")
    with SOURCE.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != EXPECTED:
        raise ValueError("Historical Fisher dataset hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        "/monster/data/model/Llama-3.2-1B-Instruct", local_files_only=True
    )
    table = pq.read_table(
        SOURCE, columns=["messages", "union_index", "prepared_example_sha256"]
    )
    rows = []
    indices = list(range(len(table)))
    if args.sample_seed is not None:
        random.Random(args.sample_seed).shuffle(indices)
    for index in indices:
        row = table.slice(index, 1).to_pylist()[0]
        ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if len(ids) < args.minimum_tokens:
            continue
        rows.append(
            {
                "union_index": row["union_index"],
                "prepared_example_sha256": row["prepared_example_sha256"],
                "input_ids": ids[:2048],
            }
        )
        if len(rows) == args.rows:
            break
    if len(rows) != args.rows:
        raise ValueError("Insufficient eligible historical documents")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "bounded recovery calibration only; no evaluation benchmark; not historical re-quantization",
                "sample_seed": args.sample_seed,
                "minimum_tokens": args.minimum_tokens,
                "selection": "shuffled without replacement"
                if args.sample_seed is not None
                else "source order",
                "source": str(SOURCE),
                "source_sha256": digest,
                "capture_scope": "calibration-only activations for residual fitting; disjoint evaluation remains required",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Verified historical Fisher source; wrote",
        len(rows),
        "bounded calibration rows",
    )


if __name__ == "__main__":
    main()
