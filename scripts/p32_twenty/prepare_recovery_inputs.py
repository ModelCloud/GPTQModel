"""Bounded recovery calibration from the verified historical Fisher dataset."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCE = Path("/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet")
EXPECTED = "5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39"
OUTPUT = Path("/root/p32-recovery-calibration/inputs.json")


def main():
    with SOURCE.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != EXPECTED:
        raise ValueError("Historical Fisher dataset hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        "/monster/data/model/Llama-3.2-1B-Instruct", local_files_only=True
    )
    table = pq.read_table(
        SOURCE, columns=["messages", "union_index", "prepared_example_sha256"]
    ).slice(0, 16)
    rows = []
    for row in table.to_pylist():
        ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False, return_dict=False
        )
        rows.append(
            {
                "union_index": row["union_index"],
                "prepared_example_sha256": row["prepared_example_sha256"],
                "input_ids": ids[:2048],
            }
        )
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "scope": "bounded recovery calibration only; no evaluation benchmark; not historical re-quantization",
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
