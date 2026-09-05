import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

root = Path("/root/p32-model-baseline")
root.mkdir(exist_ok=True)
tok = AutoTokenizer.from_pretrained("/monster/data/model/Llama-3.2-1B-Instruct")


def norm(t):
    return " ".join(t.casefold().split())


cal = pq.read_table(
    "/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet",
    columns=["messages"],
).to_pylist()
seen = {norm(m["content"]) for row in cal for m in row["messages"]}
rows = []
source = Path("/monster/data/model/dataset/c4-train.00000-of-01024.json.gz")
with gzip.open(source, "rt") as f:
    for i, line in enumerate(f):
        doc = json.loads(line)
        text = doc["text"]
        if norm(text) in seen:
            continue
        ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=256)
        if len(ids) < 256:
            continue
        if norm(tok.decode(ids)) in seen:
            continue
        rows.append(
            {
                "source_row": i,
                "input_ids": ids,
                "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
        if len(rows) == 16:
            break
with source.open("rb") as f:
    sha = hashlib.file_digest(f, "sha256").hexdigest()
(root / "inputs.json").write_text(
    json.dumps(
        {
            "source": str(source),
            "source_sha256": sha,
            "rows": rows,
            "scope": "16 C4 documents, 256 tokens each; bounded baseline, no calibration or tuning",
            "disjointness": "Exact normalized text exclusions against historical Fisher messages; not semantic near-duplicate proof",
        },
        indent=2,
    )
)
print("Prepared", len(rows), "documents")
