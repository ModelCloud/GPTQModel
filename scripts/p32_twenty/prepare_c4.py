"""Prepare document-disjoint ordinary-text streams for the F6 seed-7 study."""

import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCE = Path("/monster/data/model/dataset/c4-train.00000-of-01024.json.gz")
OUTPUT = Path("/root/work/p32-twenty-data")
MODEL = Path("/monster/data/model/Llama-3.2-1B-Instruct")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    streams = [
        ("lifecycle", 128),
        ("teacher_calibration", 10178),
        ("candidate_calibration", 512),
        ("tuning", 128),
        ("heldout", 256),
    ]
    manifest = {
        "source": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "token_limit": 512,
        "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
        "streams": {},
    }
    seen_text, seen_url = set(), set()
    with gzip.open(SOURCE, "rt") as handle:
        rows = enumerate(handle)
        for name, count in streams:
            records, bindings = [], []
            while len(records) < count:
                index, line = next(rows)
                doc = json.loads(line)
                original = doc["text"]
                digest = hashlib.sha256(
                    " ".join(original.casefold().split()).encode()
                ).hexdigest()
                url = doc.get("url", "")
                if digest in seen_text or (url and url in seen_url):
                    continue
                seen_text.add(digest)
                if url:
                    seen_url.add(url)
                ids = tokenizer.encode(
                    original, add_special_tokens=False, truncation=True, max_length=512
                )
                if len(ids) < 64:
                    continue
                text = tokenizer.decode(ids, skip_special_tokens=True)
                records.append({"text": text})
                bindings.append(
                    {
                        "source_row": index,
                        "document_sha256": digest,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "tokens": len(ids),
                    }
                )
            path = OUTPUT / f"{name}.parquet"
            pq.write_table(pa.Table.from_pylist(records), path)
            manifest["streams"][name] = {
                "path": str(path),
                "sha256": sha(path),
                "rows": len(records),
                "tokens": sum(row["tokens"] for row in bindings),
                "documents": bindings,
            }
            print(name, len(records), manifest["streams"][name]["tokens"], flush=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
