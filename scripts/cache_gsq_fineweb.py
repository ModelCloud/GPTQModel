#!/usr/bin/env python3
"""Build a reusable paper-geometry FineWeb-Edu packed-token cache."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer

DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--gpt-samples", type=int, default=512)
    parser.add_argument("--evaluation-samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    counts = {
        "train": args.train_samples,
        "validation": args.validation_samples,
        "gpt": args.gpt_samples,
        "evaluation": args.evaluation_samples,
    }
    if any(value < 1 for value in (*counts.values(), args.sequence_length, args.shuffle_buffer)):
        raise ValueError("FineWeb-Edu cache sizes must be positive")
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite a token cache or manifest")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    total = sum(counts.values())
    chunks = []
    buffer = []
    iterator = iter(dataset)
    try:
        for row in iterator:
            # This intentionally matches official GSQ's tokenizer call. It
            # does not override add_special_tokens.
            buffer.extend(tokenizer(row["text"], return_tensors=None)["input_ids"])
            while len(buffer) >= args.sequence_length:
                chunks.append(torch.tensor(buffer[:args.sequence_length], dtype=torch.int32))
                del buffer[:args.sequence_length]
                if len(chunks) % 128 == 0 or len(chunks) == total:
                    print(f"PACKED {len(chunks)}/{total}", flush=True)
                if len(chunks) == total:
                    break
            if len(chunks) == total:
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if len(chunks) != total:
        raise RuntimeError(f"FineWeb-Edu stream yielded only {len(chunks)}/{total} chunks")
    tokens = torch.stack(chunks)
    hashes = [sha256_bytes(row.numpy().tobytes()) for row in tokens]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("packed FineWeb-Edu cache contains duplicate token chunks")
    starts = {
        "train": 0,
        "validation": counts["train"],
        "gpt": counts["train"] + counts["validation"],
        "evaluation": counts["train"] + counts["validation"] + counts["gpt"],
    }
    splits = {
        name: {
            "start": starts[name],
            "count": count,
            "tokens": count * args.sequence_length,
            "hashes": hashes[starts[name]:starts[name] + count],
        }
        for name, count in counts.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"tokens": tokens},
        args.output,
        metadata={
            "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
            "dataset_revision": DATASET_REVISION,
            "sequence_length": str(args.sequence_length),
            "shuffle_seed": str(args.seed),
            "shuffle_buffer": str(args.shuffle_buffer),
            "tokenizer_special_tokens": "default",
        },
    )
    manifest = {
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
        "dataset_revision": DATASET_REVISION,
        "source_role": "packed sequential train/selection/QK-metric/report-only ranges",
        "shuffle_seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "sequence_length": args.sequence_length,
        "tokenizer": str(args.model.resolve()),
        "tokenizer_json_sha256": file_digest(args.model / "tokenizer.json"),
        "tokenizer_add_special_tokens": "default (matches official GSQ)",
        "splits": splits,
        "strict_chunk_hash_disjointness": True,
        "cache": str(args.output.resolve()),
        "cache_sha256": file_digest(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "cache": str(args.output.resolve()),
        "cache_sha256": manifest["cache_sha256"],
        "samples": total,
        "tokens": int(tokens.numel()),
        "splits": {name: {key: value for key, value in split.items() if key != "hashes"}
                   for name, split in splits.items()},
    }))


if __name__ == "__main__":
    main()
