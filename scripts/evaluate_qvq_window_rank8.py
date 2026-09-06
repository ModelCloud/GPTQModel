#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fit real P32 modules on calibration documents and audit on a third disjoint fold."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_idle_preflight import add_gpu_idle_preflight_args, bootstrap_gpu_idle_preflight


def main():
    idle = bootstrap_gpu_idle_preflight()
    parser = argparse.ArgumentParser(description=__doc__)
    add_gpu_idle_preflight_args(parser)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-parquet", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj"],
    )
    parser.add_argument(
        "--rows-per-document",
        type=int,
        default=64,
        help="bounded activation rows retained for each module/document",
    )
    parser.add_argument(
        "--max-capture-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="hard CPU activation-capture bound; increase only with memory headroom",
    )
    parser.add_argument(
        "--max-solver-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="per-module bounded fitting workspace",
    )
    args = parser.parse_args()
    import hashlib
    import json
    import time

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.qvq_rank8 import (
        P32WindowConfig,
        _metrics,
        fit_rank8,
        prepare_rank8,
        save_window_package,
    )
    from gptqmodel.quantization.qvq_rank8_capture import (
        Rank8Capture,
        Rank8Document,
        capture_rank8_calibration,
    )
    from gptqmodel.utils.backend import BACKEND

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = args.calibration_parquet
    teacher_path = args.teacher
    checkpoint = args.checkpoint
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    names = tuple(args.modules)
    tok = AutoTokenizer.from_pretrained(teacher_path)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, dtype=torch.float16, device_map="cuda"
    ).eval()
    batch = next(pq.ParquetFile(source).iter_batches(
        batch_size=16, columns=["messages", "normalized_user_sha256"]
    ), None)
    rows = [] if batch is None else batch.to_pylist()
    docs = []
    if len(rows) != 16:
        raise ValueError(
            "requires 16 calibration documents (8 train, 4 selection, 4 audit)"
        )
    for row in rows:
        ids = tok.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )["input_ids"][:512]
        docs.append(
            Rank8Document(
                row["normalized_user_sha256"],
                {"input_ids": torch.tensor([ids], device="cuda")},
            )
        )
    print("Capturing dense inputs", flush=True)
    calibration = capture_rank8_calibration(
        teacher,
        Rank8Capture(
            names,
            tuple(docs[:8]),
            tuple(docs[8:12]),
            rows_per_document=args.rows_per_document,
            max_bytes=args.max_capture_bytes,
            max_solver_bytes=args.max_solver_bytes,
        ),
    )
    audit = capture_rank8_calibration(
        teacher,
        Rank8Capture(
            names,
            tuple(docs[12:14]),
            tuple(docs[14:]),
            rows_per_document=args.rows_per_document,
            max_bytes=args.max_capture_bytes,
            max_solver_bytes=args.max_solver_bytes,
        ),
    )
    for name in names:
        torch.save(
            {
                "train": calibration[name].train_inputs,
                "selection": calibration[name].heldout_inputs,
                "audit_1": audit[name].train_inputs,
                "audit_2": audit[name].heldout_inputs,
            },
            out / (name + ".activations.pt"),
        )
    print("Loading deployed checkpoint", flush=True)
    quantized = GPTQModel.load(
        checkpoint, backend=BACKEND.QVQ, dtype=torch.float16, device_map={"": "cuda:0"}
    ).model.eval()
    with open(source, "rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "scope": (
            "Real P32 module fitting for the caller-selected module set with "
            "separate calibration, selection, and audit documents; not full-model evaluation."
        ),
        "teacher": teacher_path,
        "checkpoint": checkpoint,
        "source": source,
        "preflight": None if idle is None else idle.as_dict(),
        "torch_version": str(torch.__version__),
        "device": str(torch.cuda.get_device_properties(0)),
        "source_sha256": source_hash,
        "audit_document_ids": [d.document_id for d in docs[12:]],
        "train_document_ids": [d.document_id for d in docs[:8]],
        "selection_document_ids": [d.document_id for d in docs[8:12]],
        "capture_contract": {
            "rows_per_document": args.rows_per_document,
            "max_capture_bytes": args.max_capture_bytes,
            "max_solver_bytes": args.max_solver_bytes,
        },
        "modules": {},
    }
    for name in names:
        layer = quantized.get_submodule(name)
        dense = teacher.get_submodule(name)
        c = calibration[name]
        print("Fitting", name, flush=True)
        started = time.time()
        fitted = fit_rank8(
            layer,
            dense,
            c.train_inputs.cuda(),
            c.heldout_inputs.cuda(),
            train_document_ids=c.train_document_ids,
            heldout_document_ids=c.heldout_document_ids,
        )
        entry = {
            "fit": fitted, "fit_seconds": time.time() - started, "audit": [],
            "in_features": layer.in_features, "out_features": layer.out_features, "bits": layer.bits,
        }
        audit_rows = torch.cat((audit[name].train_inputs, audit[name].heldout_inputs))
        counts = [
            min(args.rows_per_document, d.inputs["input_ids"].shape[1])
            for d in docs[12:]
        ]
        for document, split in zip(docs[12:], audit_rows.split(counts)):
            x = split.cuda()
            with torch.no_grad():
                y = dense(x).float()
                prepare_rank8(layer, P32WindowConfig())
                fast = layer(x).float()
                prepare_rank8(
                    layer, P32WindowConfig(recovery_mode="auto", quality_mode="quality")
                )
                recovered = layer(x).float()
            entry["audit"].append(
                {
                    "document_id": document.document_id,
                    "rows": x.shape[0],
                    "baseline": _metrics(y - fast),
                    "recovered": _metrics(y - recovered),
                }
            )
        if fitted["validated"]:
            entry["storage"] = save_window_package(layer, out / (name + ".pt"))
        report["modules"][name] = entry
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            name,
            "validated",
            fitted["validated"],
            "fit seconds",
            entry["fit_seconds"],
            "audit",
            entry["audit"],
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
