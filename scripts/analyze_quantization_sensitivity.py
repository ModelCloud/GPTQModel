#!/usr/bin/env python3
"""Run the model-agnostic pre-quantization sensitivity profiler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.config import QVQConfig
from gptqmodel.quantization.sensitivity import SensitivityProfiler


def _batches(path: Path, tokenizer, *, batch_size: int, max_length: int):
    rows = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                rows.append(value.get("text") or value.get("prompt") or value.get("question") or str(value))
    else:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=str(path), split="train")
        for row in ds:
            rows.append(row.get("text") or row.get("prompt") or row.get("question") or str(row))
    for start in range(0, len(rows), batch_size):
        enc = tokenizer(rows[start:start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        yield enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant-config")
    ap.add_argument("--analysis-data", type=Path)
    ap.add_argument("--candidate-rates", default="2,2.5,3,3.5,4")
    ap.add_argument("--target-bpw", type=float, action="append")
    ap.add_argument("--mode", choices=("fast", "balanced"), default="balanced")
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    qcfg = QVQConfig(bits=2, format="qvq_v2b2_p32")
    if args.quant_config:
        payload = json.loads(Path(args.quant_config).read_text(encoding="utf-8"))
        qcfg = QVQConfig(**{k: v for k, v in payload.items() if k in {"bits", "format", "group_size", "sym", "dynamic"}})
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if "cuda" in args.device else torch.float32, device_map=args.device)
    profiler = SensitivityProfiler(qcfg, candidate_rates=[float(x) for x in args.candidate_rates.split(",")], device=args.device)
    batches = _batches(args.analysis_data, tokenizer, batch_size=args.batch_size, max_length=args.max_length) if args.analysis_data else None
    report = profiler.scan(model, batches=batches, max_batches=args.max_batches, mode=args.mode)
    if args.target_bpw:
        report["plans"] = [profiler.plan(report["records"], target_bpw=t) for t in args.target_bpw]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sensitivity_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "module_scores.json").write_text(json.dumps(report["records"], indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
