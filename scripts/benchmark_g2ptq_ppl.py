# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""WikiText-2 perplexity for the G2PTQ solver.

Quantizes one model with plain GPTQ and with G2PTQ at the requested
trust-region budgets, then scores WikiText-2 perplexity for every result and
for the unquantized model. Calibration and evaluation windows are disjoint
slices of the tokenizer's train/test streams.

Example:

    python scripts/benchmark_g2ptq_ppl.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --calib-samples 24 --eval-chunks 16 --alphas 0.05,0.2
"""

import argparse
import json
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import G2PTQConfig, QuantizeConfig


def token_chunks(tokenizer, split, seqlen, limit):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row for row in dataset["text"] if row.strip())
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    chunks = []
    for start in range(0, len(ids) - seqlen + 1, seqlen):
        chunks.append(ids[start:start + seqlen])
        if len(chunks) >= limit:
            break
    return chunks


def calibration_texts(tokenizer, seqlen, limit):
    chunks = token_chunks(tokenizer, "train", seqlen, limit)
    return [tokenizer.decode(chunk) for chunk in chunks]


def perplexity(model, chunks, device):
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for chunk in chunks:
            input_ids = chunk.unsqueeze(0).to(device)
            # The causal-LM loss shifts labels internally; a pre-shifted pair
            # would score every token against the wrong target.
            out = model.model(input_ids=input_ids, labels=input_ids)
            count = input_ids.shape[1] - 1
            total_nll += float(out.loss) * count
            total_tokens += count
    return math.exp(total_nll / total_tokens)


def _is_cuda(device):
    return torch.device(device).type == "cuda"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calib-samples", type=int, default=24)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--eval-chunks", type=int, default=16)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--desc-act", action="store_true")
    parser.add_argument("--alphas", default="0.05")
    parser.add_argument("--methods", default="dense,gptq,g2ptq")
    parser.add_argument("--out", default="g2ptq_ppl.json")
    args = parser.parse_args()

    device = args.device
    cuda = _is_cuda(device)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    calibration = calibration_texts(tokenizer, args.calib_seqlen, args.calib_samples)
    eval_chunks = token_chunks(tokenizer, "test", args.eval_seqlen, args.eval_chunks)
    print(
        f"model={args.model} calib={len(calibration)}x{args.calib_seqlen} "
        f"eval={len(eval_chunks)}x{args.eval_seqlen}",
        flush=True,
    )

    results = {}
    methods = args.methods.split(",")
    alphas = [float(value) for value in args.alphas.split(",")]

    if "dense" in methods:
        dense = GPTQModel.load(args.model, device=device, dtype=torch.float16)
        dense.model.to(device)
        start = time.time()
        dense_ppl = perplexity(dense, eval_chunks, device)
        results["dense"] = {"ppl": dense_ppl, "seconds": time.time() - start}
        print(f"dense ppl={dense_ppl:.4f}", flush=True)
        del dense
        if cuda:
            torch.cuda.empty_cache()

    runs = []
    if "gptq" in methods:
        runs.append(("gptq", None))
    if "g2ptq" in methods:
        runs.extend((f"g2ptq_a{alpha:g}", alpha) for alpha in alphas)

    for name, alpha in runs:
        kwargs = {
            "bits": args.bits,
            "group_size": args.group_size,
            "desc_act": args.desc_act,
            "device": device,
        }
        if alpha is not None:
            kwargs["g2ptq"] = G2PTQConfig(alpha_mse=alpha, alpha_kl=alpha)
        qcfg = QuantizeConfig(**kwargs)

        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = GPTQModel.load(args.model, qcfg, dtype=torch.float16)
        start = time.time()
        model.quantize(
            CalibrationConfig(
                dataset=calibration,
                concat_size=args.calib_seqlen,
                batch_size=1,
                sort=None,
            ),
            backend=BACKEND.GPTQ_TORCH,
        )
        quant_seconds = time.time() - start
        model.model.to(device)
        peak_gib = (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3 if cuda else 0.0
        )
        start = time.time()
        ppl = perplexity(model, eval_chunks, device)
        eval_seconds = time.time() - start
        results[name] = {
            "ppl": ppl,
            "quantize_seconds": quant_seconds,
            "eval_seconds": eval_seconds,
            "peak_gib": peak_gib,
            "alpha": alpha,
        }
        print(
            f"{name} ppl={ppl:.4f} quantize={quant_seconds:.1f}s "
            f"eval={eval_seconds:.1f}s peak={peak_gib:.2f}GiB",
            flush=True,
        )
        del model
        if cuda:
            torch.cuda.empty_cache()

    if "dense" in results:
        base = results["dense"]["ppl"]
        for name, value in results.items():
            if name != "dense":
                value["delta_vs_dense_pct"] = 100.0 * (value["ppl"] - base) / base
    if "gptq" in results:
        base = results["gptq"]["ppl"]
        for name, value in results.items():
            if name != "gptq":
                value["delta_vs_gptq_pct"] = 100.0 * (value["ppl"] - base) / base

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "results": results}, handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
