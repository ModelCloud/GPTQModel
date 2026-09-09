"""CPU real-checkpoint slice screen for GSQ-inspired P32 path selection.

Uses first-block q_proj inputs (embedding followed by RMSNorm), disjoint documents,
and a bounded inner-basis submatrix. This is not full-module/model validation.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from gptqmodel.quantization.qvq import (
    decode_p32_window_tiles,
    pack_qvq_binary_bank_ids,
    repack_p32_planar_to_window,
    repack_p32_window_to_planar,
    rht_preprocess_weight,
    unpack_qvq_binary_bank_ids,
)
from gptqmodel.quantization.qvq_gsq import refine_p32_candidates
from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gsq", action="store_true", help="Enable experimental GSQ refinement (default: disabled)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    prefix = "model.layers.0.self_attn.q_proj"
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]

    def read(name):
        with safe_open(str(args.snapshot / index[name]), framework="pt") as f:
            return f.get_tensor(name)

    t, su, sv, bank, alt = [read(prefix + "." + suffix) for suffix in
                            ("trellis", "SU", "SV", "bank_ids", "bank_alt_id")]
    k, n = su.numel(), sv.numel()
    bits = t.shape[-1] / 8
    cfg = json.loads((args.snapshot / "quantize_config.json").read_text())
    with safe_open(str(args.dense / "model.safetensors"), framework="pt") as f:
        weight = f.get_tensor(prefix + ".weight").float()
        norm = f.get_tensor("model.layers.0.input_layernorm.weight").float()
        embedding = f.get_tensor("model.embed_tokens.weight")
    tokenizer = AutoTokenizer.from_pretrained(args.dense, local_files_only=True)
    rows = pq.read_table(args.data).slice(0, 8).to_pylist()
    ids = [tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False,
                                       return_dict=False)[:64] for row in rows]
    fingerprints = [hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in ids]
    if len(set(fingerprints)) != 8:
        raise ValueError("expected eight distinct tokenized documents")
    eps = json.loads((args.dense / "config.json").read_text())["rms_norm_eps"]
    xrows = []
    for row in ids:
        x = embedding[torch.tensor(row)].float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm
        xrows.append(matmul_hadU(x * su.float())[:, :32].contiguous())
    del embedding
    # Invert deployed diagonal scales before each orthogonal transform.
    target = rht_preprocess_weight(weight, su.float().reciprocal(), sv.float().reciprocal())[:32, :32]
    tile_ids = torch.tensor([0, 1, n // 16, n // 16 + 1])
    base = repack_p32_planar_to_window(t, bits=bits)[tile_ids]
    bank = pack_qvq_binary_bank_ids(unpack_qvq_binary_bank_ids(bank, k * n // 32).reshape(-1, 8)[tile_ids].flatten())
    candidates = base.unsqueeze(0).repeat(33, 1, 1)
    generator = torch.Generator().manual_seed(args.seed)
    for c in range(1, len(candidates)):
        for tile in range(4):
            bit = int(torch.randint(base.shape[-1] * 32, (), generator=generator))
            value = int(candidates[c, tile, bit // 32]) ^ (1 << (bit % 32))
            candidates[c, tile, bit // 32] = (value + 2**31) % 2**32 - 2**31
    train = torch.cat(xrows[:4])
    result = refine_p32_candidates(candidates, bits=bits, bank_ids=bank, bank_alt_id=alt,
                                   target=target, inputs=train, codebook_version=cfg["codebook"],
                                   enabled=args.gsq, steps=args.steps, seed=args.seed)

    def matrix(words):
        return decode_p32_window_tiles(words, bits=bits, bank_ids=bank, bank_alt_id=alt,
                                       codebook_version=cfg["codebook"]).reshape(2, 2, 16, 16).permute(
                                           0, 2, 1, 3).reshape(32, 32)

    before, after = matrix(base), matrix(result.window_words)
    # Matched candidate-set control: coordinate descent uses hard calibration
    # loss only, without a relaxation. Do not attribute its gains to GSQ.
    greedy = base.clone()
    train_target = train @ target
    for _ in range(3):
        changed = False
        for tile in range(4):
            best_loss = float((train @ matrix(greedy) - train_target).square().mean())
            best_words = greedy[tile].clone()
            for candidate in candidates:
                trial = greedy.clone()
                trial[tile] = candidate[tile]
                value = float((train @ matrix(trial) - train_target).square().mean())
                if value < best_loss:
                    best_loss, best_words = value, candidate[tile].clone()
            changed |= not torch.equal(greedy[tile], best_words)
            greedy[tile] = best_words
        if not changed:
            break
    greedy_matrix = matrix(greedy)
    per_row = []
    for x in xrows[4:]:
        y = x @ target
        denom = y.square().mean()
        per_row.append({"before": float((x @ before - y).square().mean() / denom),
                        "after": float((x @ after - y).square().mean() / denom),
                        "greedy": float((x @ greedy_matrix - y).square().mean() / denom)})
    deltas = torch.tensor([row["after"] - row["before"] for row in per_row])
    samples = torch.randint(4, (2000, 4), generator=torch.Generator().manual_seed(0))
    interval = deltas[samples].mean(1).quantile(torch.tensor([0.025, 0.975])).tolist()
    roundtrip = repack_p32_planar_to_window(repack_p32_window_to_planar(result.window_words, bits=bits), bits=bits)
    assert torch.equal(roundtrip, result.window_words)
    torch.save({"baseline": base, "candidate": result.window_words, "bank_ids": bank, "bank_alt_id": alt,
                "target": target, "inputs": xrows, "input_ids": ids, "candidates": candidates,
                "greedy": greedy}, args.output / "slice.pt")
    reloaded = torch.load(args.output / "slice.pt", weights_only=True)
    assert torch.equal(matrix(reloaded["candidate"]), after)
    report = {"scope": "first q_proj 32x32 inner-basis slice; no full-module or final-logit quality claim",
              "method": "GSQ-inspired whole-tile categorical candidate relaxation; fixed scales and banks",
              "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "torch": torch.__version__, "hardware": platform.machine(), "backend": "CPU FP32 reference",
              "dense": str(args.dense), "snapshot": str(args.snapshot), "data": str(args.data),
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "module": prefix, "bits": bits, "codebook": cfg["codebook"], "seed": args.seed,
              "gsq_enabled": args.gsq,
              "steps": args.steps, "train_rows": [0, 1, 2, 3], "heldout_rows": [4, 5, 6, 7],
              "token_hashes": fingerprints, "token_counts": [len(row) for row in ids],
              "paired_delta_bootstrap95": interval,
              "uncertainty_scope": "four heldout documents only; conditional on this slice and candidate seed",
              "disjointness": "refinement train/eval documents disjoint; historical quantization overlap not audited",
              "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), Path("gptqmodel/quantization/qvq_gsq.py")]},
              "calibration_before": result.calibration_before, "calibration_after": result.calibration_after,
              "heldout": per_row, "choices": result.choices.tolist(),
              "payload_bytes_before": base.numel() * 4, "payload_bytes_after": result.window_words.numel() * 4,
              "roundtrip_exact": True, "reload_exact": True}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
