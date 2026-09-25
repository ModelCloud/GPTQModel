#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Arbitrate serialized P32 vocabulary-head blocks under one full-head Fisher."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-parquet", type=Path, required=True)
    parser.add_argument("--factor-cache", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Permit a non-serving smoke artifact")
    return parser.parse_args()


def choose_guarded_blocks(base_z, base_penalty, deltas, h, input_damping, output_damping):
    """Greedily choose block changes that improve both complete FP64 oracles."""
    hd = h.clone()
    hd.diagonal().add_(input_damping)
    z = base_z.clone()
    penalty = base_penalty.clone()
    selected = []
    remaining = set(range(len(deltas)))
    self_costs = [
        ((dz * (h @ dz)).sum(), (dz * (hd @ dz)).sum() + output_damping * dp)
        for dz, dp in deltas
    ]
    while remaining:
        hz = h @ z
        hdz = hd @ z
        best = None
        for index in sorted(remaining):
            dz, _ = deltas[index]
            undamped_change = 2 * (dz * hz).sum() + self_costs[index][0]
            damped_change = 2 * (dz * hdz).sum() + self_costs[index][1]
            if undamped_change < 0 and damped_change < 0 and (best is None or damped_change < best[1]):
                best = (index, damped_change)
        if best is None:
            break
        index = best[0]
        dz, dp = deltas[index]
        z.add_(dz)
        penalty.add_(dp)
        remaining.remove(index)
        selected.append(index)
    return selected, z, penalty


def main() -> None:
    args = parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.quantization.qvq import (
        reconstruct_qvq_inner_weight,
        rht_reconstruct_weight,
    )
    from gptqmodel.utils.model import untie_word_embeddings
    from scripts.experiments.qvq_vocab_block_probe import (
        load_shared_factor_cache,
        sha256_file,
    )
    from scripts.experiments.qvq_vocab_head_artifact import (
        _oracle,
        _oracle_contribution,
    )

    artifact = args.artifact_dir
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest.get("schema") != "qvq.vocab-head-delta.v1":
        raise ValueError("unsupported vocabulary delta schema")
    if manifest.get("complete_head") != "true" and not args.allow_partial:
        raise ValueError("whole-head arbitration requires a complete vocabulary delta")
    if float(manifest["bits"]) >= 4:
        raise NotImplementedError("whole-head serialization arbitration currently supports P32 only")
    if str(args.model_path.resolve()) != manifest["model_path"]:
        raise ValueError("artifact model provenance mismatch")
    if sha256_file(args.factor_cache) != manifest["factor_cache_sha256"]:
        raise ValueError("factor-cache hash differs from artifact provenance")
    if (artifact / "guarded.safetensors").exists() or (artifact / "arbitration.json").exists():
        raise FileExistsError("Refusing to overwrite guarded vocabulary-head artifact")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    model = GPTQModel.load(
        str(args.model_path), device=DEVICE.CUDA, backend=BACKEND.AUTO,
        attn_implementation="eager",
    )
    model.model = untie_word_embeddings(model.model)
    head = model.get_output_embeddings()
    device = head.weight.device
    expected = {
        "schema": "qvq.yaqa.shared-head-factor.v2",
        "model_path": manifest["model_path"],
        "calibration_sha256": sha256_file(args.calibration_parquet),
        "requested_sequences": str(manifest["independent_sequences"]),
        "gram_rank": str(manifest["gram_rank"]),
        "batch_size": "1", "seed": str(manifest["seed"]),
        "input_features": str(head.in_features),
        "output_features": str(head.out_features),
    }
    inputs, outputs, _ = load_shared_factor_cache(args.factor_cache, expected)
    h = inputs["lm_head"].materialize(device=device)
    s = outputs["lm_head"].factor(device=device)
    hdamp = float(manifest["input_damping"])
    gdamp = float(manifest["output_damping"])
    base_z = torch.zeros((head.in_features, s.shape[1]), device=device, dtype=torch.float64)
    base_penalty = torch.zeros((), device=device, dtype=torch.float64)
    cand_z = torch.zeros_like(base_z)
    cand_penalty = torch.zeros_like(base_penalty)
    deltas = []
    z32 = {arm: torch.zeros((head.in_features, s.shape[1]), device=device, dtype=torch.float32)
           for arm in ("baseline", "candidate")}
    penalty32 = {arm: torch.zeros((), device=device, dtype=torch.float32)
                 for arm in ("baseline", "candidate")}
    deltas32 = []
    count = int(manifest["block_count"])
    bits = float(manifest["bits"])
    block_rows = int(manifest["block_rows"])
    with (
        safe_open(str(artifact / "baseline.safetensors"), framework="pt", device="cpu") as baseline,
        safe_open(str(artifact / "candidate.safetensors"), framework="pt", device="cpu") as candidate,
    ):
        for index in range(count):
            start = index * block_rows
            stop = min(start + block_rows, head.out_features)
            target = head.weight.detach()[start:stop].float()
            sf = s[start:stop]
            contributions = {}
            contributions32 = {}
            for arm, packed in (("baseline", baseline), ("candidate", candidate)):
                prefix = f"lm_head.blocks.{index}."
                tensors = {key.removeprefix(prefix): packed.get_tensor(key).to(device=device)
                           for key in packed.keys() if key.startswith(prefix)}  # noqa: SIM118
                inner = reconstruct_qvq_inner_weight(
                    tensors["trellis"], bits=bits, in_features=head.in_features,
                    out_features=stop - start,
                    bank_ids=tensors["bank_ids"], bank_alt_id=tensors["bank_alt_id"],
                    v2b2_p32=True,
                )
                reconstructed = rht_reconstruct_weight(
                    inner, tensors["SU"], tensors["SV"],
                ).to(dtype=head.weight.dtype)
                contributions[arm] = _oracle_contribution(
                    reconstructed.float() - target, sf, h, hdamp, dtype=torch.float64,
                )
                contributions32[arm] = _oracle_contribution(
                    reconstructed.float() - target, sf, h, hdamp, dtype=torch.float32,
                )
            bz, bp = contributions["baseline"]
            cz, cp = contributions["candidate"]
            base_z.add_(bz)
            base_penalty.add_(bp)
            cand_z.add_(cz)
            cand_penalty.add_(cp)
            deltas.append((cz - bz, cp - bp))
            for arm in ("baseline", "candidate"):
                zi, pi = contributions32[arm]
                z32[arm].add_(zi)
                penalty32[arm].add_(pi)
            deltas32.append((contributions32["candidate"][0] - contributions32["baseline"][0],
                             contributions32["candidate"][1] - contributions32["baseline"][1]))
            print(f"decoded={index + 1}/{count}", flush=True)
        base_scores = _oracle(base_z, base_penalty, h, hdamp, gdamp, dtype=torch.float64)
        cand_scores = _oracle(cand_z, cand_penalty, h, hdamp, gdamp, dtype=torch.float64)
        base_scores32 = _oracle(z32["baseline"], penalty32["baseline"], h, hdamp, gdamp,
                                dtype=torch.float32)
        cand_scores32 = _oracle(z32["candidate"], penalty32["candidate"], h, hdamp, gdamp,
                                dtype=torch.float32)
        recorded = manifest["covered_rows_oracle"]
        for arm, precision, scores in (
            ("baseline", "fp64", base_scores), ("candidate", "fp64", cand_scores),
            ("baseline", "fp32", base_scores32), ("candidate", "fp32", cand_scores32),
        ):
            for metric, observed in zip(("undamped", "damped"), scores):
                expected_score = recorded[arm][precision][metric]
                if abs(observed - expected_score) > max(abs(expected_score) * 1e-5, 1e-12):
                    raise ValueError(
                        f"serialized {arm} {precision} {metric} oracle {observed:.12g} differs from "
                        f"in-memory {expected_score:.12g}"
                    )
        selected, guarded_z, guarded_penalty = choose_guarded_blocks(
            base_z, base_penalty, deltas, h.to(torch.float64), hdamp, gdamp,
        )
        guarded_scores = _oracle(
            guarded_z, guarded_penalty, h, hdamp, gdamp, dtype=torch.float64,
        )
        selected_set = set(selected)
        guarded_z32 = z32["baseline"].clone()
        guarded_penalty32 = penalty32["baseline"].clone()
        for index in selected:
            guarded_z32.add_(deltas32[index][0])
            guarded_penalty32.add_(deltas32[index][1])
        guarded_scores32 = _oracle(
            guarded_z32, guarded_penalty32, h, hdamp, gdamp, dtype=torch.float32,
        )
        if any(candidate > baseline * (1 + 1e-6)
               for candidate, baseline in zip(guarded_scores32, base_scores32)):
            raise ValueError("FP32 guarded whole-head oracle regressed after FP64 arbitration")
        guarded_tensors = {}
        for index in range(count):
            source = candidate if index in selected_set else baseline
            prefix = f"lm_head.blocks.{index}."
            for key in source.keys():  # noqa: SIM118 - safetensors reader API
                if key.startswith(prefix):
                    guarded_tensors[key] = source.get_tensor(key).contiguous()
        save_file(
            guarded_tensors, str(artifact / "guarded.safetensors"),
            metadata=baseline.metadata(),
        )
    report = {
        "schema": "qvq.vocab-head-arbitration.v1",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "selected_blocks": selected,
        "selected_count": len(selected),
        "baseline_fp64": {"undamped": base_scores[0], "damped": base_scores[1]},
        "all_candidate_fp64": {"undamped": cand_scores[0], "damped": cand_scores[1]},
        "guarded_fp64": {"undamped": guarded_scores[0], "damped": guarded_scores[1]},
        "baseline_fp32": {"undamped": base_scores32[0], "damped": base_scores32[1]},
        "all_candidate_fp32": {"undamped": cand_scores32[0], "damped": cand_scores32[1]},
        "guarded_fp32": {"undamped": guarded_scores32[0], "damped": guarded_scores32[1]},
        "source": "reconstructed_serialized_p32_weights",
        "status": "offline_delta_no_serving_claim",
    }
    (artifact / "arbitration.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
