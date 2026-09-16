#!/usr/bin/env python3
"""Sampled FP64 accuracy gate for the fused sparse QVQ-GSQ relaxation."""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from gptqmodel.quantization.qvq import repack_p32_planar_to_window, rht_preprocess_weight
from gptqmodel.quantization.qvq_gsq import fisher_screened_trellis_candidates
from gptqmodel.quantization.qvq_gsq_triton import (
    build_compact_position_map,
    compact_position_error,
    gumbel_softmax,
)


ORACLE_ROOT = Path("/root/inference-ultra/deepseekv4.1-a100-custom")
sys.path.insert(0, str(ORACLE_ROOT))
from dsv41.numerical_oracle import closer_counts, error_metrics  # noqa: E402


def main():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    module = "model.layers.0.mlp.down_proj"
    qdir = Path("/root/qvq-results/w3-p32-gsq-ab-20260915/gsq")
    index = json.loads((qdir / "model.safetensors.index.json").read_text())["weight_map"]

    def quantized(suffix):
        key = module + "." + suffix
        with safe_open(str(qdir / index[key]), framework="pt", device="cuda:0") as handle:
            return handle.get_tensor(key)

    with safe_open("/monster/data/model/Llama-3.2-1B-Instruct/model.safetensors",
                   framework="pt", device="cuda:0") as handle:
        weight = handle.get_tensor(module + ".weight").float()
    trellis, su, sv, banks, alt = [
        quantized(name) for name in ("trellis", "SU", "SV", "bank_ids", "bank_alt_id")]
    baseline = repack_p32_planar_to_window(trellis, bits=3)
    target = rht_preprocess_weight(weight, su.reciprocal(), sv.reciprocal()).float()
    hessian_in = torch.eye(target.shape[0], device="cuda")
    hessian_out = torch.eye(target.shape[1], device="cuda")
    _, decoded, sparse_indices, sparse_deltas = fisher_screened_trellis_candidates(
        baseline, count=33, seed=7, bits=3, layout="p32_window", target=target,
        input_hessian=hessian_in, output_hessian=hessian_out,
        bank_ids=banks, bank_alt_id=alt, return_decoded=True, return_sparse=True,
    )
    decoded_by_tile = decoded.reshape(33, -1, 256).transpose(0, 1).contiguous()
    indices_by_tile = sparse_indices.permute(1, 0, 2).contiguous()
    deltas_by_tile = sparse_deltas.permute(1, 0, 2).bfloat16().contiguous()
    logits = torch.zeros(decoded_by_tile.shape[:2], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(7)
    uniform = torch.rand(logits.shape, device="cuda", generator=generator)
    eager_probability = ((100.0 * logits - torch.log(-torch.log(
        uniform.clamp(1e-6, 1 - 1e-6)))) / 2.0).softmax(-1)
    fused_probability = torch.empty_like(logits)
    gumbel_softmax(logits, uniform, fused_probability, 2.0, 100.0)

    dense_tiles = torch.bmm(
        eager_probability.bfloat16().unsqueeze(1), decoded_by_tile.bfloat16(),
    ).squeeze(1)
    dense_weight = dense_tiles.reshape(
        target.shape[0] // 16, target.shape[1] // 16, 16, 16,
    ).permute(0, 2, 1, 3).reshape_as(target)
    position_indices, position_choices, position_deltas = build_compact_position_map(
        indices_by_tile, deltas_by_tile,
    )
    sparse_error_weight = torch.empty_like(target, dtype=torch.bfloat16)
    compact_position_error(
        fused_probability, decoded_by_tile[:, 0].bfloat16(), position_indices,
        position_choices, position_deltas, target.bfloat16(), sparse_error_weight,
    )
    sparse_weight = sparse_error_weight + target.bfloat16().float()
    torch.cuda.synchronize()

    rows = torch.linspace(0, target.shape[0] - 1, 64, device="cuda").long().unique()
    columns = torch.linspace(0, target.shape[1] - 1, 64, device="cuda").long().unique()
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    output_tiles = target.shape[1] // 16
    tile_ids = (row_grid // 16) * output_tiles + column_grid // 16
    inner_rows, inner_columns = row_grid % 16, column_grid % 16
    exact_values = decoded[:, tile_ids, inner_rows, inner_columns].permute(1, 2, 0).double()
    exact_probabilities = (
        (100.0 * logits.double() - torch.log(-torch.log(
            uniform.double().clamp(1e-6, 1 - 1e-6)))) / 2.0
    ).softmax(-1)
    reference = (
        exact_values * exact_probabilities[tile_ids].double()
    ).sum(-1)
    dense_sample = dense_weight.index_select(0, rows).index_select(1, columns)
    sparse_sample = sparse_weight.index_select(0, rows).index_select(1, columns)
    probability_reference = (
        (100.0 * logits.double() - torch.log(-torch.log(
            uniform.double().clamp(1e-6, 1 - 1e-6)))) / 2.0
    ).softmax(-1)
    result = {
        "policy": {"float32_matmul_precision": "highest", "tf32": False},
        "samples": int(reference.numel()),
        "dense_bf16": error_metrics(dense_sample, reference),
        "fused_sparse_fp32_accum": error_metrics(sparse_sample, reference),
        "closer_counts": closer_counts(sparse_sample, dense_sample, reference),
        "eager_probability": error_metrics(eager_probability, probability_reference),
        "fused_probability": error_metrics(fused_probability, probability_reference),
        "probability_closer_counts": closer_counts(
            fused_probability, eager_probability, probability_reference),
        "max_dense_sparse_difference": float(
            (dense_sample.float() - sparse_sample.float()).abs().max()),
    }
    output = Path("/root/qvq-results/gsq-performance-20260916/fp64-oracle.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
