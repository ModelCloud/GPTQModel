#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-device, synchronized NF12 checks and measurements for Blackhole."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

from gptqmodel.utils.nf12_tenstorrent import (
    LINEAR_PACKED_BUFFER_DEPTH,
    LINEAR_PACKED_SCRATCH_BYTES,
    prepare_nf12,
    round_up,
)


def _bits(x):
    return x.contiguous().view(torch.int16)


def _same(a, b):
    torch.testing.assert_close(_bits(a), _bits(b), atol=0, rtol=0)


def _gather_oracle(w, ids):
    valid = (ids >= 0) & (ids < len(w))
    return w[ids.clamp(0, len(w) - 1)].masked_fill(~valid[..., None], 0)


def _errors(output, oracle):
    output, oracle = output.float(), oracle.float()
    diff = output - oracle
    return {
        "max_abs": float(diff.abs().max()),
        "scaled_rmse": float(diff.square().mean().sqrt() / oracle.square().mean().sqrt().clamp_min(1e-20)),
        **{
            f"top{k}_overlap": float(
                (output.topk(k).indices.unsqueeze(-1) == oracle.topk(k).indices.unsqueeze(-2)).any(-1).float().mean()
            )
            for k in (1, 5, 10) if k <= oracle.shape[-1]
        },
    }


def _measure(op, sync, repeats, byte_count):
    out = op()
    sync(out)
    samples = []
    for _ in range(repeats):
        del out
        sync()
        start = time.perf_counter_ns()
        out = op()
        sync(out)
        samples.append((time.perf_counter_ns() - start) / 1000)
    median = statistics.median(samples)
    return out, {
        "median_us": median,
        "p95_us": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "effective_GBps": byte_count / median / 1000,
    }


@contextlib.contextmanager
def _runtime(backend, chip):
    if backend == "ttnn":
        from gptqmodel.utils.nf12_ttnn import TTNNNF12
        from gptqmodel.utils.ttnn_runtime import close_mesh, import_ttnn, open_mesh

        ttnn = import_ttnn()
        mesh = open_mesh(device_id=chip, owner="nf12-benchmark")
        try:
            yield TTNNNF12, mesh.device, ttnn.to_torch, lambda _output=None: ttnn.synchronize_device(mesh.device)
        finally:
            close_mesh(owner="nf12-benchmark")
    else:
        import torch_xla
        import torch_xla.runtime as xr
        import tt_torch  # noqa: F401

        from gptqmodel.utils.nf12_ttxla import TTXLANF12

        if xr.device_type() != "TT":
            raise RuntimeError("Launch the TT-XLA benchmark with PJRT_DEVICE=TT")
        device = torch_xla.device(chip)
        def complete(output=None):
            if output is None:
                torch_xla.sync(wait=True)
            else:
                # Force this specific lazy result to complete. Marking an XLA
                # step alone can time only submission for inference tensors.
                output.cpu()
        yield TTXLANF12, device, torch.Tensor.cpu, complete


def _embedding(w, name, runtime, backend, repeats, *, exhaustive=False):
    make, device, copy, sync = runtime
    descriptor, packed = prepare_nf12(w, name)
    layer = make(device, descriptor, packed)
    rows = len(w)
    if backend == "ttnn":
        import ttnn

        dense = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dense = ttnn.to_layout(dense, ttnn.ROW_MAJOR_LAYOUT)
        def bf16(index):
            valid = index.clamp(0, rows - 1).flatten()[None, :].to(torch.int32)
            device_ids = ttnn.from_torch(valid, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            return lambda: ttnn.embedding(device_ids, dense)
    else:
        dense = w.to(device)
        def bf16(index):
            device_ids = index.clamp(0, rows - 1).flatten().to(torch.int32).to(device)
            return lambda: dense.index_select(0, device_ids)

    ids = {
        "one": torch.tensor([min(201023, rows - 1)]),
        "repeat": torch.full((1, 32), min(16, rows - 1), dtype=torch.long),
        "contiguous": torch.arange(32).reshape(2, 16),
        "scattered": torch.tensor([[15, 16, 31, 32, rows - 1, 0, 16]]),
    }
    invalid = torch.tensor([[-1, rows, 2**32 + 16, rows - 1]])
    _same(layer.embedding(invalid).cpu(), _gather_oracle(w, invalid))
    for label, index in ids.items():
        resident = layer.upload_ids(index)
        operation = lambda: layer.gather_device(resident, index.numel())  # noqa: E731
        _, padded_k = descriptor.padded_shape
        result, measured = _measure(operation, sync, repeats, index.numel() * w.shape[1] * 2)
        out = copy(result).reshape(index.numel(), -1)[:, :w.shape[1]].reshape(*index.shape, w.shape[1])
        _same(out, _gather_oracle(w, index))
        dense_result, baseline = _measure(bf16(index), sync, repeats, index.numel() * w.shape[1] * 2)
        # The dense PJRT gather can canonicalize signed zeros. The packed
        # result above is still checked against the literal CPU BF16 bits.
        torch.testing.assert_close(copy(dense_result).reshape_as(out), out, atol=0, rtol=0, equal_nan=True)
        atom_read = {}
        if backend == "ttnn":
            full, atom_read = _measure(lambda: layer.gather_device(resident, index.numel(), full_atoms=True),
                                       sync, repeats, index.numel() * w.shape[1] * 2)
            _same(copy(full)[:, :w.shape[1]].reshape_as(out), out)
        print(json.dumps({
            "op": "embedding", "tensor": name, "case": label, "ids": index.numel(), "shape": list(w.shape),
            "packed_bytes": descriptor.packed_bytes, "resident_bytes": layer.resident_bytes,
            "resident_ids_on_both_paths": True,
            "canonical_bytes_per_uncached_row": (192 if backend == "ttnn" else 2 * 1536) * padded_k // 64,
            "bf16_gather": baseline, "full_atoms": atom_read, **measured,
        }), flush=True)
    if exhaustive:
        for start in range(0, rows, 2048):
            index = torch.arange(start, min(start + 2048, rows))
            _same(layer.embedding(index).cpu(), w[index])
    return descriptor


def _linear(w, name, runtime, backend, repeats, sizes, *, bias=None, exhaustive=False):
    make, device, copy, sync = runtime
    descriptor, packed = prepare_nf12(w, name)
    layer = make(device, descriptor, packed)
    baseline = layer.upload_baseline(w)
    n, k = w.shape
    indices = torch.arange(min(2048, n))
    _, decode = _measure(lambda: layer.embedding(indices), sync, repeats, len(indices) * k * 2)
    _same(layer.embedding(indices).cpu(), w[indices])
    if exhaustive:
        for start in range(len(indices), n, 2048):
            ids = torch.arange(start, min(start + 2048, n))
            _same(layer.embedding(ids).cpu(), w[ids])
    print(json.dumps({"op": "decode", "tensor": name, "rows": len(indices), "shape": [n, k],
                      "packed_bytes": descriptor.packed_bytes, "resident_bytes": layer.resident_bytes,
                      **decode}), flush=True)
    if backend == "ttnn":
        resident_bias = layer.upload_bias(bias) if bias is not None else None
        def call(x, dense, block):
            return layer.linear_device(x, resident_bias, baseline=baseline if dense else None, m_block=block)
    else:
        resident_bias = bias.to(device) if bias is not None else None
        def call(x, dense, block):
            return layer.linear(x, resident_bias, baseline=baseline if dense else None, m_block=block)

    for m in sizes:
        x = torch.randn((m, k)).bfloat16() * 0.03125
        resident = layer.upload_activation(x) if backend == "ttnn" else x.to(device)
        block = min(4, round_up(m, 32) // 32)
        out, result = _measure(lambda: call(resident, False, block), sync, repeats, descriptor.packed_bytes)
        ref, matched = _measure(lambda: call(resident, True, block), sync, repeats, w.nbytes)
        out, ref = copy(out)[:m, :n], copy(ref)[:m, :n]
        _same(out, ref)
        golden = torch.nn.functional.linear(x.float(), w.float(), None if bias is None else bias.float()).bfloat16()
        record = {
            "op": "linear", "tensor": name, "shape": list(w.shape), "m": m, "m_block": block,
            "packed_bytes": descriptor.packed_bytes, "resident_bytes": layer.resident_bytes,
            "packed": result, "bf16": matched, "oracle": _errors(out, golden),
            "fp32_output_scratch_bytes": round_up(m, 32) * round_up(n, 32) * 4 if bias is not None else 0,
            "l1_circular_bytes_per_core": (
                LINEAR_PACKED_SCRATCH_BYTES + 2048 * LINEAR_PACKED_BUFFER_DEPTH * (1 + block)
                + (8192 if bias is not None else 4096)
            ),
        }
        if block > 1:
            _, record["single_m_tile"] = _measure(lambda: call(resident, False, 1), sync, repeats,
                                                 descriptor.packed_bytes)
        print(json.dumps(record), flush=True)
    return descriptor


def _smoke(runtime, backend, repeats):
    w = torch.randn(545, 129).bfloat16()
    # Fill an entire canonical lane group at a nonzero vocabulary slice.
    raw = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32641, -63], dtype=torch.int16)
    special = w.view(torch.int16)
    special[32, [0, 1, 8, 9]] = raw[[0, 1, 4, 5]]
    special[40, [0, 1, 8, 9]] = raw[[2, 3, 6, 7]]
    _embedding(w, "synthetic.special", runtime, backend, repeats, exhaustive=True)
    w = torch.randn(97, 97).bfloat16() * 0.03125
    _linear(w, "synthetic.linear", runtime, backend, repeats, [1, 67, 160])
    _linear(w, "synthetic.bias", runtime, backend, repeats, [4, 128], bias=torch.randn(len(w)).bfloat16() * 0.03125)

    make, device, copy, _ = runtime
    w = torch.zeros((16, 64), dtype=torch.bfloat16)
    w[:, 0] = 1.5
    w[1, 0] = -1.5
    x = torch.zeros((1, 64), dtype=torch.bfloat16)
    x[0, 0] = 1.0078125
    bias = torch.full((16,), -1.5078125, dtype=torch.bfloat16)
    bias[1] = 1.5078125
    layer = make(device, *prepare_nf12(w, "synthetic.rounding"))
    if backend == "ttxla":
        x, bias = x.to(device), bias.to(device)
    unbiased = copy(layer.linear(x)) if backend == "ttxla" else layer.linear(x)
    biased = copy(layer.linear(x, bias)) if backend == "ttxla" else layer.linear(x, bias)
    expected = torch.tensor([1.515625, -1.515625], dtype=torch.bfloat16)
    _same(unbiased[0, :2], expected)
    _same(biased[0, :2], torch.tensor([0.00390625, -0.00390625], dtype=torch.bfloat16))
    print(json.dumps({"op": "rounding", "tie_to_even": True, "bias_before_rounding": True}), flush=True)

    # Equal shapes, identical escape counts, and different exponent bases
    # exercise executable reuse in the Torch/XLA lazy cache.
    variants = [
        (torch.full((32, 64), value, dtype=torch.bfloat16), base)
        for value, base in ((0.03125, 57), (0.015625, 55))
    ]
    x = torch.ones((1, 64), dtype=torch.bfloat16)
    if backend == "ttxla":
        x = x.to(device)
    layers = [make(device, *prepare_nf12(w, "synthetic.exponents", exp_base=base)) for w, base in variants]
    for index in (0, 1, 0):
        w, _ = variants[index]
        layer = layers[index]
        _same(layer.embedding(torch.arange(len(w))).cpu(), w)
        _same(layer.linear(x).cpu(), w.sum(1)[None, :])
    print(json.dumps({"op": "cache_reuse", "distinct_exponent_bases": True}), flush=True)


def _checkpoint(path, runtime, backend, repeats, exhaustive, *, lm_head_only=False, embedding_only=False):
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    embedding = "model.llm.embed.weight"
    lm_head = "model.llm.unembed.weight"
    matrices = [
        (embedding, ()),
        (lm_head, (1, 4, 8, 16, 32, 128, 256)),
        ("model.visual.backbone.embeddings.patch_embedding.weight", (16, 64, 256)),
        ("model.visual.backbone.embeddings.position_embedding", ()),
        ("model.visual.backbone.layers.0.self_attn.q_proj.weight", (18, 65, 257)),
        ("model.visual.backbone.layers.0.mlp.c_fc.weight", (65,)),
        ("model.visual.backbone.layers.0.mlp.c_proj.weight", (65,)),
        ("model.visual.proj_in.weight", (4, 16, 64)),
        ("model.visual.proj_out.weight", (4, 16, 64)),
    ]
    for name, sizes in matrices:
        if (lm_head_only and name != lm_head) or (embedding_only and name != embedding):
            continue
        with safe_open(path / weight_map[name], framework="pt", device="cpu") as shard:
            w = shard.get_tensor(name)
            if w.ndim == 4:
                w = w.reshape(w.shape[0], -1)
            if sizes:
                bias_key = name.removesuffix(".weight") + ".bias"
                bias = shard.get_tensor(bias_key) if bias_key in weight_map else None
                _linear(w, name, runtime, backend, repeats, sizes, bias=bias, exhaustive=exhaustive)
            else:
                _embedding(w, name, runtime, backend, repeats, exhaustive=exhaustive)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("ttnn", "ttxla"))
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--exhaustive", action="store_true", help="Check every logical embedding row on the device")
    selected = parser.add_mutually_exclusive_group()
    selected.add_argument("--lm-head-only", action="store_true", help="Benchmark only the Inkling LM head")
    selected.add_argument("--embedding-only", action="store_true", help="Benchmark only the Inkling token embedding")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if (args.lm_head_only or args.embedding_only) and args.checkpoint is None:
        parser.error("--lm-head-only and --embedding-only require --checkpoint")
    torch.set_num_threads(6)
    torch.manual_seed(12)
    print(json.dumps({"backend": args.backend, "chip": args.chip, "repeats": args.repeats, "torch": torch.__version__,
                      "pjrt_plugin": _package_version("pjrt-plugin-tt")}), flush=True)
    with torch.inference_mode(), _runtime(args.backend, args.chip) as runtime:
        if args.checkpoint is None:
            _smoke(runtime, args.backend, args.repeats)
        else:
            _checkpoint(args.checkpoint, runtime, args.backend, args.repeats, args.exhaustive,
                        lm_head_only=args.lm_head_only, embedding_only=args.embedding_only)


if __name__ == "__main__":
    main()
