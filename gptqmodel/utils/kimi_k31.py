# SPDX-License-Identifier: Apache-2.0
"""Bounded K3.1 source reads and accurate causal calibration accumulators.

The decoder and DSA implementation come from the caller's pinned fidelity
reference. This adapter owns neither an alternative attention implementation
nor a new quantizer: K3's ordinary MCG serialization/reconstruction is reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from .exl3_hessian import HESSIAN_ACCUMULATION_CONTRACT, BoundedHessianAccumulator
from .kimi_k3 import KimiK3Source, situ
from .kimi_k3_core_format import (
    KimiK3EXL3Source,
)


REVISION = "6d0a0202956796543364d666ab19c6ad1a494ee1"


def _indexer_f32_inputs(module, inputs):
    """Keep the shared reference's index projection/norm arithmetic in F32."""
    return tuple(value.float() if isinstance(value, torch.Tensor) else value for value in inputs)


class KimiK31Source(KimiK3EXL3Source):
    """Official MXFP4 source with one active shard and deferred BF16 experts."""
    def __init__(self, snapshot, *, reference_root):
        self.snapshot = Path(snapshot).resolve()
        self.config = json.loads((self.snapshot / "config.json").read_text())
        if (self.snapshot.name != REVISION or self.config.get("model_type") != "kimi_k31"
                or self.config.get("architectures") != ["KimiK31ForConditionalGeneration"]):
            raise ValueError("expected the pinned official Kimi K3.1 Pro checkpoint")
        self.text_config = self.config["text_config"]
        self.weight_map = json.loads((self.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        for shard in set(self.weight_map.values()):
            if Path(shard).name != shard or not (self.snapshot / shard).is_file():
                raise ValueError(f"invalid/missing K3.1 shard: {shard}")
        root = str(Path(reference_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from families.kimi_k3.model import load_official, source_identity
        self.reference, self.reference_config, quantization = load_official(self.snapshot)
        self.reference_identity = source_identity(self.snapshot)
        if quantization.get("format") != "mxfp4-pack-quantized":
            raise ValueError("expected checkpoint-native MXFP4 routed experts")
        self.reference_config._attn_implementation = "eager"
        self.reference_config.use_cache = False

    def decoded(self, name, device="cpu"):
        # The inherited EXL3 reader's decoder expects already-quantized source
        # parts. This source instead uses the original exact MXFP4 dequantizer.
        value = KimiK3Source.decoded(self, name, device)
        if name.endswith(".self_attn.A_log"):
            heads = self.text_config["linear_attn_config"]["num_heads"]
            if value.ndim != 1 or value.numel() != heads:
                if (heads != 96 or value.dtype != torch.float32 or value.shape != (128,)
                        or bool(torch.count_nonzero(value[96:].contiguous().view(torch.uint8)))):
                    raise ValueError("unsupported or nonzero KDA A_log padding")
        return value

    @torch.no_grad()
    def load_block(self, layer, device="cuda:0", *, include_experts=False):
        if type(layer) is not int or not 0 <= layer < self.text_config["num_hidden_layers"]:
            raise ValueError("layer outside K3.1 architecture")
        config = copy.deepcopy(self.reference_config)
        with torch.device("meta"):
            block = self.reference.KimiDecoderLayer(config, layer)
        self._materialize(block, layer, device, experts_only=False, include_experts=include_experts)
        indexer = getattr(block.self_attn, "indexer", None)
        if indexer is not None:
            # Fidelity runs all weights/inputs in F32; quant calibration keeps
            # native BF16 main projections but must retain F32 index math too.
            # These casts are transient execution storage, never export bytes.
            indexer.float()
            indexer.register_forward_pre_hook(_indexer_f32_inputs)
        return block.eval()


class DenseRawHessian(BoundedHessianAccumulator):
    """Actual observations only, bounded FP32 products and FP64 outer sum."""
    def __init__(self, width, *, device="cuda:0", chunk_rows=128):
        super().__init__(width, device=device, max_rows=chunk_rows)

    @torch.inference_mode()
    def raw_dict(self, *, minimum_rows=1):
        if type(minimum_rows) is not int or minimum_rows <= 0:
            raise ValueError("minimum observation count must be a positive integer")
        if self.count < minimum_rows:
            raise ValueError(f"insufficient dense Hessian observations: {self.count} < {minimum_rows}")
        matrix = self.snapshot()
        if float(matrix.diagonal().sum()) <= 0:
            raise ValueError("invalid dense Hessian covariance")
        return {"H": matrix, "count": self.count, "finalized": False}

    as_dict = raw_dict


@torch.inference_mode()
def capture_routed_hessian(spool, expert, *, projections=None, chunk_rows=2048, device="cuda:0"):
    """Preserve K3 route/recovery order and floor with accurate accumulation.

    The spool supplies the exact native routes followed by the original
    rank-major cold-expert recovery rows. Down inputs use serialized/reloaded
    gate and up weights passed by the caller, with native BF16 SiTU rounding.
    """
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("gather chunk size must be positive")
    down = projections is not None
    width = int(projections["w1"].shape[0] if down else spool.inputs.shape[-1])
    accumulator = BoundedHessianAccumulator(width, device=device)
    natural, recovery = spool.route_rows[expert], spool.recovery_rows[expert]
    for selection in (natural, recovery):
        for start in range(0, len(selection), chunk_rows):
            rows = spool.gather(selection[start:start + chunk_rows], device=device)
            if down:
                rows = situ(F.linear(rows, projections["w1"]),
                            F.linear(rows, projections["w3"])).to(torch.bfloat16)
            accumulator.add(rows)
    residual = max(0, 1024 - len(natural) - len(recovery))
    matrix = accumulator.snapshot()
    matrix.diagonal().add_(float(residual))
    count = len(natural) + len(recovery) + residual
    if not bool(torch.isfinite(matrix).all()) or float(matrix.diagonal().sum()) <= 0:
        raise ValueError("invalid captured routed Hessian")
    evidence = {"natural_rows": len(natural), "augmented_rows": len(recovery), "identity_rows": residual,
                    "effective_rows": count, "candidate_selection_sha256": hashlib.sha256(recovery.tobytes()).hexdigest(),
                    "ranks": [17, 32], "selection_policy": "rank-major-then-corpus-row-v1",
                    "down_inputs": "serialized-quantized-gate-up-situ-bf16" if down else None,
                    "hessian_accumulation": HESSIAN_ACCUMULATION_CONTRACT}
    return {"H": matrix, "count": count, "finalized": False}, evidence
