"""Bounded adjacent-router-rank recovery, independent of natural forward.

Keep rank-major then corpus-row order, matching the fork's established policy.
Every retained row is owned CPU storage; live routing is verified, never changed.
"""
import hashlib
import json

import torch
from torch.nn import functional as F

from .exl3_router_candidates import learned_router_ranked_choices


class V41Recovery:
    def __init__(self, block, expert_ids, *, target_count=1024):
        self.block = block
        self.expert_ids = tuple(expert_ids)
        self.router = block.mlp.gate
        self.top_k = self.router.top_k
        self.rank_max = min(2 * self.top_k, len(block.mlp.experts))
        if (type(target_count) is not int or target_count < 1 or not self.expert_ids
                or len(set(self.expert_ids)) != len(self.expert_ids)
                or any(type(index) is not int or not 0 <= index < len(block.mlp.experts) for index in self.expert_ids)):
            raise ValueError("invalid recovery target/count")
        self.target_count = target_count
        self.rows = {(expert, rank): [] for expert in self.expert_ids
                     for rank in range(self.top_k + 1, self.rank_max + 1)}
        self.coordinates = {key: [] for key in self.rows}
        self.observed = dict.fromkeys(self.expert_ids, 0)
        self.offset = 0
        self.handle = None
        self.failed = False

    @torch.no_grad()
    def _observe(self, module, args, output):
        logits, _, selected = output
        _, ranked = learned_router_ranked_choices(self.router, logits, rank_max=self.rank_max,
                                                   selected_indices=selected)
        inputs = args[0].detach().reshape(-1, args[0].shape[-1])
        for expert in self.expert_ids:
            for rank in range(self.top_k + 1, self.rank_max + 1):
                indices = torch.where(ranked[:, rank - 1] == expert)[0]
                self.observed[expert] += indices.numel()
                key = expert, rank
                remaining = self.target_count - len(self.coordinates[key])
                if remaining > 0 and indices.numel():
                    indices = indices[:remaining]
                    self.rows[key].append(inputs[indices.to(inputs.device)].to("cpu", copy=True))
                    self.coordinates[key].extend((indices.cpu() + self.offset).tolist())
        self.offset += inputs.shape[0]

    def observe_routed(self, batch):
        """Reuse the captured router result without executing attention or MoE."""
        if self.handle is not None or self.failed:
            raise RuntimeError("direct recovery observation must be detached and healthy")
        device = self.router.e_score_correction_bias.device
        try:
            self._observe(self.router, (batch.hidden,),
                          (batch.logits.to(device), batch.weights.to(device), batch.indices.to(device)))
        except BaseException:
            self.failed = True
            raise

    def __enter__(self):
        if self.handle is not None or self.failed:
            raise RuntimeError("recovery is active or failed")
        self.handle = self.router.register_forward_hook(self._observe)
        return self

    def __exit__(self, kind, value, traceback):
        self.failed |= kind is not None
        self.handle.remove()
        self.handle = None

    @torch.no_grad()
    def projection(self, capture, expert, projection):
        """Top up an owned raw Hessian; never modify natural evidence or capture.

        For down, caller must already have installed the selected gate/up tier.
        Recovery-only down inputs use unit routing weight, as in the established
        direct-expert augmentation policy, with V4.1's FP32 SwiGLU then BF16 store.
        Raw identity adds missing*I; fork normalization's 2/count convention is
        equivalent to normalized 2I. EXL3 receives raw sums and divides by count.
        """
        if self.handle is not None or self.failed:
            raise RuntimeError("recovery must complete before exporting")
        if capture.block is not self.block or int(capture.route_counts.sum()) != self.offset * self.top_k:
            raise ValueError("recovery and natural capture cover different blocks or row counts")
        hessian, natural = capture.projection(expert, projection)
        need = max(0, self.target_count - hessian["count"])
        selected_rows, coords, histogram = [], [], {}
        for rank in range(self.top_k + 1, self.rank_max + 1):
            key = expert, rank
            count = min(need, len(self.coordinates[key]))
            histogram[str(rank)] = count
            if count:
                selected_rows.append(torch.cat(self.rows[key])[:count])
                coords.extend((rank, coordinate) for coordinate in self.coordinates[key][:count])
                need -= count
        augmented = len(coords)
        target = self.block.mlp.experts[expert]
        device = capture.device
        for inputs in selected_rows:
            for start in range(0, inputs.shape[0], capture.chunk_rows):
                rows = inputs[start:start + capture.chunk_rows].to(device)
                if projection == "w2":
                    gate, up = target.gate_proj(rows).float(), target.up_proj(rows).float()
                    if target.limit > 0:
                        gate = gate.clamp(max=target.limit)
                        up = up.clamp(min=-target.limit, max=target.limit)
                    rows = (F.silu(gate) * up).to(rows.dtype)
                rows = rows.float()
                if not torch.isfinite(rows).all():
                    raise ValueError("nonfinite recovery activation")
                hessian["H"].add_((rows.T @ rows).cpu())
        hessian["H"].diagonal().add_(float(need))
        hessian["count"] += augmented + need
        evidence = {**natural, "augmented_rows": augmented, "identity_rows": need,
                    "effective_rows": hessian["count"], "candidate_rows_observed": self.observed[expert],
                    "candidate_rank_histogram": histogram,
                    "selection_sha256": hashlib.sha256(json.dumps(coords, separators=(",", ":")).encode()).hexdigest(),
                    "selection_policy": "rank-major-then-corpus-row-v1",
                    "recovery_down_route_weight": "unit"}
        return hessian, evidence
