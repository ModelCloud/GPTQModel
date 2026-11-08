"""Reusable owned FFN input frontier for efficient expert-subset calibration."""
from dataclasses import dataclass

import torch

from .v41_replay import owned_tree


@dataclass
class V41RoutedBatch:
    hidden: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor

    def validate(self):
        if any(not isinstance(value, torch.Tensor) or value.ndim != 2
               for value in (self.hidden, self.logits, self.weights, self.indices)):
            raise ValueError("routed frontier tensors must be matrices")
        rows = self.hidden.shape[0]
        if (self.logits.shape[0] != rows or self.indices.shape[0] != rows
                or self.weights.shape != self.indices.shape or self.indices.dtype != torch.int64
                or not 0 < self.indices.shape[1] <= self.logits.shape[1]):
            raise ValueError("invalid routed frontier geometry")
        if any(not value.is_floating_point() or not torch.isfinite(value).all()
               for value in (self.hidden, self.logits, self.weights)):
            raise ValueError("routed frontier values must be finite floating-point tensors")
        if rows and (self.indices.min() < 0 or self.indices.max() >= self.logits.shape[1]):
            raise ValueError("routed frontier expert index is out of range")
        ordered = self.indices.sort(dim=-1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError("routed frontier contains duplicate expert assignments")

    @classmethod
    @torch.no_grad()
    def from_replay(cls, block, state, device):
        """Run attention once, stop just after routing and before expert GEMMs.

        Every expert subset reuses these identical normalized FFN inputs and
        natural routes. This is not the post-quantization propagation frontier.
        """
        captured = []

        class Captured(Exception):
            pass

        def hook(module, args, output):
            logits, weights, indices = output
            captured.append(cls(owned_tree(args[0].reshape(-1, args[0].shape[-1]), "cpu"),
                                owned_tree(logits, "cpu"), owned_tree(weights, "cpu"),
                                owned_tree(indices, "cpu")))
            raise Captured()

        handle = block.mlp.gate.register_forward_hook(hook)
        try:
            try:
                state.advance(block, device)
            except Captured:
                pass
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("V4.1 routed frontier was not captured exactly once")
        return captured[0]
