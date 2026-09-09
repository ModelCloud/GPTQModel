"""Experimental GSQ-inspired selection of valid P32 circular tile paths.

This is a bounded candidate relaxation, not scalar GSQ: scales, banks and
codebooks stay fixed. Each categorical choice owns an entire circular tile,
so hard export cannot break dependencies between overlapping state windows.
No production quantization dispatch uses this helper.
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch

from .qvq import decode_p32_window_tiles
from .qvq_codecs import PGC16_CODEBOOK_VERSION


@dataclass
class P32GSQResult:
    window_words: torch.Tensor
    choices: torch.Tensor
    calibration_before: float | None
    calibration_after: float | None
    history: list[float]


def _candidate_probabilities(logits: torch.Tensor, uniform: torch.Tensor, temperature: float) -> torch.Tensor:
    """GSQ's softmax((kappa * logits + Gumbel(0,1)) / tau), with kappa=1.

    The caller supplies fixed noise for gradient checks and a private RNG for
    fitting. Clamp the uniform endpoints before this function, not the logits.
    """
    gumbel = -(-uniform.log()).log()
    return ((logits + gumbel) / temperature).softmax(-1)


def refine_p32_candidates(
    candidates: torch.Tensor,
    *,
    bits: float,
    bank_ids: torch.Tensor,
    bank_alt_id: torch.Tensor,
    target: torch.Tensor,
    inputs: torch.Tensor,
    enabled: bool = False,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    steps: int = 100,
    learning_rate: float = 0.1,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    seed: int = 0,
    progress: Callable[[int, float], None] | None = None,
) -> P32GSQResult:
    """Fit tile choices using calibration-only activation reconstruction loss.

    candidates: [choices, K/16*N/16, words], choice zero is the baseline.
    target: [K, N] teacher weights in the same inner coordinate system.
    inputs: [tokens, K] calibration activations in that coordinate system.
    Callers must evaluate the hard returned payload on independent data.
    The best *hard* calibration checkpoint (including baseline) is returned.
    Disabled by default: returns an independent, byte-identical baseline with
    no decoding, calibration reads, optimizer, or random sampling. Losses are
    None and history is empty because no objective was evaluated.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be boolean")
    if candidates.ndim != 3 or candidates.dtype != torch.int32 or candidates.shape[0] < 2:
        raise ValueError("candidates must be int32 [choices>=2, tiles, words]")
    if not enabled:
        return P32GSQResult(
            candidates[0].detach().clone().contiguous(),
            torch.zeros(candidates.shape[1], device=candidates.device, dtype=torch.int64),
            None, None, [],
        )
    if target.ndim != 2 or min(target.shape) < 16 or any(d % 16 for d in target.shape):
        raise ValueError("target must be [K, N] with positive dimensions divisible by 16")
    k, n = target.shape
    if candidates.shape[1] != k * n // 256:
        raise ValueError("candidate tile count does not match target")
    if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] != k:
        raise ValueError("inputs must be nonempty [tokens, K]")
    tensors = (target, inputs, bank_ids, bank_alt_id)
    if any(t.device != candidates.device for t in tensors):
        raise ValueError("all tensors must share a device")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (target, inputs)):
        raise ValueError("target and inputs must be finite floating point")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if any(not math.isfinite(v) or v <= 0 for v in (learning_rate, temperature_start, temperature_end)):
        raise ValueError("learning rate and temperatures must be finite and positive")
    decoded = torch.stack([
        decode_p32_window_tiles(
            c, bits=bits, bank_ids=bank_ids, bank_alt_id=bank_alt_id, codebook_version=codebook_version
        ) for c in candidates
    ]).detach().transpose(0, 1).contiguous()  # [tile, choice, 256]
    x = inputs.detach().float()
    teacher = x @ target.detach().float()
    normalizer = teacher.square().mean().clamp_min(torch.finfo(torch.float32).tiny)

    def loss(tiles):
        weight = tiles.reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)
        return ((x @ weight - teacher).square().mean() / normalizer)

    tile_ids = torch.arange(candidates.shape[1], device=candidates.device)
    best_choices = torch.zeros_like(tile_ids)
    before = float(loss(decoded[:, 0]))
    if not math.isfinite(before):
        raise ValueError("non-finite calibration objective")
    best = before
    history = [before]
    logits = torch.zeros(decoded.shape[:2], device=candidates.device, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 2.0
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    generator = torch.Generator(device=candidates.device).manual_seed(seed)
    with torch.enable_grad():
        for step in range(steps):
            tau = temperature_start * (temperature_end / temperature_start) ** (step / max(steps - 1, 1))
            uniform = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-6, 1 - 1e-6)
            probabilities = _candidate_probabilities(logits, uniform, tau)
            objective = loss((probabilities.unsqueeze(-1) * decoded).sum(1))
            if not torch.isfinite(objective):
                raise ValueError("non-finite relaxed objective")
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                choices = logits.argmax(-1)
                hard_loss = float(loss(decoded[tile_ids, choices]))
                if not math.isfinite(hard_loss):
                    raise ValueError("non-finite hard objective")
                history.append(hard_loss)
                if hard_loss < best:
                    best, best_choices = hard_loss, choices.clone()
                if progress is not None:
                    progress(step + 1, best)
    words = candidates[best_choices, tile_ids].detach().clone().contiguous()
    return P32GSQResult(words, best_choices, before, best, history)
