"""Experimental GSQ-inspired selection of format-aware circular tile paths.

This is a bounded candidate relaxation, not scalar GSQ: scales, banks and
codebooks stay fixed. Each categorical choice owns an entire circular tile,
so hard export cannot break dependencies between overlapping state windows.
QVQ optionally uses the prepared YAQA Fisher metric before final packing.
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch

from .qvq import (
    decode_p32_window_tiles,
    decode_trellis_tiles,
    pack_trellis_states,
    repack_p32_planar_to_window,
    unpack_p32_window_states,
    unpack_trellis_states,
)
from .qvq_codecs import PGC16_CODEBOOK_VERSION
from .qvq_rates import normalize_qvq_rate


@dataclass
class GSQResult:
    words: torch.Tensor
    choices: torch.Tensor
    calibration_before: float | None
    calibration_after: float | None
    history: list[float]

    @property
    def window_words(self):
        """Compatibility spelling for callers of the P32-only wrapper."""
        return self.words


@dataclass(frozen=True)
class TrellisCandidateAdapter:
    """Legal tile histories in their actual format, never a dense-only edit."""

    layout: str
    bits: float
    codebook_version: str = PGC16_CODEBOOK_VERSION

    def __post_init__(self):
        rate = normalize_qvq_rate(self.bits)
        if not ((self.layout == "p32_window" and rate <= 3.5)
                or (self.layout == "qvq_planar" and 4 <= rate <= 8)):
            raise ValueError("GSQ adapter requires P32 W1-W3.5 or non-banked V2/L16 planar W4-W8")

    def pack(self, states):
        planar = pack_trellis_states(states, bits=self.bits)
        return repack_p32_planar_to_window(planar, bits=self.bits) if self.layout == "p32_window" else planar

    def unpack(self, words):
        return (unpack_p32_window_states(words, bits=self.bits) if self.layout == "p32_window"
                else unpack_trellis_states(words, bits=self.bits))

    def decode(self, words, bank_ids=None, bank_alt_id=None):
        if self.layout == "p32_window":
            if bank_ids is None or bank_alt_id is None:
                raise ValueError("GSQ P32 adapter requires selectors and an alternative bank")
            return decode_p32_window_tiles(words, bits=self.bits, bank_ids=bank_ids, bank_alt_id=bank_alt_id,
                                           codebook_version=self.codebook_version)
        if bank_ids is not None or bank_alt_id is not None:
            raise ValueError("GSQ non-banked adapter cannot accept bank metadata")
        return decode_trellis_tiles(words, bits=self.bits, codebook_version=self.codebook_version).float()

    def inner(self, words, k, n, bank_ids=None, bank_alt_id=None):
        tiles = self.decode(words, bank_ids, bank_alt_id)
        return tiles.reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n).contiguous()


def _candidate_probabilities(logits: torch.Tensor, uniform: torch.Tensor, temperature: float) -> torch.Tensor:
    """GSQ's softmax((kappa * logits + Gumbel(0,1)) / tau), with kappa=1.

    The caller supplies fixed noise for gradient checks and a private RNG for
    fitting. Clamp the uniform endpoints before this function, not the logits.
    """
    gumbel = -(-uniform.log()).log()
    return ((logits + gumbel) / temperature).softmax(-1)


def refine_trellis_candidates(
    candidates: torch.Tensor,
    *,
    bits: float,
    bank_ids: torch.Tensor | None = None,
    bank_alt_id: torch.Tensor | None = None,
    target: torch.Tensor,
    inputs: torch.Tensor,
    right_factor: torch.Tensor | None = None,
    enabled: bool = False,
    layout: str = "p32_window",
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    steps: int = 100,
    learning_rate: float = 0.1,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    seed: int = 0,
    progress: Callable[[int, float], None] | None = None,
) -> GSQResult:
    """Fit tile choices using calibration-only activation reconstruction loss.

    candidates: [choices, K/16*N/16, words], choice zero is the baseline.
    target: [K, N] teacher weights in the same inner coordinate system.
    inputs: [tokens, K] calibration activations in that coordinate system.
    right_factor: optional [N, N] factor of the output sensitivity metric.
    With inputs=L_H.T and right_factor=L_G, the loss is normalized
    tr(G E.T H E), where H=L_H L_H.T, G=L_G L_G.T and E=W-W_teacher.
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
        return GSQResult(
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
    adapter = TrellisCandidateAdapter(layout, bits, codebook_version)
    tensors = [t for t in (target, inputs, bank_ids, bank_alt_id) if t is not None]
    if any(t.device != candidates.device for t in tensors):
        raise ValueError("all tensors must share a device")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (target, inputs)):
        raise ValueError("target and inputs must be finite floating point")
    if right_factor is not None:
        if (right_factor.shape != (n, n) or right_factor.device != candidates.device
                or not right_factor.is_floating_point() or not torch.isfinite(right_factor).all()):
            raise ValueError("right_factor must be finite floating point [N, N] on the candidate device")
        right_factor = right_factor.detach().float()
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if any(not math.isfinite(v) or v <= 0 for v in (learning_rate, temperature_start, temperature_end)):
        raise ValueError("learning rate and temperatures must be finite and positive")
    decoded = torch.stack([
        adapter.decode(c, bank_ids, bank_alt_id) for c in candidates
    ]).detach().transpose(0, 1).contiguous()  # [tile, choice, 256]
    x = inputs.detach().float()
    teacher = x @ target.detach().float()
    if right_factor is not None:
        teacher = teacher @ right_factor
    normalizer = teacher.square().mean().clamp_min(torch.finfo(torch.float32).tiny)

    def loss(tiles):
        weight = tiles.reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)
        prediction = x @ weight
        if right_factor is not None:
            prediction = prediction @ right_factor
        return ((prediction - teacher).square().mean() / normalizer)

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
    return GSQResult(words, best_choices, before, best, history)


def baseline_bitflip_candidates(baseline, *, count, seed):
    """Frozen, reproducible pool shared by stochastic and deterministic fits.

    Each non-baseline entry flips exactly one payload bit per tile; duplicates
    are retained to preserve the historical experiment's distribution.
    """
    if baseline.ndim != 2 or baseline.dtype != torch.int32 or min(baseline.shape) <= 0:
        raise ValueError("GSQ baseline must be nonempty int32 [tiles,words]")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("GSQ candidate count must be an integer >= 2")
    candidates = baseline.detach().clone().unsqueeze(0).repeat(count, 1, 1)
    generator = torch.Generator(device=baseline.device).manual_seed(seed)
    tiles = torch.arange(len(baseline), device=baseline.device)
    for candidate in range(1, count):
        bit = torch.randint(baseline.shape[1] * 32, (len(baseline),),
                            device=baseline.device, generator=generator)
        candidates[candidate, tiles, bit // 32] ^= (torch.ones_like(bit) << (bit % 32)).to(torch.int32)
    return candidates


def refine_trellis_fisher(baseline, *, target, input_hessian, output_hessian, config,
                          bits, bank_ids=None, bank_alt_id=None, codebook_version=PGC16_CODEBOOK_VERSION,
                          layout="p32_window"):
    """Build legal candidates and fit the existing damped YAQA quadratic.

    Hessians and target must already share the normalized trellis inner basis.
    Cholesky uses the prepared metric as-is: no hidden extra regularization.
    The config's memory limit describes decoded candidates, not peak memory.
    """
    from .config import normalize_gsq_config

    config = normalize_gsq_config(config)
    if config is None or not config.enabled:
        raise ValueError("refine_trellis_fisher requires enabled GSQConfig")
    if config.learn_scales:
        raise ValueError("QVQ GSQ does not support scale learning")
    required = config.candidates * target.numel() * 4
    if required > config.max_candidate_bytes:
        raise ValueError(f"GSQ decoded candidates need {required} bytes, exceeding max_candidate_bytes="
                         f"{config.max_candidate_bytes}; reduce candidates or select smaller modules")
    # Quantization may be invoked from an inference-mode lifecycle. Clone the
    # constants outside it so autograd can save them for logit gradients.
    with torch.inference_mode(False), torch.enable_grad():
        target = target.detach().float().clone()
        left = torch.linalg.cholesky(input_hessian.detach().float().clone()).T.contiguous()
        right = torch.linalg.cholesky(output_hessian.detach().float().clone())
        candidates = baseline_bitflip_candidates(baseline, count=config.candidates, seed=config.seed)
        return refine_trellis_candidates(
            candidates, bits=bits, bank_ids=None if bank_ids is None else bank_ids.detach().clone(),
            bank_alt_id=None if bank_alt_id is None else bank_alt_id.detach().clone(), layout=layout,
            target=target, inputs=left, right_factor=right, enabled=True, codebook_version=codebook_version,
            steps=config.steps, seed=config.seed, learning_rate=config.learning_rate,
            temperature_start=config.temperature_start, temperature_end=config.temperature_end)


def refine_p32_candidates(candidates, **kwargs) -> GSQResult:
    """Backward-compatible P32 window entry point; never selects another layout."""
    return refine_trellis_candidates(candidates, layout="p32_window", **kwargs)


def refine_p32_fisher(baseline, **kwargs) -> GSQResult:
    """Backward-compatible P32 Fisher entry point."""
    return refine_trellis_fisher(baseline, layout="p32_window", **kwargs)


@torch.no_grad()
def deterministic_trellis_candidates(candidates, *, target, inputs, right_factor,
                                      bits, layout, bank_ids=None, bank_alt_id=None,
                                      codebook_version=PGC16_CODEBOOK_VERSION, sweeps=3):
    """Sequential hard coordinate search on the same frozen GSQ candidate pool.

    For a tile update D, the unnormalized Fisher loss change is
    2 <(H E G)[tile], D> + <H_ii D G_jj, D>. Maintain H E G after
    every accepted tile, so dense output-block coupling is retained.
    Full objective recomputation guards each sweep against accumulated drift.
    This is a deterministic comparator, not a globally optimal search.
    """
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 1:
        raise ValueError("deterministic GSQ comparator requires positive integer sweeps")
    adapter = TrellisCandidateAdapter(layout, bits, codebook_version)
    if target.ndim != 2 or target.shape[0] % 16 or target.shape[1] % 16:
        raise ValueError("deterministic GSQ target must have tile-aligned [K,N] shape")
    k, n = target.shape
    if candidates.ndim != 3 or candidates.shape[1] != k * n // 256 or candidates.dtype != torch.int32:
        raise ValueError("deterministic GSQ candidates must be int32 [choices,tiles,words]")
    if inputs.ndim != 2 or inputs.shape[1] != k or right_factor.shape != (n, n):
        raise ValueError("deterministic GSQ factors must match target dimensions")
    if any(t.device != candidates.device or not t.is_floating_point() or not torch.isfinite(t).all()
           for t in (target, inputs, right_factor)):
        raise ValueError("deterministic GSQ target/factors must be finite floating tensors on the candidate device")
    x, right, teacher = inputs.float(), right_factor.float(), target.float()
    h, g = x.T @ x, right @ right.T
    values = torch.stack([adapter.decode(c, bank_ids, bank_alt_id) for c in candidates]).reshape(
        candidates.shape[0], candidates.shape[1], 16, 16)
    current = adapter.inner(candidates[0], k, n, bank_ids, bank_alt_id)
    choices = torch.zeros(candidates.shape[1], device=candidates.device, dtype=torch.long)
    normalizer = (x @ teacher @ right).square().mean().clamp_min(torch.finfo(torch.float32).tiny)

    def score(weight):
        return float((x @ (weight - teacher) @ right).square().mean() / normalizer)

    before = score(current)
    if not math.isfinite(before):
        raise ValueError("non-finite deterministic GSQ baseline objective")
    best, best_choices, history = before, choices.clone(), [before]
    for _ in range(sweeps):
        metric_error = h @ (current - teacher) @ g
        for tile in range(candidates.shape[1]):
            ib, jb = divmod(tile, n // 16)
            i, j = slice(ib * 16, (ib + 1) * 16), slice(jb * 16, (jb + 1) * 16)
            delta = values[:, tile] - current[i, j]
            cost = 2 * (delta * metric_error[i, j]).sum((1, 2))
            cost += (torch.matmul(torch.matmul(h[i, i], delta), g[j, j]) * delta).sum((1, 2))
            selected = int(cost.argmin())
            if float(cost[selected]) < 0:
                update = delta[selected]
                current[i, j] = values[selected, tile]
                choices[tile] = selected
                metric_error += h[:, i] @ update @ g[j, :]
        value = score(current)
        if not math.isfinite(value):
            raise ValueError("non-finite deterministic GSQ hard objective")
        history.append(value)
        if value < best:
            best, best_choices = value, choices.clone()
    tile_ids = torch.arange(candidates.shape[1], device=candidates.device)
    return GSQResult(candidates[best_choices, tile_ids].clone(), best_choices, before, best, history)
