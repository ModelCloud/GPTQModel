"""Experimental GSQ-inspired selection of format-aware circular tile paths.

This is a bounded candidate relaxation, not scalar GSQ: scales, banks and
codebooks stay fixed. Each categorical choice owns an entire circular tile,
so hard export cannot break dependencies between overlapping state windows.
QVQ optionally uses the prepared YAQA Fisher metric before final packing.
"""

import math
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from ..utils.hadamard import (
    hadamard_transform,
    hadamard_transform_sandwich,
    hadamard_transform_scaled,
)
from .qvq import (
    QVQ_V2B2_P32_SEGMENTS_PER_TILE,
    QVQ_V2B2_P32_STEPS_PER_SEGMENT,
    decode_p32_window_tiles,
    decode_trellis_tiles,
    pack_trellis_states,
    planar_pack_rows,
    repack_p32_planar_to_window,
    unpack_p32_window_states,
    unpack_qvq_binary_bank_ids,
    unpack_trellis_states,
)
from .qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    pgc16_decode_states_v2_banked,
    pgc16_levels_for_version,
)
from .qvq_rates import normalize_qvq_rate


_CUDA_GRAPH_CAPTURE_LOCK = threading.Lock()


def _right_fisher_hadamard(values, diagonal, scale):
    if (values.dtype == torch.bfloat16 and diagonal.dtype == torch.bfloat16
            and values.shape[-1] in (512, 2048)):
        return hadamard_transform_sandwich(values.contiguous(), diagonal, scale)
    intermediate = hadamard_transform_scaled(values.contiguous(), diagonal, scale)
    return hadamard_transform(intermediate.contiguous(), scale)


def _nvtx_range(name: str, tensor: torch.Tensor):
    return torch.cuda.nvtx.range(name) if tensor.device.type == "cuda" else nullcontext()


@dataclass
class GSQResult:
    words: torch.Tensor
    choices: torch.Tensor
    calibration_before: float | None
    calibration_after: float | None
    history: list[float]
    diagnostics: dict = field(default_factory=dict)

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


def _candidate_probabilities(logits: torch.Tensor, uniform: torch.Tensor, temperature: float,
                             kappa: float = 1.0) -> torch.Tensor:
    """GSQ's softmax((kappa * logits + Gumbel(0,1)) / tau).

    The caller supplies fixed noise for gradient checks and a private RNG for
    fitting. Clamp the uniform endpoints before this function, not the logits.
    """
    gumbel = -(-uniform.log()).log()
    return ((kappa * logits + gumbel) / temperature).softmax(-1)


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
    learning_rate: float = 1e-4,
    temperature_start: float = 2.0,
    temperature_end: float = 0.05,
    kappa_start: float = 100.0,
    kappa_end: float = 500.0,
    weight_decay: float = 1.0,
    initialization_std: float = 0.01,
    initialization_strength: float = 6.0,
    gumbel_samples: int = 1,
    soft_dtype: str = "float32",
    coordinate_sweeps: int = 1,
    coordinate_chunk_tiles: int = 1024,
    hard_eval_interval: int = 10,
    relaxation_patience: int = 10,
    decoded_candidates: torch.Tensor | None = None,
    decoded_baseline: torch.Tensor | None = None,
    sparse_candidate_indices: torch.Tensor | None = None,
    sparse_candidate_deltas: torch.Tensor | None = None,
    sparse_candidate_values: torch.Tensor | None = None,
    sparse_candidate_shifts: torch.Tensor | None = None,
    input_metric: torch.Tensor | None = None,
    output_metric: torch.Tensor | None = None,
    output_metric_hadamard_diagonal: torch.Tensor | None = None,
    hard_dense_verify_topk: int = 4,
    cuda_graph_updates_per_replay: int = 1,
    fast_position_map: bool = True,
    capture_barrier: threading.Barrier | None = None,
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
    if isinstance(hard_dense_verify_topk, bool) or not isinstance(hard_dense_verify_topk, int):
        raise TypeError("hard_dense_verify_topk must be an integer")
    if hard_dense_verify_topk < 0:
        raise ValueError("hard_dense_verify_topk must be nonnegative")
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
    if any(not math.isfinite(v) or v <= 0 for v in
           (learning_rate, temperature_start, temperature_end, kappa_start, kappa_end)):
        raise ValueError("learning rate, temperatures and kappa values must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(initialization_std) or initialization_std <= 0:
        raise ValueError("initialization_std must be finite and positive")
    if not math.isfinite(initialization_strength) or initialization_strength < 0:
        raise ValueError("initialization_strength must be finite and nonnegative")
    if soft_dtype not in ("float32", "bfloat16"):
        raise ValueError("soft_dtype must be 'float32' or 'bfloat16'")
    for name, value, minimum in (("gumbel_samples", gumbel_samples, 1),
                                 ("coordinate_sweeps", coordinate_sweeps, 0),
                                 ("coordinate_chunk_tiles", coordinate_chunk_tiles, 1),
                                 ("hard_eval_interval", hard_eval_interval, 1),
                                 ("cuda_graph_updates_per_replay", cuda_graph_updates_per_replay, 1),
                                 ("relaxation_patience", relaxation_patience, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    sparse_relaxation = sparse_candidate_indices is not None or sparse_candidate_deltas is not None
    sparse_baseline_only = bool(
        sparse_relaxation
        and decoded_baseline is not None
        and sparse_candidate_values is not None
        and input_metric is not None
        and output_metric is not None
        and coordinate_sweeps == 0
    )
    with _nvtx_range("gsq.decode_candidate_bank", candidates):
        if sparse_baseline_only:
            expected = (candidates.shape[1], 256)
            if (decoded_baseline.numel() != candidates.shape[1] * 256
                    or decoded_baseline.device != candidates.device):
                raise ValueError(f"decoded_baseline must contain {expected} values on the candidate device")
            baseline_tiles = decoded_baseline.detach().reshape(expected).contiguous()
            decoded = None
        elif decoded_candidates is None:
            decoded = torch.stack([
                adapter.decode(c, bank_ids, bank_alt_id) for c in candidates
            ]).detach().transpose(0, 1).contiguous()  # [tile, choice, 256]
            baseline_tiles = decoded[:, 0]
        else:
            expected = (candidates.shape[0], candidates.shape[1], 16, 16)
            if decoded_candidates.shape != expected or decoded_candidates.device != candidates.device:
                raise ValueError(f"decoded_candidates must have shape {expected} on the candidate device")
            decoded = decoded_candidates.detach().reshape(*expected[:2], 256).transpose(0, 1).contiguous()
            baseline_tiles = decoded[:, 0]
    if sparse_relaxation:
        expected_prefix = (candidates.shape[0] - 1, candidates.shape[1])
        if (sparse_candidate_indices is None or sparse_candidate_deltas is None
                or sparse_candidate_indices.shape != sparse_candidate_deltas.shape
                or sparse_candidate_indices.shape[:2] != expected_prefix
                or sparse_candidate_indices.device != candidates.device
                or sparse_candidate_deltas.device != candidates.device
                or sparse_candidate_indices.dtype != torch.int64
                or sparse_candidate_indices.ndim != 3):
            raise ValueError(
                "sparse candidate indices/deltas must match int64/float [choices-1,tiles,width] tensors"
            )
        if not sparse_candidate_deltas.is_floating_point():
            raise ValueError("sparse candidate deltas must be floating point")
        if (sparse_candidate_values is not None
                and (sparse_candidate_values.shape != sparse_candidate_indices.shape
                     or sparse_candidate_values.device != candidates.device
                     or not sparse_candidate_values.is_floating_point())):
            raise ValueError(
                "sparse candidate values must match floating-point sparse candidate metadata"
            )
        if (sparse_candidate_shifts is not None
                and (sparse_candidate_shifts.shape != expected_prefix
                     or sparse_candidate_shifts.device != candidates.device
                     or sparse_candidate_shifts.dtype != torch.int64)):
            raise ValueError("sparse candidate shifts must match int64 [choices-1,tiles]")
        if bool(((sparse_candidate_indices < 0) | (sparse_candidate_indices >= 256)).any()):
            raise ValueError("sparse candidate indices must be in [0,256)")
        sparse_indices_hard_by_tile = sparse_candidate_indices.permute(1, 0, 2).contiguous()
        sparse_values_hard_by_tile = (
            None if sparse_candidate_values is None else
            sparse_candidate_values.permute(1, 0, 2).contiguous()
        )
    else:
        sparse_indices_hard_by_tile = sparse_values_hard_by_tile = None
    fisher_objective = input_metric is not None and output_metric is not None
    x = inputs.detach().float()
    if fisher_objective:
        h_metric = input_metric.detach().float()
        g_metric = output_metric.detach().float()
        if h_metric.shape != (k, k) or g_metric.shape != (n, n):
            raise ValueError("input_metric and output_metric must match the Fisher objective dimensions")
        if output_metric_hadamard_diagonal is not None:
            if (output_metric_hadamard_diagonal.shape != (n,)
                    or output_metric_hadamard_diagonal.device != candidates.device
                    or not output_metric_hadamard_diagonal.is_floating_point()
                    or not torch.isfinite(output_metric_hadamard_diagonal).all()
                    or n < 8 or n > 32768 or n & (n - 1)):
                raise ValueError(
                    "structured output Fisher requires a finite power-of-two Hadamard diagonal"
                )
        target_metric = h_metric @ target.detach().float() @ g_metric
        fisher_denominator = (target.detach().float() * target_metric).sum().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        relaxation_dtype = (
            torch.bfloat16 if soft_dtype == "bfloat16" and candidates.device.type == "cuda"
            else torch.float32
        )
        h_relax = h_metric.to(relaxation_dtype)
        g_relax = g_metric.to(relaxation_dtype)
        g_hadamard_diagonal = (
            output_metric_hadamard_diagonal.detach().to(relaxation_dtype)
            if output_metric_hadamard_diagonal is not None else None
        )
        g_hadamard_diagonal_hard = (
            output_metric_hadamard_diagonal.detach().float()
            if output_metric_hadamard_diagonal is not None else None
        )
        target_relax = target.detach().to(relaxation_dtype)
        if sparse_relaxation:
            # Candidate zero plus a tiny per-choice delta is sufficient for
            # P32 local-path relaxation. W3 has six changed values per choice,
            # versus reading all 256 values in the dense candidate bank.
            decoded_relax = None
            baseline_relax = baseline_tiles.to(relaxation_dtype)
            sparse_indices_by_tile = sparse_indices_hard_by_tile
            sparse_deltas_by_tile = sparse_candidate_deltas.permute(1, 0, 2).to(
                relaxation_dtype).contiguous()
            sparse_indices_flat = sparse_indices_by_tile.flatten(1)
        else:
            decoded_relax = decoded.to(relaxation_dtype)
    else:
        teacher = x @ target.detach().float()
        if right_factor is not None:
            teacher = teacher @ right_factor
        normalizer = teacher.square().mean().clamp_min(torch.finfo(torch.float32).tiny)

    def tiles_to_weight(tiles):
        return tiles.reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)

    def loss(tiles, *, structured_output=False):
        weight = tiles_to_weight(tiles)
        if fisher_objective:
            error = weight - target
            metric_error = h_metric @ error
            if structured_output:
                if g_hadamard_diagonal_hard is None:
                    raise ValueError("structured hard Fisher loss requires a Hadamard diagonal")
                hadamard_scale = 1. / math.sqrt(n)
                metric_error = hadamard_transform_scaled(
                    metric_error.contiguous(), g_hadamard_diagonal_hard,
                    hadamard_scale,
                )
                metric_error = hadamard_transform(
                    metric_error.contiguous(), hadamard_scale,
                )
            else:
                metric_error = metric_error @ g_metric
            return (error * metric_error).sum() / fisher_denominator
        prediction = x @ weight
        if right_factor is not None:
            prediction = prediction @ right_factor
        return ((prediction - teacher).square().mean() / normalizer)

    tile_ids = torch.arange(candidates.shape[1], device=candidates.device)

    def hard_candidate_tiles(choices):
        if decoded is not None:
            return decoded[tile_ids, choices]
        if sparse_indices_hard_by_tile is None or sparse_values_hard_by_tile is None:
            raise RuntimeError("sparse hard candidate materialization requires exact candidate values")
        alternative = (choices - 1).clamp_min(0)
        selected_indices = sparse_indices_hard_by_tile[tile_ids, alternative]
        selected_values = sparse_values_hard_by_tile[tile_ids, alternative]
        tiles = baseline_tiles.clone()
        baseline_values = tiles.gather(1, selected_indices)
        selected_values = torch.where(
            (choices > 0)[:, None], selected_values, baseline_values,
        )
        tiles.scatter_(1, selected_indices, selected_values)
        return tiles

    best_choices = torch.zeros_like(tile_ids)
    before = float(loss(baseline_tiles))
    if not math.isfinite(before):
        raise ValueError("non-finite calibration objective")
    best = before
    history = [before]
    coordinate_before = coordinate_after = before
    coordinate_changed = 0
    if coordinate_sweeps:
        # Start the continuous relaxation from an on-manifold assignment that
        # is already no worse than candidate zero.  This is also the matched
        # hard control used to measure the relaxation gap.
        with _nvtx_range("gsq.coordinate_initializer", candidates):
            coordinate_right = right_factor
            if not fisher_objective and coordinate_right is None:
                coordinate_right = torch.eye(n, device=target.device, dtype=target.dtype)
            coordinate = batched_trellis_candidates(
                candidates, target=target, inputs=inputs,
                right_factor=coordinate_right, bits=bits, layout=layout,
                bank_ids=bank_ids, bank_alt_id=bank_alt_id, codebook_version=codebook_version,
                sweeps=coordinate_sweeps, chunk_tiles=coordinate_chunk_tiles,
                decoded_candidates=decoded.transpose(0, 1).reshape(candidates.shape[0], -1, 16, 16),
                input_metric=input_metric, output_metric=output_metric,
            )
        coordinate_before, coordinate_after = coordinate.calibration_before, coordinate.calibration_after
        coordinate_changed = int((coordinate.choices != 0).sum())
        history.extend(coordinate.history[1:])
        if coordinate.calibration_after < best:
            best, best_choices = coordinate.calibration_after, coordinate.choices.clone()
    fused_sparse_relaxation = (
        fisher_objective and sparse_relaxation and candidates.device.type == "cuda"
        and gumbel_samples == 1
    )
    if sparse_candidate_shifts is not None:
        # Paper Appendix A initialization for b>2: a Gaussian-like prior over
        # local shifts, plus isotropic Gaussian noise.  QVQ's choices are legal
        # one-edge path edits rather than scalar codes, but each still carries
        # an exact {-2,-1,0,+1,+2} transition shift.  Keep the coordinate result
        # only as the hard no-regression guard; biasing logits toward that
        # already-minimized solution would pre-empt the stochastic search.
        shifts = torch.nn.functional.pad(sparse_candidate_shifts.T, (1, 0)).float()
        prior = -0.5 * shifts.square()
        prior -= prior.mean(-1, keepdim=True)
        init_generator = torch.Generator(device=candidates.device).manual_seed(seed)
        logits = initialization_std * (
            torch.randn(prior.shape, device=prior.device, generator=init_generator)
            + initialization_strength * prior
        )
        logits.requires_grad_(not fused_sparse_relaxation)
        initialization = "paper_local_shift_gaussian"
    else:
        logits = torch.zeros(
            (candidates.shape[1], candidates.shape[0]), device=candidates.device,
            requires_grad=not fused_sparse_relaxation,
        )
        # Legacy adapters do not expose a local-shift coordinate. Keep a small
        # unambiguous prior without pretending that it is paper initialization.
        with torch.no_grad():
            logits[tile_ids, best_choices] = 1.0 / kappa_start
        initialization = "guard_choice_margin"
    if fused_sparse_relaxation:
        from .qvq_gsq_triton import (
            build_compact_position_map,
            build_p32_dense_position_map,
            compact_position_error,
            scheduled_grouped_gumbel_softmax,
            scheduled_grouped_sparse_lion,
            update_compact_position_error,
        )
        from .qvq_gsq_triton import gumbel_softmax as fused_gumbel_softmax
        from .qvq_gsq_triton import sparse_error as fused_sparse_error
        from .qvq_gsq_triton import sparse_lion as fused_sparse_lion
        probabilities_buffer = torch.empty_like(logits)
        error_accumulation_buffer = torch.empty_like(target_relax, dtype=torch.float32)
        error_buffer = torch.empty_like(target_relax)
        metric_left_buffer = torch.empty_like(target_relax)
        metric_error_buffer = torch.empty_like(target_relax)
        momentum = torch.zeros_like(logits)
        gradient_norm_square = torch.zeros((), device=candidates.device)
        graph_chunk_steps = min(64, max(steps, 1))
        uniform_chunk = torch.empty(
            (graph_chunk_steps, *logits.shape), device=logits.device,
        )
        temperature_schedule = torch.tensor([
            temperature_start + (temperature_end - temperature_start)
            * step / max(steps - 1, 1) for step in range(steps)
        ], device=logits.device)
        kappa_schedule = torch.tensor([
            kappa_start + (kappa_end - kappa_start)
            * step / max(steps - 1, 1) for step in range(steps)
        ], device=logits.device)
        graph_step = torch.zeros((), dtype=torch.int32, device=logits.device)
        use_fast_position_map = bool(
            fast_position_map and layout == "p32_window"
            and sparse_indices_by_tile.shape[2] == 6
        )
        position_indices, position_choices, position_deltas = (
            build_p32_dense_position_map(
                sparse_indices_by_tile, sparse_deltas_by_tile,
            ) if use_fast_position_map else
            build_compact_position_map(
                sparse_indices_by_tile, sparse_deltas_by_tile,
            )
        )
        packed_sparse_indices_by_tile = sparse_indices_by_tile.to(torch.uint8)
    else:
        use_fast_position_map = False
        from .gsq_training import GSQLion
        optimizer = GSQLion([logits], lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device=candidates.device).manual_seed(seed)
    gradient_norm_min = torch.full((), math.inf, device=candidates.device)
    gradient_norm_max = torch.zeros((), device=candidates.device)
    finite_state = torch.ones((), dtype=torch.bool, device=candidates.device)
    relaxed_objective_last = None
    initial_entropy = initial_max_probability = None
    final_entropy = final_max_probability = None
    initializer_best = best
    initializer_choices = best_choices.clone()
    best_device = torch.tensor(best, device=candidates.device)
    relaxation_improved_initializer = False
    stale_steps = 0
    hard_evaluations = 0
    completed_steps = 0
    relaxation_graph = None
    graph_updates_per_replay = 1
    structured_hard_oracle = bool(
        fisher_objective and g_hadamard_diagonal_hard is not None
        and not relaxation_patience and hard_dense_verify_topk
    )
    hard_checkpoint_count = (
        steps // hard_eval_interval + int(bool(steps % hard_eval_interval))
    )
    if structured_hard_oracle and hard_checkpoint_count:
        hard_checkpoint_losses = torch.empty(
            hard_checkpoint_count, device=candidates.device, dtype=torch.float32,
        )
        hard_checkpoint_choices = torch.empty(
            (hard_checkpoint_count, candidates.shape[1]),
            device=candidates.device,
            dtype=torch.uint8 if candidates.shape[0] <= 256 else torch.int32,
        )
    if fused_sparse_relaxation and steps:
        # CUDA graph capture needs every kernel/module loaded first. Warm the
        # exact scheduled path on scratch state, then capture one update. The
        # graph reads its schedule and uniform slice from the GPU step counter,
        # so one graph is valid for the complete requested schedule.
        warm_logits = logits.detach().clone()
        warm_probabilities = torch.empty_like(probabilities_buffer)
        warm_error = torch.empty_like(error_buffer)
        warm_metric_left = torch.empty_like(metric_left_buffer)
        warm_metric_error = torch.empty_like(metric_error_buffer)
        warm_momentum = torch.zeros_like(momentum)
        warm_norm_square = torch.zeros_like(gradient_norm_square)
        warm_norm_minimum = torch.full_like(gradient_norm_min, math.inf)
        warm_norm_maximum = torch.zeros_like(gradient_norm_max)
        warm_finite = torch.ones_like(finite_state)
        warm_step = torch.zeros_like(graph_step)
        uniform_chunk.uniform_(generator=generator)
        scheduled_grouped_gumbel_softmax(
            warm_logits, uniform_chunk, warm_probabilities,
            temperature_schedule, kappa_schedule, warm_step,
        )
        compact_position_error(
            warm_probabilities, baseline_relax, position_indices, position_choices,
            position_deltas, target_relax, warm_error,
        )
        # Only compact positions vary across relaxation updates. Preserve the
        # dense baseline once; captured updates overwrite every active entry.
        error_buffer.copy_(warm_error)
        update_compact_position_error(
            warm_probabilities, baseline_relax, position_indices,
            position_choices, position_deltas, target_relax, error_buffer,
        )
        torch.mm(h_relax, warm_error, out=warm_metric_left)
        if g_hadamard_diagonal is None:
            torch.mm(warm_metric_left, g_relax, out=warm_metric_error)
        else:
            hadamard_scale = 1. / math.sqrt(n)
            warm_metric_error = _right_fisher_hadamard(
                warm_metric_left, g_hadamard_diagonal, hadamard_scale,
            )
        scheduled_grouped_sparse_lion(
            warm_probabilities, warm_metric_error, packed_sparse_indices_by_tile,
            sparse_deltas_by_tile, fisher_denominator, warm_logits,
            warm_momentum, warm_norm_square, warm_norm_minimum,
            warm_norm_maximum, warm_finite, temperature_schedule,
            kappa_schedule, warm_step, learning_rate, weight_decay,
        )
        torch.cuda.current_stream().synchronize()
        if capture_barrier is not None:
            try:
                capture_barrier.wait(timeout=60.)
            except threading.BrokenBarrierError as error:
                raise RuntimeError(
                    "concurrent GSQ CUDA graph warm-up barrier failed"
                ) from error
        generator.manual_seed(seed)
        relaxation_graph = torch.cuda.CUDAGraph()
        if (not relaxation_patience and progress is None
                and steps % cuda_graph_updates_per_replay == 0
                and hard_eval_interval % cuda_graph_updates_per_replay == 0
                and graph_chunk_steps % cuda_graph_updates_per_replay == 0):
            graph_updates_per_replay = cuda_graph_updates_per_replay
        # Q and K Fisher refinements may be captured concurrently on independent
        # host threads and CUDA streams. Thread-local capture keeps unrelated
        # work on the peer stream from invalidating either graph.
        with _CUDA_GRAPH_CAPTURE_LOCK:
            with torch.cuda.graph(relaxation_graph, capture_error_mode="thread_local"):
                for _ in range(graph_updates_per_replay):
                    scheduled_grouped_gumbel_softmax(
                        logits, uniform_chunk, probabilities_buffer,
                        temperature_schedule, kappa_schedule, graph_step,
                    )
                    update_compact_position_error(
                        probabilities_buffer, baseline_relax, position_indices, position_choices,
                        position_deltas, target_relax, error_buffer,
                    )
                    torch.mm(h_relax, error_buffer, out=metric_left_buffer)
                    if g_hadamard_diagonal is None:
                        torch.mm(metric_left_buffer, g_relax, out=metric_error_buffer)
                    else:
                        metric_error_buffer = _right_fisher_hadamard(
                            metric_left_buffer, g_hadamard_diagonal, hadamard_scale,
                        )
                    scheduled_grouped_sparse_lion(
                        probabilities_buffer, metric_error_buffer,
                        packed_sparse_indices_by_tile, sparse_deltas_by_tile,
                        fisher_denominator, logits, momentum, gradient_norm_square,
                        gradient_norm_min, gradient_norm_max, finite_state,
                        temperature_schedule, kappa_schedule, graph_step,
                        learning_rate, weight_decay,
                    )
            with torch.no_grad():
                if sparse_candidate_shifts is not None:
                    shifts = torch.nn.functional.pad(sparse_candidate_shifts.T, (1, 0)).float()
                    prior = -0.5 * shifts.square()
                    prior -= prior.mean(-1, keepdim=True)
                    init_generator = torch.Generator(device=candidates.device).manual_seed(seed)
                    logits.copy_(initialization_std * (
                        torch.randn(prior.shape, device=prior.device, generator=init_generator)
                        + initialization_strength * prior
                    ))
                else:
                    logits.zero_()
                    logits[tile_ids, best_choices] = 1.0 / kappa_start
                momentum.zero_()
                gradient_norm_square.zero_()
                gradient_norm_min.fill_(math.inf)
                gradient_norm_max.zero_()
                finite_state.fill_(True)
                graph_step.zero_()
            torch.cuda.current_stream().synchronize()
            generator.manual_seed(seed)
        if capture_barrier is not None:
            try:
                capture_barrier.wait(timeout=60.)
            except threading.BrokenBarrierError as error:
                raise RuntimeError("concurrent GSQ CUDA graph capture barrier failed") from error
    with torch.enable_grad(), _nvtx_range("gsq.lion_relaxation", candidates):
        for replay_start in range(0, steps, graph_updates_per_replay):
            step = replay_start + graph_updates_per_replay - 1
            fraction = step / max(steps - 1, 1)
            tau = temperature_start + (temperature_end - temperature_start) * fraction
            kappa = kappa_start + (kappa_end - kappa_start) * fraction
            evaluate_hard = (step + 1) % hard_eval_interval == 0 or step + 1 == steps
            if fused_sparse_relaxation:
                if relaxation_graph is not None:
                    if replay_start % graph_chunk_steps == 0:
                        uniform_chunk.uniform_(generator=generator)
                    relaxation_graph.replay()
                else:
                    uniform = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-6, 1 - 1e-6)
                    fused_gumbel_softmax(logits, uniform, probabilities_buffer, tau, kappa)
                    fused_sparse_error(
                        probabilities_buffer, baseline_relax, sparse_indices_by_tile,
                        sparse_deltas_by_tile, target_relax, error_accumulation_buffer,
                    )
                    error_buffer.copy_(error_accumulation_buffer)
                    torch.mm(h_relax, error_buffer, out=metric_left_buffer)
                    if g_hadamard_diagonal is None:
                        torch.mm(metric_left_buffer, g_relax, out=metric_error_buffer)
                    else:
                        metric_error_buffer = _right_fisher_hadamard(
                            metric_left_buffer, g_hadamard_diagonal, hadamard_scale,
                        )
                objective = None
                # The relaxed objective is diagnostic only; hard checkpoints
                # below remain the authoritative best-state selector.  Report
                # the final value without reducing two full KxN tensors at
                # every hard checkpoint.
                if step + 1 == steps:
                    objective_error = (
                        error_buffer.float() if relaxation_graph is not None
                        else error_accumulation_buffer
                    )
                    objective = (
                        objective_error * metric_error_buffer.float()
                    ).sum() / fisher_denominator
                    finite_state &= torch.isfinite(objective)
                    relaxed_objective_last = objective.detach()
                if relaxation_graph is None:
                    fused_sparse_lion(
                        probabilities_buffer, metric_error_buffer,
                        sparse_indices_by_tile, sparse_deltas_by_tile,
                        fisher_denominator, logits, momentum, gradient_norm_square,
                        gradient_norm_min, gradient_norm_max, finite_state,
                        tau, kappa, learning_rate, weight_decay,
                    )
                probability_stats = probabilities_buffer
            else:
                objectives = []
                probability_stats = None
                gradient_surrogates = []
                for _ in range(gumbel_samples):
                    uniform = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-6, 1 - 1e-6)
                    probabilities = _candidate_probabilities(logits, uniform, tau, kappa)
                    if fisher_objective:
                        # Evaluate the exact Kronecker Fisher objective with two
                        # GEMMs, then inject its analytic dL/dp through softmax.
                        # This avoids retaining and backpropagating through the
                        # huge KxK and NxN GEMM graph.
                        with torch.no_grad():
                            if sparse_relaxation:
                                # Local P32 candidates share candidate zero at
                                # all but a handful of scalar positions.
                                weighted_deltas = (
                                    probabilities.detach()[:, 1:].to(relaxation_dtype).unsqueeze(-1)
                                    * sparse_deltas_by_tile
                                )
                                soft_tiles = baseline_relax.clone()
                                soft_tiles.scatter_add_(
                                    1, sparse_indices_flat, weighted_deltas.flatten(1),
                                )
                            else:
                                # Batched 1xC @ Cx256 avoids materializing a
                                # [tile,candidate,256] product (2.2 GiB here).
                                soft_tiles = torch.bmm(
                                    probabilities.detach().to(relaxation_dtype).unsqueeze(1), decoded_relax,
                                ).squeeze(1)
                            soft_weight = tiles_to_weight(soft_tiles)
                            error = soft_weight - target_relax
                            metric_error = h_relax @ error @ g_relax
                        # This scalar is diagnostic only: the exact analytic
                        # gradient below already consumes H E G. Reducing two
                        # full KxN FP32 tensors every update needlessly costs
                        # more than the sparse relaxation itself, so sample it
                        # at the same authoritative hard checkpoints.
                            if evaluate_hard:
                                soft_objective = (
                                    error.float() * metric_error.float()
                                ).sum() / fisher_denominator
                            weight_gradient = 2 * metric_error.float() / fisher_denominator
                            tile_gradient = (
                                weight_gradient.reshape(k // 16, 16, n // 16, 16)
                                .permute(0, 2, 1, 3).reshape(-1, 256)
                            )
                            if sparse_relaxation:
                                changed_gradient = tile_gradient.to(relaxation_dtype).gather(
                                    1, sparse_indices_flat,
                                ).reshape_as(sparse_deltas_by_tile)
                                alternative_gradient = (
                                    changed_gradient * sparse_deltas_by_tile
                                ).sum(-1).float()
                            # The baseline dot-product is common to every
                            # choice and cancels exactly in the softmax
                            # Jacobian, so candidate zero can be represented
                            # by zero here.
                                probability_gradient = torch.nn.functional.pad(
                                    alternative_gradient, (1, 0), value=0,
                                )
                            else:
                                probability_gradient = torch.bmm(
                                    decoded_relax, tile_gradient.to(relaxation_dtype).unsqueeze(-1),
                                ).squeeze(-1).float()
                        if evaluate_hard:
                            objectives.append(soft_objective)
                        gradient_surrogates.append((probabilities * probability_gradient).sum())
                    else:
                        value = loss((probabilities.unsqueeze(-1) * decoded).sum(1))
                        objectives.append(value)
                        gradient_surrogates.append(value)
                    probability_stats = probabilities if probability_stats is None else probability_stats + probabilities
                objective = torch.stack(objectives).mean() if objectives else None
                optimizer.zero_grad()
                torch.stack(gradient_surrogates).mean().backward()
                gradient_norm = logits.grad.norm()
                finite_state &= torch.isfinite(gradient_norm)
                if objective is not None:
                    finite_state &= torch.isfinite(objective)
                gradient_norm_min = torch.minimum(gradient_norm_min, gradient_norm.detach())
                gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach())
                if objective is not None:
                    relaxed_objective_last = objective.detach()
                optimizer.step()
            completed_steps = step + 1
            with torch.no_grad():
                mean_probability = probability_stats / gumbel_samples
                # Only endpoints are exported.  Scanning every tile/choice at
                # intermediate hard checkpoints cannot affect optimization.
                if initial_entropy is None or step + 1 == steps:
                    entropy_value = (
                        -(mean_probability.clamp_min(1e-20).log() * mean_probability).sum(-1)
                    ).mean()
                    max_probability_value = mean_probability.max(-1).values.mean()
                    if initial_entropy is None:
                        initial_entropy = entropy_value
                        initial_max_probability = max_probability_value
                    final_entropy = entropy_value
                    final_max_probability = max_probability_value
                if evaluate_hard:
                    choices = logits.argmax(-1)
                    hard_loss_tensor = loss(
                        hard_candidate_tiles(choices),
                        structured_output=structured_hard_oracle,
                    )
                    if structured_hard_oracle:
                        hard_checkpoint_losses[hard_evaluations].copy_(hard_loss_tensor)
                        hard_checkpoint_choices[hard_evaluations].copy_(choices)
                    hard_evaluations += 1
                    if relaxation_patience:
                        if not bool(finite_state):
                            raise ValueError("non-finite GSQ relaxed objective or logit gradient")
                        hard_loss = float(hard_loss_tensor)
                        if not math.isfinite(hard_loss):
                            raise ValueError("non-finite hard objective")
                        history.append(hard_loss)
                        if hard_loss < best:
                            best, best_choices = hard_loss, choices.clone()
                            best_device.fill_(best)
                            relaxation_improved_initializer = True
                            stale_steps = 0
                        else:
                            stale_steps += 1
                    else:
                        # A no-patience paper schedule never needs a host-side
                        # decision inside the loop. Keep the exact hard oracle
                        # and strict tie rule on device, then transfer once.
                        torch._assert_async(
                            finite_state,
                            "non-finite GSQ relaxed objective or logit gradient",
                        )
                        torch._assert_async(
                            torch.isfinite(hard_loss_tensor),
                            "non-finite hard objective",
                        )
                        history.append(hard_loss_tensor.detach())
                        if not structured_hard_oracle:
                            improved = hard_loss_tensor < best_device
                            best_choices.copy_(torch.where(
                                improved, choices, best_choices,
                            ))
                            best_device.copy_(torch.minimum(
                                best_device, hard_loss_tensor,
                            ))
                if progress is not None:
                    progress(step + 1, float(best_device))
                if evaluate_hard and relaxation_patience and stale_steps >= relaxation_patience:
                    break
    if not relaxation_patience and hard_evaluations:
        if structured_hard_oracle:
            verify_count = min(hard_dense_verify_topk, hard_evaluations)
            # Stable ordering makes the verification set and strict dense
            # tie-breaking deterministic when structured FP32 scores collide.
            verify_indices = torch.argsort(
                hard_checkpoint_losses[:hard_evaluations], stable=True,
            )[:verify_count].sort().values
            for checkpoint_index in verify_indices.unbind():
                choices = hard_checkpoint_choices[checkpoint_index].long()
                hard_loss_tensor = loss(hard_candidate_tiles(choices))
                torch._assert_async(
                    torch.isfinite(hard_loss_tensor),
                    "non-finite dense verification objective",
                )
                improved = hard_loss_tensor < best_device
                best_choices.copy_(torch.where(improved, choices, best_choices))
                best_device.copy_(torch.minimum(best_device, hard_loss_tensor))
        best = float(best_device)
        relaxation_improved_initializer = best < initializer_best
        tensor_indices = [
            index for index, value in enumerate(history)
            if isinstance(value, torch.Tensor)
        ]
        if tensor_indices:
            tensor_values = torch.stack([
                history[index] for index in tensor_indices
            ]).cpu().tolist()
            for index, value in zip(tensor_indices, tensor_values):
                history[index] = value
    words = candidates[best_choices, tile_ids].detach().clone().contiguous()
    diagnostics = {
        "optimizer": "lion", "weight_decay": weight_decay,
        "initialization": initialization,
        "initialization_std": initialization_std,
        "initialization_strength": initialization_strength,
        "temperature_start": temperature_start, "temperature_end": temperature_end,
        "kappa_start": kappa_start, "kappa_end": kappa_end,
        "gumbel_samples": gumbel_samples,
        "soft_dtype": str(relaxation_dtype).removeprefix("torch.") if fisher_objective else "float32",
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "hard_eval_interval": hard_eval_interval,
        "hard_evaluations": hard_evaluations,
        "hard_oracle": (
            "structured_fp32_dense_topk" if structured_hard_oracle else "dense_fp32"
        ),
        "hard_dense_verify_topk": (
            min(hard_dense_verify_topk, hard_evaluations)
            if structured_hard_oracle else 0
        ),
        "relaxation_patience": relaxation_patience,
        "optimization_regime": "exact_full_fisher" if fisher_objective else "activation_reconstruction",
        "sparse_relaxation": sparse_relaxation,
        "sparse_hard_candidate_bank": sparse_baseline_only,
        "fused_sparse_relaxation": fused_sparse_relaxation,
        "cuda_graph_relaxation": relaxation_graph is not None,
        "cuda_graph_updates_per_replay": graph_updates_per_replay,
        "initial_probability_update": graph_updates_per_replay,
        "sparse_width": int(sparse_candidate_indices.shape[2]) if sparse_relaxation else None,
        "sparse_accumulation_dtype": "float32" if fused_sparse_relaxation else None,
        "sparse_index_dtype": "uint8" if fused_sparse_relaxation else None,
        "compact_positions": int(position_indices.shape[1]) if fused_sparse_relaxation else None,
        "position_map_backend": "p32_direct" if use_fast_position_map else "generic_sort",
        "updates_are_epochs": False,
        "initial_entropy": None if initial_entropy is None else float(initial_entropy),
        "final_entropy": None if final_entropy is None else float(final_entropy),
        "initial_mean_max_probability": None if initial_max_probability is None else float(initial_max_probability),
        "final_mean_max_probability": None if final_max_probability is None else float(final_max_probability),
        "gradient_norm_min": None if not completed_steps else float(gradient_norm_min),
        "gradient_norm_max": None if not completed_steps else float(gradient_norm_max),
        "relaxed_objective_last": None if relaxed_objective_last is None else float(relaxed_objective_last),
        "initializer_after": initializer_best,
        "initializer_changed_tiles": int((initializer_choices != 0).sum()),
        "relaxation_hard_after": best,
        "relaxation_changed_tiles": int((best_choices != 0).sum()),
        "relaxation_improved_initializer": relaxation_improved_initializer,
        "coordinate_before": coordinate_before,
        "coordinate_after": coordinate_after,
        "coordinate_changed_tiles": coordinate_changed,
        "selected_arm": "relaxation" if relaxation_improved_initializer else (
            "coordinate" if coordinate_sweeps else "baseline"),
    }
    return GSQResult(words, best_choices, before, best, history, diagnostics)


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


def _transition_positions(tile_count, edge_count, group_count, *, device, generator):
    """Generate distinct per-tile transition positions without a 128-way sort."""
    offsets = torch.randint(edge_count, (tile_count,), device=device, generator=generator)
    # P32 has 128 edges.  Every odd stride is coprime to 128, so this affine
    # walk is a permutation and the first ``group_count`` positions are unique.
    strides = 2 * torch.randint(edge_count // 2, (tile_count,), device=device, generator=generator) + 1
    groups = torch.arange(group_count, device=device, dtype=torch.int64)
    return (offsets[:, None] + strides[:, None] * groups[None, :]) % edge_count


def _p32_shift_alternatives(baseline, positions, shifts, transition_bits):
    """Edit one logical P32 edge per tile directly in continuous-window words."""
    word_mask = (1 << 32) - 1
    edge_mask = (1 << transition_bits) - 1
    flat = baseline.reshape(len(baseline), -1).to(torch.int64) & word_mask
    tile_ids = torch.arange(len(flat), device=flat.device)
    bit_positions = (127 - positions.to(torch.int64)) * transition_bits
    word_ids, offsets = bit_positions >> 5, bit_positions & 31
    old = (flat[tile_ids, word_ids] >> offsets) & edge_mask
    crosses = offsets + transition_bits > 32
    crossing_tiles = tile_ids[crosses]
    crossing_words = word_ids[crosses] + 1
    old[crosses] |= (flat[crossing_tiles, crossing_words] << (32 - offsets[crosses])) & edge_mask

    alternatives = flat.unsqueeze(0).repeat(len(shifts), 1, 1)
    low_width = torch.minimum(torch.full_like(offsets, transition_bits), 32 - offsets)
    low_mask = ((torch.ones_like(low_width) << low_width) - 1) << offsets
    for shift_index, shift in enumerate(shifts):
        new = (old + shift) & edge_mask
        words = alternatives[shift_index, tile_ids, word_ids]
        words = (words & (word_mask ^ low_mask)) | ((new << offsets) & word_mask)
        alternatives[shift_index, tile_ids, word_ids] = words
        high_width = offsets[crosses] + transition_bits - 32
        high_mask = (torch.ones_like(high_width) << high_width) - 1
        words = alternatives[shift_index, crossing_tiles, crossing_words]
        words = (words & (word_mask ^ high_mask)) | ((new[crosses] >> (32 - offsets[crosses])) & high_mask)
        alternatives[shift_index, crossing_tiles, crossing_words] = words
    return alternatives.to(torch.int32).contiguous()


def _p32_grouped_shift_alternatives(baseline, positions, shifts, transition_bits):
    """Edit several independent P32 edges per tile in one batched dispatch.

    ``positions`` is ``[tiles, groups]``.  The returned group-major tensor has
    shape ``[groups, shifts, tiles, words]`` and is bit-identical to stacking
    :func:`_p32_shift_alternatives` over the group dimension.  Keeping several
    groups in one tensor amortizes P32 unpack/decode and Fisher-screen launch
    overhead without changing the legal one-edge candidate definition.
    """
    if positions.ndim != 2 or positions.shape[0] != len(baseline):
        raise ValueError("grouped P32 positions must have shape [tiles,groups]")
    word_mask = (1 << 32) - 1
    edge_mask = (1 << transition_bits) - 1
    flat = baseline.reshape(len(baseline), -1).to(torch.int64) & word_mask
    tile_count, group_count = positions.shape
    shift_count = len(shifts)
    bit_positions = (127 - positions.to(torch.int64)) * transition_bits
    word_ids, offsets = bit_positions >> 5, bit_positions & 31
    old = flat.gather(1, word_ids) >> offsets
    crosses = offsets + transition_bits > 32
    crossing_words = word_ids + 1
    high_source = flat.gather(1, crossing_words.clamp_max(flat.shape[1] - 1))
    old |= torch.where(crosses, high_source << (32 - offsets), torch.zeros_like(old))
    old &= edge_mask

    # [group,shift,tile,word].  Expand is allocation-free; clone materializes
    # the output once instead of repeating the baseline separately per group.
    alternatives = flat.T.mT[None, None].expand(
        group_count, shift_count, tile_count, flat.shape[1]
    ).clone()
    group_word_ids = word_ids.T[:, None, :, None].expand(group_count, shift_count, tile_count, 1)
    group_offsets = offsets.T[:, None, :]
    group_crosses = crosses.T[:, None, :]
    low_width = torch.minimum(torch.full_like(group_offsets, transition_bits), 32 - group_offsets)
    low_mask = ((torch.ones_like(low_width) << low_width) - 1) << group_offsets
    shifts_tensor = torch.as_tensor(shifts, device=flat.device, dtype=torch.int64)[None, :, None]
    new = (old.T[:, None, :] + shifts_tensor) & edge_mask

    words = alternatives.gather(3, group_word_ids).squeeze(3)
    words = (words & (word_mask ^ low_mask)) | ((new << group_offsets) & word_mask)
    alternatives.scatter_(3, group_word_ids, words.unsqueeze(3))

    crossing_word_ids = crossing_words.T[:, None, :, None].expand_as(group_word_ids)
    high_width = group_offsets + transition_bits - 32
    high_mask = torch.where(
        group_crosses,
        (torch.ones_like(high_width) << high_width.clamp_min(0)) - 1,
        torch.zeros_like(high_width),
    )
    words = alternatives.gather(3, crossing_word_ids.clamp_max(flat.shape[1] - 1)).squeeze(3)
    high_values = (new >> (32 - group_offsets)) & high_mask
    words = torch.where(group_crosses, (words & (word_mask ^ high_mask)) | high_values, words)
    alternatives.scatter_(3, crossing_word_ids.clamp_max(flat.shape[1] - 1), words.unsqueeze(3))
    return alternatives.to(torch.int32).contiguous()


def _p32_selected_shift_alternatives(baseline, positions, selected_shifts, transition_bits):
    """Edit one already-selected edge shift for each ``[tile,group]`` pair."""
    if positions.ndim != 2 or positions.shape != selected_shifts.shape or positions.shape[0] != len(baseline):
        raise ValueError("selected P32 positions and shifts must have shape [tiles,groups]")
    word_mask = (1 << 32) - 1
    edge_mask = (1 << transition_bits) - 1
    flat = baseline.reshape(len(baseline), -1).to(torch.int64) & word_mask
    tile_count, group_count = positions.shape
    bit_positions = (127 - positions.to(torch.int64)) * transition_bits
    word_ids, offsets = bit_positions >> 5, bit_positions & 31
    old = flat.gather(1, word_ids) >> offsets
    crosses = offsets + transition_bits > 32
    crossing_words = word_ids + 1
    high_source = flat.gather(1, crossing_words.clamp_max(flat.shape[1] - 1))
    old |= torch.where(crosses, high_source << (32 - offsets), torch.zeros_like(old))
    new = (old + selected_shifts.to(torch.int64)) & edge_mask

    # Work group-major so the output can be appended directly to the candidate
    # bank.  Only the selected payload is materialized, not all four trials.
    alternatives = flat[None].expand(group_count, tile_count, flat.shape[1]).clone()
    group_word_ids = word_ids.T[:, :, None]
    group_offsets = offsets.T
    low_width = torch.minimum(torch.full_like(group_offsets, transition_bits), 32 - group_offsets)
    low_mask = ((torch.ones_like(low_width) << low_width) - 1) << group_offsets
    words = alternatives.gather(2, group_word_ids).squeeze(2)
    words = (words & (word_mask ^ low_mask)) | ((new.T << group_offsets) & word_mask)
    alternatives.scatter_(2, group_word_ids, words.unsqueeze(2))

    group_crosses = crosses.T
    crossing_word_ids = crossing_words.T[:, :, None]
    high_width = group_offsets + transition_bits - 32
    high_mask = torch.where(
        group_crosses,
        (torch.ones_like(high_width) << high_width.clamp_min(0)) - 1,
        torch.zeros_like(high_width),
    )
    words = alternatives.gather(2, crossing_word_ids.clamp_max(flat.shape[1] - 1)).squeeze(2)
    high_values = (new.T >> (32 - group_offsets)) & high_mask
    words = torch.where(group_crosses, (words & (word_mask ^ high_mask)) | high_values, words)
    alternatives.scatter_(2, crossing_word_ids.clamp_max(flat.shape[1] - 1), words.unsqueeze(2))
    return alternatives.to(torch.int32).contiguous()


def _p32_sparse_shift_screen(
    baseline,
    positions,
    shifts,
    *,
    bits,
    bank_ids,
    bank_alt_id,
    baseline_tiles,
    metric_tiles,
    h_tiles,
    g_tiles,
    codebook_version,
    identity_metric=False,
    materialize_dense=True,
    fused_identity=False,
):
    """Exactly score local P32 edits using only decoder states they change.

    A state is a 16-bit circular window beginning on an edge boundary.  An
    edited ``b``-bit edge can therefore affect only ``ceil(16 / b)`` states.
    At W3 that is three states (six scalar weights), versus decoding and
    multiplying the full 256-value tile for every trial in the old path.
    """
    if bank_ids is None or bank_alt_id is None:
        raise ValueError("sparse P32 screening requires selectors and an alternative bank")
    transition_bits = round(normalize_qvq_rate(bits) * 2)
    tile_count, group_count = positions.shape
    if fused_identity:
        if not identity_metric or transition_bits != 6:
            raise ValueError("fused P32 screening requires the W3 identity metric")
        from .qvq_gsq_triton import screen_p32_w3_identity_shifts

        levels = pgc16_levels_for_version(codebook_version).to(device=baseline.device)
        (selected_words, scalar_indices, selected_delta, selected_shifts,
         selected_new_values) = screen_p32_w3_identity_shifts(
            baseline, positions, bank_ids, bank_alt_id, baseline_tiles,
            metric_tiles, levels,
        )
        if materialize_dense:
            selected_values = baseline_tiles.reshape(tile_count, 256)[None].expand(
                group_count, -1, -1,
            ).clone()
            selected_values.scatter_(2, scalar_indices, selected_new_values)
        else:
            selected_values = None
        return (
            selected_words,
            (None if selected_values is None else
             selected_values.reshape(group_count, tile_count, 16, 16)),
            None,
            scalar_indices,
            selected_delta,
            selected_shifts,
            selected_new_values,
        )
    affected_count = (16 + transition_bits - 1) // transition_bits
    state_offsets = torch.arange(affected_count, device=baseline.device, dtype=torch.int64)
    affected_states = (positions.T[:, :, None] + state_offsets[None, None, :]) % 128

    states = unpack_p32_window_states(baseline, bits=bits).reshape(tile_count, 128)
    tile_ids = torch.arange(tile_count, device=baseline.device, dtype=torch.int64)
    old_states = states[tile_ids[None, :, None], affected_states]
    old_edges = old_states[:, :, 0] & ((1 << transition_bits) - 1)
    shift_tensor = torch.as_tensor(shifts, device=baseline.device, dtype=torch.int64)
    new_edges = (old_edges[:, None, :] + shift_tensor[None, :, None]) & ((1 << transition_bits) - 1)

    bit_offsets = state_offsets * transition_bits
    widths = torch.minimum(torch.full_like(bit_offsets, transition_bits), 16 - bit_offsets)
    value_masks = (torch.ones_like(widths) << widths) - 1
    state_masks = value_masks << bit_offsets
    new_states = (
        old_states[:, None].to(torch.int64) & (0xFFFF ^ state_masks[None, None, None, :])
    ) | (
        (new_edges[:, :, :, None] & value_masks[None, None, None, :])
        << bit_offsets[None, None, None, :]
    )

    alt_id = int(bank_alt_id.item())
    binary_ids = unpack_qvq_binary_bank_ids(
        bank_ids, tile_count * QVQ_V2B2_P32_SEGMENTS_PER_TILE,
    )
    state_bank_ids = (
        binary_ids.reshape(tile_count, QVQ_V2B2_P32_SEGMENTS_PER_TILE)
        .repeat_interleave(QVQ_V2B2_P32_STEPS_PER_SEGMENT, dim=1)
        .mul(alt_id)
    )
    affected_banks = state_bank_ids[tile_ids[None, :, None], affected_states]
    levels = pgc16_levels_for_version(codebook_version).to(device=baseline.device)
    new_values = pgc16_decode_states_v2_banked(
        new_states,
        affected_banks[:, None].expand_as(new_states),
        bits=bits,
        levels=levels,
    ).float()

    scalar_indices = torch.stack((2 * affected_states, 2 * affected_states + 1), dim=-1)
    scalar_indices = scalar_indices.reshape(group_count, tile_count, -1)
    baseline_flat = baseline_tiles.reshape(tile_count, 256)
    old_values = baseline_flat[tile_ids[None, :, None], scalar_indices]
    delta = new_values.reshape(group_count, len(shifts), tile_count, -1) - old_values[:, None]
    rows, columns = scalar_indices // 16, scalar_indices % 16
    metric_values = metric_tiles[tile_ids[None, :, None], rows, columns]
    cost = 2 * (delta * metric_values[:, None]).sum(-1)

    # Exact sparse form of tr(delta.T H delta G).  The largest W1 case has
    # only 16 changed scalars, while W3 has six.
    if identity_metric:
        pair_metric = (
            (rows[:, :, :, None] == rows[:, :, None, :])
            & (columns[:, :, :, None] == columns[:, :, None, :])
        ).float()
    else:
        h_pairs = h_tiles[
            tile_ids[None, :, None, None], rows[:, :, :, None], rows[:, :, None, :]
        ]
        g_pairs = g_tiles[
            tile_ids[None, :, None, None], columns[:, :, :, None], columns[:, :, None, :]
        ]
        pair_metric = h_pairs * g_pairs
    cost += (
        delta[..., :, None] * delta[..., None, :] * pair_metric[:, None]
    ).sum((-1, -2))

    selected = cost.argmin(1)
    group_ids = torch.arange(group_count, device=baseline.device)[:, None]
    selected_new_values = new_values[group_ids, selected, tile_ids[None]]
    if materialize_dense:
        selected_values = baseline_flat[None].expand(group_count, -1, -1).clone()
        selected_values.scatter_(
            2, scalar_indices,
            selected_new_values.reshape(group_count, tile_count, -1),
        )
    else:
        selected_values = None
    selected_shift_values = shift_tensor[selected].T
    selected_words = _p32_selected_shift_alternatives(
        baseline, positions, selected_shift_values, transition_bits,
    )
    selected_new_values = selected_new_values.reshape(group_count, tile_count, -1)
    selected_delta = selected_new_values - old_values
    return (
        selected_words,
        (None if selected_values is None else
         selected_values.reshape(group_count, tile_count, 16, 16)),
        cost,
        scalar_indices,
        selected_delta,
        selected_shift_values.T.contiguous(),
        selected_new_values,
    )


def trellis_local_candidates(baseline, *, count, seed, bits, layout, codebook_version=PGC16_CODEBOOK_VERSION):
    """Create reproducible local path substitutions in transition space.

    Candidate zero is the exact payload.  Remaining candidates are grouped in
    the paper's four non-zero W3 shifts {-2,-1,+1,+2}; each group selects a
    different transition in every tile without replacement.  Packing from the
    edited circular edge stream reconstructs all overlapping 16-bit states, so
    every candidate is a legal tail-biting QVQ path and round-trips exactly.
    """
    if baseline.ndim != 2 or baseline.dtype != torch.int32 or min(baseline.shape) <= 0:
        raise ValueError("GSQ baseline must be nonempty int32 [tiles,words]")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("GSQ candidate count must be an integer >= 2")
    adapter = TrellisCandidateAdapter(layout, bits, codebook_version)
    edge_count = 128
    transition_bits = round(normalize_qvq_rate(bits) * 2)
    edge_mask = (1 << transition_bits) - 1
    generator = torch.Generator(device=baseline.device).manual_seed(seed)
    positions = _transition_positions(
        len(baseline), edge_count, (count - 2) // 4 + 1,
        device=baseline.device, generator=generator)
    shifts = (-2, -1, 1, 2)
    candidates = [baseline.detach().clone()]
    if layout == "p32_window":
        for group in range(positions.shape[1]):
            alternatives = _p32_shift_alternatives(
                baseline, positions[:, group], shifts, transition_bits)
            candidates.extend(alternatives.unbind(0))
        return torch.stack(candidates[:count])

    states = adapter.unpack(baseline)
    base_edges = states & edge_mask
    tile_ids = torch.arange(len(baseline), device=baseline.device)
    for index in range(1, count):
        group, shift_index = divmod(index - 1, len(shifts))
        edges = base_edges.clone()
        position = positions[:, group]
        edges[tile_ids, position] = (edges[tile_ids, position] + shifts[shift_index]) & edge_mask
        columns = edges.reshape(-1, edge_count).transpose(0, 1).contiguous()
        planar = planar_pack_rows(columns, transition_bits).transpose(0, 1).contiguous()
        consistent_states = unpack_trellis_states(planar, bits=bits)
        candidate = adapter.pack(consistent_states)
        if not torch.equal(adapter.unpack(candidate), consistent_states):
            raise RuntimeError("GSQ local trellis candidate failed exact round-trip")
        candidates.append(candidate)
    return torch.stack(candidates)


@torch.no_grad()
def fisher_screened_trellis_candidates(
    baseline, *, count, seed, bits, layout, target, input_hessian, output_hessian,
    bank_ids=None, bank_alt_id=None, codebook_version=PGC16_CODEBOOK_VERSION,
    return_decoded=False,
    return_sparse=False,
    return_shifts=False,
    compact_sparse=False,
    identity_metric=False,
    fused_identity_screen=True,
):
    """Screen four paper-local shifts at each candidate transition.

    A 33-candidate budget cannot retain {-2,-1,+1,+2} independently at all 128
    transitions of a tile.  For each of 32 distinct transition positions this
    routine evaluates all four shifts with the exact one-tile Fisher delta and
    retains the best shift independently per tile.  The resulting shared bank
    covers four times as many path coordinates as grouping four output choices
    per position, while the later coupled search remains authoritative.

    ``compact_sparse`` replaces the dense decoded bank in the return tuple
    with the decoded baseline and appends exact values for the sparse entries.
    """
    if target.ndim != 2:
        raise ValueError("GSQ Fisher candidate screen requires a rank-two target")
    if identity_metric and layout != "p32_window":
        raise ValueError("identity candidate screening currently requires P32 window layout")
    if not isinstance(fused_identity_screen, bool):
        raise TypeError("fused identity screening control must be boolean")
    if compact_sparse and (layout != "p32_window" or not return_decoded or not return_sparse):
        raise ValueError(
            "compact sparse candidate screening requires decoded sparse P32 output"
        )
    if not identity_metric and input_hessian.shape != (target.shape[0], target.shape[0]):
        raise ValueError("GSQ Fisher candidate screen requires a matching target and input Hessian")
    if not identity_metric and output_hessian.shape != (target.shape[1], target.shape[1]):
        raise ValueError("GSQ Fisher candidate screen requires a matching output Hessian")
    adapter = TrellisCandidateAdapter(layout, bits, codebook_version)
    transition_bits = round(normalize_qvq_rate(bits) * 2)
    generator = torch.Generator(device=baseline.device).manual_seed(seed)
    positions = _transition_positions(
        len(baseline), 128, count - 1, device=baseline.device, generator=generator)
    raw = None
    if layout != "p32_window":
        with _nvtx_range("gsq.candidates.raw_local_shifts", baseline):
            raw = trellis_local_candidates(
                baseline, count=1 + 4 * (count - 1), seed=seed, bits=bits, layout=layout,
                codebook_version=codebook_version)
    k, n = target.shape
    input_tiles, output_tiles = k // 16, n // 16
    current = adapter.inner(baseline, k, n, bank_ids, bank_alt_id).float()
    with _nvtx_range("gsq.candidates.metric_error", baseline):
        metric_error = (
            current - target.float()
            if identity_metric else
            input_hessian.float() @ (current - target.float()) @ output_hessian.float()
        )
    metric_tiles = metric_error.reshape(
        input_tiles, 16, output_tiles, 16,
    ).permute(0, 2, 1, 3).reshape(-1, 16, 16).contiguous()
    if identity_metric:
        h_tiles = g_tiles = None
    else:
        h, g = input_hessian.float(), output_hessian.float()
        h_blocks = torch.stack([h[16*i:16*(i+1), 16*i:16*(i+1)] for i in range(input_tiles)])
        g_blocks = torch.stack([g[16*j:16*(j+1), 16*j:16*(j+1)] for j in range(output_tiles)])
        h_tiles = h_blocks.repeat_interleave(output_tiles, 0)
        g_tiles = g_blocks.repeat(input_tiles, 1, 1)
    baseline_tiles = adapter.decode(baseline, bank_ids, bank_alt_id).reshape(-1, 16, 16).float()
    tile_ids = torch.arange(len(baseline), device=baseline.device)
    screened = [baseline.detach().clone()]
    screened_values = None if compact_sparse else [baseline_tiles]
    sparse_indices = []
    sparse_deltas = []
    sparse_shifts = []
    sparse_values = []
    with _nvtx_range("gsq.candidates.decode_and_screen", baseline):
        # Sparse P32 scoring touches at most 16 scalars (six at W3).  Hopper can
        # screen the full default bank together so state and selector unpacking
        # happens once per projection instead of once per half-bank.  Preserve
        # the lower-memory chunk on unmeasured architectures and cap custom
        # candidate banks at the measured 32-group launch.
        # The candidate order and each per-tile argmin remain unchanged.
        hopper = (
            baseline.device.type == "cuda"
            and torch.cuda.get_device_capability(baseline.device)[0] >= 9
        )
        screen_groups = (
            min(count - 1, 32 if hopper else 16)
            if layout == "p32_window" else 1
        )
        for start in range(0, count - 1, screen_groups):
            stop = min(start + screen_groups, count - 1)
            group_count = stop - start
            if layout == "p32_window":
                (selected_words, selected_values, _, selected_indices,
                 selected_deltas, selected_shifts, selected_sparse_values) = _p32_sparse_shift_screen(
                    baseline,
                    positions[:, start:stop],
                    (-2, -1, 1, 2),
                    bits=bits,
                    bank_ids=bank_ids,
                    bank_alt_id=bank_alt_id,
                    baseline_tiles=baseline_tiles,
                    metric_tiles=metric_tiles,
                    h_tiles=h_tiles,
                    g_tiles=g_tiles,
                    codebook_version=codebook_version,
                    identity_metric=identity_metric,
                    materialize_dense=not compact_sparse,
                    fused_identity=(
                        fused_identity_screen and identity_metric
                        and baseline.device.type == "cuda" and transition_bits == 6
                    ),
                )
                screened.extend(selected_words.unbind(0))
                if screened_values is not None:
                    screened_values.extend(selected_values.unbind(0))
                sparse_indices.extend(selected_indices.unbind(0))
                sparse_deltas.extend(selected_deltas.unbind(0))
                sparse_shifts.extend(selected_shifts.unbind(0))
                sparse_values.extend(selected_sparse_values.unbind(0))
                continue
            alternatives = (_p32_grouped_shift_alternatives(
                baseline, positions[:, start:stop], (-2, -1, 1, 2), transition_bits)
                if layout == "p32_window" else
                raw[1 + 4*start:1 + 4*stop].reshape(group_count, 4, *raw.shape[1:]))
            # Group-major flattening requires one selector copy per shift in
            # every group.  All groups share the same packed selector stream.
            decode_banks = bank_ids.repeat(group_count * 4) if bank_ids is not None else None
            values = adapter.decode(
                alternatives.reshape(-1, alternatives.shape[-1]), decode_banks, bank_alt_id,
            ).reshape(group_count, 4, -1, 16, 16).float()
            delta = values - baseline_tiles[None, None]
            cost = 2 * (delta * metric_tiles[None, None]).sum((3, 4))
            quadratic = torch.matmul(
                torch.matmul(h_tiles[None, None], delta), g_tiles[None, None]
            )
            cost += (quadratic * delta).sum((3, 4))
            selected = cost.argmin(1)
            group_ids = torch.arange(group_count, device=baseline.device)[:, None]
            selected_words = alternatives[group_ids, selected, tile_ids[None]].contiguous()
            selected_values = values[group_ids, selected, tile_ids[None]].contiguous()
            screened.extend(selected_words.unbind(0))
            screened_values.extend(selected_values.unbind(0))
    candidates = torch.stack(screened)
    if return_decoded:
        if compact_sparse:
            return (
                candidates,
                baseline_tiles,
                torch.stack(sparse_indices),
                torch.stack(sparse_deltas),
                torch.stack(sparse_shifts),
                torch.stack(sparse_values),
            )
        decoded = torch.stack(screened_values)
        if return_sparse:
            if layout != "p32_window" or len(sparse_indices) != count - 1:
                result = (candidates, decoded, None, None)
                return (*result, None) if return_shifts else result
            result = (candidates, decoded, torch.stack(sparse_indices), torch.stack(sparse_deltas))
            return (*result, torch.stack(sparse_shifts)) if return_shifts else result
        return candidates, decoded
    if return_sparse:
        raise ValueError("return_sparse requires return_decoded")
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
        # The exact Fisher path consumes H and G directly and uses an analytic
        # gradient.  Cholesky factors previously cost time and K^2/N^2 memory
        # despite being mathematically redundant here.
        factor_placeholder = target.new_zeros((1, target.shape[0]))
        decoded_candidates = None
        sparse_candidate_indices = sparse_candidate_deltas = sparse_candidate_shifts = None
        if config.qvq_candidate_policy == "trellis_local":
            candidates, decoded_candidates, sparse_candidate_indices, sparse_candidate_deltas, sparse_candidate_shifts = fisher_screened_trellis_candidates(
                baseline, count=config.candidates, seed=config.seed, bits=bits, layout=layout,
                target=target, input_hessian=input_hessian, output_hessian=output_hessian,
                bank_ids=bank_ids, bank_alt_id=bank_alt_id, codebook_version=codebook_version,
                return_decoded=True, return_sparse=True, return_shifts=True)
        else:
            candidates = baseline_bitflip_candidates(baseline, count=config.candidates, seed=config.seed)
        return refine_trellis_candidates(
            candidates, bits=bits, bank_ids=None if bank_ids is None else bank_ids.detach().clone(),
            bank_alt_id=None if bank_alt_id is None else bank_alt_id.detach().clone(), layout=layout,
            target=target, inputs=factor_placeholder, right_factor=None, enabled=True,
            codebook_version=codebook_version,
            steps=config.steps, seed=config.seed, learning_rate=config.qvq_learning_rate,
            temperature_start=config.qvq_temperature_start, temperature_end=config.qvq_temperature_end,
            kappa_start=config.qvq_kappa_start, kappa_end=config.qvq_kappa_end,
            weight_decay=config.qvq_weight_decay, gumbel_samples=config.qvq_gumbel_samples,
            initialization_std=config.qvq_initialization_std,
            initialization_strength=config.qvq_initialization_strength,
            soft_dtype=config.qvq_soft_dtype,
            coordinate_sweeps=config.qvq_coordinate_sweeps,
            coordinate_chunk_tiles=config.qvq_coordinate_chunk_tiles,
            hard_eval_interval=config.qvq_hard_eval_interval,
            cuda_graph_updates_per_replay=config.qvq_cuda_graph_updates_per_replay,
            relaxation_patience=config.qvq_relaxation_patience,
            decoded_candidates=decoded_candidates,
            sparse_candidate_indices=sparse_candidate_indices,
            sparse_candidate_deltas=sparse_candidate_deltas,
            sparse_candidate_shifts=sparse_candidate_shifts,
            input_metric=input_hessian, output_metric=output_hessian)


def refine_p32_candidates(candidates, **kwargs) -> GSQResult:
    """Backward-compatible P32 window entry point; never selects another layout."""
    return refine_trellis_candidates(candidates, layout="p32_window", **kwargs)


def refine_p32_fisher(baseline, **kwargs) -> GSQResult:
    """Backward-compatible P32 Fisher entry point."""
    return refine_trellis_fisher(baseline, layout="p32_window", **kwargs)


@torch.no_grad()
def batched_trellis_candidates(
    candidates, *, target, inputs, right_factor, bits, layout,
    bank_ids=None, bank_alt_id=None, codebook_version=PGC16_CODEBOOK_VERSION,
    sweeps=1, chunk_tiles=1024, decoded_candidates=None,
    input_metric=None, output_metric=None,
):
    """GPU-batched hard initializer with an exact full-objective guard.

    Candidate deltas are scored in tile chunks using the same Fisher quadratic
    as the sequential comparator.  A simultaneous proposal is accepted only
    after recomputing the complete objective; interacting proposals are backed
    off geometrically down to one tile.  This removes one Python dispatch and
    synchronization per tile while retaining the baseline/no-regression guard.
    """
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 1:
        raise ValueError("batched GSQ initializer requires positive integer sweeps")
    if isinstance(chunk_tiles, bool) or not isinstance(chunk_tiles, int) or chunk_tiles < 1:
        raise ValueError("batched GSQ chunk_tiles must be a positive integer")
    adapter = TrellisCandidateAdapter(layout, bits, codebook_version)
    if target.ndim != 2 or target.shape[0] % 16 or target.shape[1] % 16:
        raise ValueError("batched GSQ target must have tile-aligned [K,N] shape")
    k, n = target.shape
    tile_count = k * n // 256
    if candidates.ndim != 3 or candidates.shape[1] != tile_count or candidates.dtype != torch.int32:
        raise ValueError("batched GSQ candidates must be int32 [choices,tiles,words]")
    exact_fisher = input_metric is not None and output_metric is not None
    if inputs.ndim != 2 or inputs.shape[1] != k:
        raise ValueError("batched GSQ input factor must match target dimensions")
    if not exact_fisher and (right_factor is None or right_factor.shape != (n, n)):
        raise ValueError("batched GSQ right factor must match target dimensions")
    tensors = (target, inputs) if right_factor is None else (target, inputs, right_factor)
    if any(t.device != candidates.device or not t.is_floating_point() or not torch.isfinite(t).all()
           for t in tensors):
        raise ValueError("batched GSQ target/factors must be finite floating tensors on the candidate device")

    x, teacher = inputs.float(), target.float()
    right = None if right_factor is None else right_factor.float()
    if decoded_candidates is None:
        values = torch.stack([adapter.decode(c, bank_ids, bank_alt_id) for c in candidates]).reshape(
            candidates.shape[0], tile_count, 16, 16).float()
    else:
        expected = (candidates.shape[0], tile_count, 16, 16)
        if decoded_candidates.shape != expected or decoded_candidates.device != candidates.device:
            raise ValueError(f"decoded_candidates must have shape {expected} on the candidate device")
        values = decoded_candidates.detach().float()

    def metric_or_factor(metric, expected, fallback, name):
        if metric is None:
            return fallback()
        if (metric.shape != expected or metric.device != candidates.device
                or not metric.is_floating_point() or not torch.isfinite(metric).all()):
            raise ValueError(f"{name} must be finite floating point {expected} on the candidate device")
        return metric.detach().float()

    h = metric_or_factor(input_metric, (k, k), lambda: x.T @ x, "input_metric")
    g = metric_or_factor(output_metric, (n, n), lambda: right @ right.T, "output_metric")
    input_tiles, output_tiles = k // 16, n // 16
    input_ids = torch.arange(input_tiles, device=candidates.device)
    output_ids = torch.arange(output_tiles, device=candidates.device)
    h_blocks = h.reshape(input_tiles, 16, input_tiles, 16)[input_ids, :, input_ids, :]
    g_blocks = g.reshape(output_tiles, 16, output_tiles, 16)[output_ids, :, output_ids, :]
    if exact_fisher:
        teacher_metric = h @ teacher @ g
        normalizer = (teacher * teacher_metric).sum().clamp_min(torch.finfo(torch.float32).tiny)
    else:
        teacher_output = x @ teacher @ right
        normalizer = teacher_output.square().mean().clamp_min(torch.finfo(torch.float32).tiny)

    def dense(tiles):
        return tiles.reshape(input_tiles, output_tiles, 16, 16).permute(0, 2, 1, 3).reshape(k, n)

    def score(tiles):
        if exact_fisher:
            error = dense(tiles) - teacher
            return float((error * (h @ error @ g)).sum() / normalizer)
        return float(((x @ dense(tiles) @ right) - teacher_output).square().mean() / normalizer)

    current_tiles = values[0].clone()
    choices = torch.zeros(tile_count, device=candidates.device, dtype=torch.long)
    before = score(current_tiles)
    if not math.isfinite(before):
        raise ValueError("non-finite batched GSQ baseline objective")
    best, best_choices, best_tiles = before, choices.clone(), current_tiles.clone()
    history = [before]
    all_tile_ids = torch.arange(tile_count, device=candidates.device)

    for _ in range(sweeps):
        metric_error = h @ (dense(current_tiles) - teacher) @ g
        metric_tiles = metric_error.reshape(input_tiles, 16, output_tiles, 16).permute(
            0, 2, 1, 3).reshape(tile_count, 16, 16)
        proposed_choices = choices.clone()
        predicted_cost = torch.zeros(tile_count, device=candidates.device)
        for start in range(0, tile_count, chunk_tiles):
            stop = min(start + chunk_tiles, tile_count)
            ids = all_tile_ids[start:stop]
            ib, jb = torch.div(ids, output_tiles, rounding_mode="floor"), ids % output_tiles
            delta = values[:, start:stop] - current_tiles[start:stop].unsqueeze(0)
            cost = 2 * (delta * metric_tiles[start:stop].unsqueeze(0)).sum((2, 3))
            quadratic = torch.matmul(torch.matmul(h_blocks[ib].unsqueeze(0), delta),
                                     g_blocks[jb].unsqueeze(0))
            cost += (quadratic * delta).sum((2, 3))
            selected_cost, selected = cost.min(0)
            improving = selected_cost < 0
            proposed_choices[start:stop] = torch.where(improving, selected, choices[start:stop])
            predicted_cost[start:stop] = torch.where(improving, selected_cost, torch.zeros_like(selected_cost))

        changed = proposed_choices != choices
        if not bool(changed.any()):
            history.append(best)
            break
        proposal_tiles = current_tiles.clone()
        proposal_tiles[changed] = values[proposed_choices[changed], all_tile_ids[changed]]
        value = score(proposal_tiles)
        accepted_choices, accepted_tiles = proposed_choices, proposal_tiles
        if not math.isfinite(value):
            raise ValueError("non-finite batched GSQ hard objective")

        if value >= best:
            ranked = torch.nonzero(changed, as_tuple=False).flatten()
            ranked = ranked[predicted_cost[ranked].argsort()]
            accepted_choices, accepted_tiles, value = None, None, best
            keep = len(ranked) // 2
            while keep >= 1:
                selected_ids = ranked[:keep]
                trial_choices = choices.clone()
                trial_choices[selected_ids] = proposed_choices[selected_ids]
                trial_tiles = current_tiles.clone()
                trial_tiles[selected_ids] = values[trial_choices[selected_ids], selected_ids]
                trial_value = score(trial_tiles)
                if not math.isfinite(trial_value):
                    raise ValueError("non-finite batched GSQ backoff objective")
                if trial_value < best:
                    accepted_choices, accepted_tiles, value = trial_choices, trial_tiles, trial_value
                    break
                keep //= 2
            if accepted_choices is None:
                history.append(best)
                break

        choices, current_tiles = accepted_choices, accepted_tiles
        history.append(value)
        if value < best:
            best, best_choices, best_tiles = value, choices.clone(), current_tiles.clone()

    del best_tiles  # The packed payload is selected directly from the legal candidate bank.
    return GSQResult(
        candidates[best_choices, all_tile_ids].clone().contiguous(), best_choices,
        before, best, history,
        {"initializer": "batched_guarded", "chunk_tiles": chunk_tiles},
    )


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
    teacher_output = x @ teacher @ right
    normalizer = teacher_output.square().mean().clamp_min(torch.finfo(torch.float32).tiny)

    def score(weight):
        # Match GSQ operation order, including FP32 matmul cancellation.
        return float(((x @ weight @ right) - teacher_output).square().mean() / normalizer)

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
