"""Differentiable staged-GSQ adapter for legal QVQ P32 candidate banks.

The soft path is a categorical mixture of exact one-edge P32 alternatives.
Only hard, round-trippable payloads are exported. Rebuilding a bank from an
accepted payload permits several rounds to compose edits without ever treating
an inconsistent collection of overlapping windows as a serialized weight.
"""

import math

import torch

from .qvq import (
    repack_p32_planar_to_window,
    rht_preprocess_weight,
    rht_reconstruct_weight,
)
from .qvq_gsq import (
    TrellisCandidateAdapter,
    _candidate_probabilities,
    fisher_screened_trellis_candidates,
)
from .rotation.hadamard_utils import matmul_hadU


def _rht_reconstruct_differentiable(
    inner_weight,
    SU,
    SV,
    *,
    input_hadamard=True,
    output_hadamard=True,
):
    """Unchecked differentiable form of ``rht_reconstruct_weight``.

    Shapes and finiteness are validated once by the owning module. Avoiding the
    public helper's runtime guards here prevents device synchronizations on
    every training microbatch.
    """
    work = inner_weight
    if input_hadamard:
        work = matmul_hadU(work.transpose(0, 1), transpose=True).transpose(0, 1)
    work = work * SU.to(work.dtype).unsqueeze(1)
    if output_hadamard:
        work = matmul_hadU(work)
    work = work * SV.to(work.dtype).unsqueeze(0)
    return work.transpose(0, 1).contiguous()


class GSQP32TrainingModule(torch.nn.Module):
    """Train one legal P32 whole-tile candidate bank in a model-level stage.

    This is deliberately narrower than scalar GSQ. One round can select at
    most one legal edge edit per tile. Callers may compose edits by exporting
    ``hard_state()``, rebuilding candidates from those words, and starting a
    new round. The relaxed weight is used only for gradients and is never
    serialized.
    """

    def __init__(
        self,
        candidates,
        baseline_tiles,
        sparse_indices,
        sparse_deltas,
        sparse_shifts,
        *,
        bits,
        bank_ids,
        bank_alt_id,
        in_features,
        out_features,
        SU,
        SV,
        seed=7,
        std=.01,
        strength=6.,
        logits_dtype=torch.float32,
        input_hadamard=True,
        output_hadamard=True,
    ):
        super().__init__()
        choices, tile_count, _ = candidates.shape
        expected_tiles = (in_features // 16) * (out_features // 16)
        expected_sparse = (choices - 1, tile_count)
        if (
            candidates.dtype != torch.int32
            or candidates.ndim != 3
            or choices < 2
            or tile_count != expected_tiles
        ):
            raise ValueError("P32 GSQ candidates must be int32 [choices,tiles,words]")
        if baseline_tiles.shape != (tile_count, 16, 16):
            raise ValueError("P32 GSQ baseline tiles do not match candidate geometry")
        if (
            sparse_indices.ndim != 3
            or sparse_indices.shape[:2] != expected_sparse
            or sparse_deltas.shape != sparse_indices.shape
            or sparse_shifts.shape != expected_sparse
        ):
            raise ValueError("P32 GSQ sparse candidate metadata does not match candidates")
        if sparse_indices.dtype != torch.int64 or sparse_shifts.dtype != torch.int64:
            raise ValueError("P32 GSQ sparse indices and shifts must be int64")
        if not sparse_deltas.is_floating_point() or not baseline_tiles.is_floating_point():
            raise ValueError("P32 GSQ decoded values must be floating point")
        if any(d % 16 for d in (in_features, out_features)):
            raise ValueError("P32 GSQ dimensions must be divisible by 16")
        if SU.shape != (in_features,) or SV.shape != (out_features,):
            raise ValueError("P32 GSQ transform scales do not match projection dimensions")
        tensors = (
            candidates,
            baseline_tiles,
            sparse_indices,
            sparse_deltas,
            sparse_shifts,
            bank_ids,
            bank_alt_id,
            SU,
            SV,
        )
        if any(value.device != candidates.device for value in tensors):
            raise ValueError("P32 GSQ training tensors must share one device")
        if not all(torch.isfinite(value).all() for value in (baseline_tiles, sparse_deltas, SU, SV)):
            raise ValueError("P32 GSQ training tensors must be finite")
        if not math.isfinite(std) or std <= 0 or not math.isfinite(strength) or strength < 0:
            raise ValueError("P32 GSQ initialization controls are invalid")

        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features
        self.input_hadamard = input_hadamard
        self.output_hadamard = output_hadamard
        self.adapter = TrellisCandidateAdapter("p32_window", bits)
        self.register_buffer("candidates", candidates.detach().clone().contiguous())
        self.register_buffer("baseline_tiles", baseline_tiles.detach().reshape(tile_count, 256).clone())
        # Training is tile-major so one scatter builds the complete inner matrix.
        self.register_buffer("sparse_indices", sparse_indices.permute(1, 0, 2).contiguous())
        self.register_buffer("sparse_deltas", sparse_deltas.permute(1, 0, 2).contiguous())
        self.register_buffer("sparse_shifts", sparse_shifts.T.contiguous())
        self.register_buffer("bank_ids", bank_ids.detach().clone().contiguous())
        self.register_buffer("bank_alt_id", bank_alt_id.detach().clone().contiguous())
        self.register_buffer("SU", SU.detach().clone().contiguous())
        self.scales = torch.nn.Parameter(SV.detach().float().clone().contiguous())

        shifts = torch.nn.functional.pad(self.sparse_shifts, (1, 0)).float()
        prior = -.5 * shifts.square()
        prior -= prior.mean(-1, keepdim=True)
        generator = torch.Generator(device=candidates.device).manual_seed(seed)
        noise = torch.randn(prior.shape, device=prior.device, generator=generator)
        logits = std * (noise + strength * prior)
        # The relaxed initialization remains the paper's shift-centred
        # Gaussian, but its hard projection must be the serialized baseline.
        # Otherwise a new composition round silently starts from random legal
        # edits and the held-out rollback compares against the wrong model.
        logits[:, 0] = torch.maximum(logits[:, 0], logits[:, 1:].amax(-1) + std * 1e-3)
        self.logits = torch.nn.Parameter(logits.to(logits_dtype))

    def _inner_from_probabilities(self, probabilities):
        if probabilities.shape != self.logits.shape:
            raise ValueError("P32 GSQ probabilities do not match logits")
        contributions = probabilities[:, 1:, None].to(self.sparse_deltas.dtype) * self.sparse_deltas
        tiles = self.baseline_tiles.clone().scatter_add(
            1,
            self.sparse_indices.flatten(1),
            contributions.flatten(1),
        )
        return (
            tiles.reshape(self.in_features // 16, self.out_features // 16, 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(self.in_features, self.out_features)
            .contiguous()
        )

    def forward(self, *, uniform, temperature, multiplier):
        probabilities = _candidate_probabilities(
            self.logits,
            uniform.clamp(1e-6, 1 - 1e-6),
            temperature,
            multiplier,
        )
        inner = self._inner_from_probabilities(probabilities)
        return _rht_reconstruct_differentiable(
            inner,
            self.SU,
            self.scales,
            input_hadamard=self.input_hadamard,
            output_hadamard=self.output_hadamard,
        )

    @torch.no_grad()
    def hard_state(self):
        choices = self.logits.argmax(-1)
        tiles = torch.arange(len(choices), device=choices.device)
        words = self.candidates[choices, tiles].contiguous()
        if not torch.equal(self.adapter.pack(self.adapter.unpack(words)), words):
            raise RuntimeError("staged P32 GSQ produced a payload that does not round-trip")
        return {
            "words": words,
            "choices": choices,
            "SV": self.scales.detach().clone(),
        }

    @torch.no_grad()
    def hard_weight(self):
        state = self.hard_state()
        inner = self.adapter.inner(
            state["words"],
            self.in_features,
            self.out_features,
            self.bank_ids,
            self.bank_alt_id,
        )
        return rht_reconstruct_weight(
            inner,
            self.SU,
            state["SV"],
            input_hadamard=self.input_hadamard,
            output_hadamard=self.output_hadamard,
        )

    def optimizer_groups(self, *, assignment_lr, scale_lr, weight_decay):
        return [
            {"params": [self.logits], "lr": assignment_lr, "weight_decay": weight_decay},
            {"params": [self.scales], "lr": scale_lr, "weight_decay": 0.},
        ]


def p32_training_module_from_payload(
    trellis,
    SU,
    SV,
    bank_ids,
    bank_alt_id,
    teacher_weight,
    *,
    candidates=33,
    seed=7,
    input_metric=None,
    output_metric=None,
    input_hadamard=True,
    output_hadamard=True,
):
    """Build a staged module from one serialized W3/P32 QVQ projection."""
    if teacher_weight.ndim != 2 or teacher_weight.device != trellis.device:
        raise ValueError("P32 staged teacher weight must be rank-2 on the payload device")
    baseline = repack_p32_planar_to_window(trellis, bits=3)
    return p32_training_module_from_words(
        baseline,
        SU,
        SV,
        bank_ids,
        bank_alt_id,
        teacher_weight,
        candidates=candidates,
        seed=seed,
        input_metric=input_metric,
        output_metric=output_metric,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )


def p32_training_module_from_words(
    baseline,
    SU,
    SV,
    bank_ids,
    bank_alt_id,
    teacher_weight,
    *,
    candidates=33,
    seed=7,
    input_metric=None,
    output_metric=None,
    input_hadamard=True,
    output_hadamard=True,
):
    """Build the next legal staged round from accepted P32 window words.

    ``baseline`` must be the exact hard state accepted by the preceding round.
    Choice zero therefore means no additional edit, which lets held-out
    checkpoint restoration reject a whole round without undoing earlier gains.
    """
    if teacher_weight.ndim != 2 or teacher_weight.device != baseline.device:
        raise ValueError("P32 staged teacher weight must be rank-2 on the payload device")
    if baseline.dtype != torch.int32 or baseline.ndim != 2:
        raise ValueError("P32 staged baseline must be rank-2 int32 window words")
    adapter = TrellisCandidateAdapter("p32_window", 3)
    if not torch.equal(adapter.pack(adapter.unpack(baseline)), baseline):
        raise ValueError("P32 staged baseline must round-trip as legal window words")
    out_features, in_features = teacher_weight.shape
    expected_tiles = (in_features // 16) * (out_features // 16)
    if len(baseline) != expected_tiles:
        raise ValueError("P32 staged baseline tile count does not match teacher weight")
    target = rht_preprocess_weight(
        teacher_weight,
        SU.reciprocal(),
        SV.reciprocal(),
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    ).float()
    if input_metric is None:
        input_metric = torch.eye(in_features, device=baseline.device)
    if output_metric is None:
        output_metric = torch.eye(out_features, device=baseline.device)
    values = fisher_screened_trellis_candidates(
        baseline,
        count=candidates,
        seed=seed,
        bits=3,
        layout="p32_window",
        target=target,
        input_hessian=input_metric,
        output_hessian=output_metric,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
        return_decoded=True,
        return_sparse=True,
        return_shifts=True,
    )
    candidate_words, decoded, indices, deltas, shifts = values
    return GSQP32TrainingModule(
        candidate_words,
        decoded[0],
        indices,
        deltas,
        shifts,
        bits=3,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
        in_features=in_features,
        out_features=out_features,
        SU=SU,
        SV=SV,
        seed=seed,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )


@torch.no_grad()
def p32_payload_weight(
    trellis,
    SU,
    SV,
    bank_ids,
    bank_alt_id,
    *,
    in_features,
    out_features,
    input_hadamard=True,
    output_hadamard=True,
):
    """Materialize the exact dense weight represented by a W3/P32 payload."""
    adapter = TrellisCandidateAdapter("p32_window", 3)
    words = repack_p32_planar_to_window(trellis, bits=3)
    inner = adapter.inner(words, in_features, out_features, bank_ids, bank_alt_id)
    return rht_reconstruct_weight(
        inner,
        SU,
        SV,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
