"""Differentiable staged-GSQ adapter for legal QVQ P32 candidate banks.

The soft path is a categorical mixture of exact one-edge P32 alternatives.
Only hard, round-trippable payloads are exported. Rebuilding a bank from an
accepted payload permits several rounds to compose edits without ever treating
an inconsistent collection of overlapping windows as a serialized weight.
"""

import math

import torch

from ..utils.hadamard import (
    hadamard_available,
    hadamard_transform,
    hadamard_transform_reverse,
    hadamard_transform_reverse_scaled,
    hadamard_transform_scaled,
    hadamard_transform_scaled_saved,
)
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


class _ExactTrainingHadamard(torch.autograd.Function):
    """Fused normalized Hadamard with the eager path's exact FP32 gradient.

    The native kernel evaluates butterfly stages in forward order. Autograd
    traverses the eager reference's stages in reverse order, which changes
    FP32 rounding despite the Walsh-Hadamard matrix being self-adjoint. The
    backward therefore uses the native descending-stage kernel.
    """

    @staticmethod
    def forward(ctx, values):
        width = values.shape[-1]
        ctx.width = width
        return hadamard_transform(values.contiguous(), 1. / math.sqrt(width))

    @staticmethod
    def backward(ctx, grad_output):
        return hadamard_transform_reverse(
            grad_output.contiguous(), 1. / math.sqrt(ctx.width),
        )


class _ExactTrainingHadamardFixedScale(torch.autograd.Function):
    """Fuse a fixed vector scale while preserving both dtype boundaries."""

    @staticmethod
    def forward(ctx, values, vector):
        width = values.shape[-1]
        ctx.width = width
        ctx.save_for_backward(vector)
        return hadamard_transform_scaled(
            values.contiguous(), vector, 1. / math.sqrt(width),
        )

    @staticmethod
    def backward(ctx, grad_output):
        vector, = ctx.saved_tensors
        return hadamard_transform_reverse_scaled(
            grad_output.contiguous(), vector, 1. / math.sqrt(ctx.width),
        ), None


class _ExactTrainingHadamardTrainableScale(torch.autograd.Function):
    """Fuse a trainable scale while retaining its exact reduction input."""

    @staticmethod
    def forward(ctx, values, vector):
        width = values.shape[-1]
        ctx.width = width
        scaled, unscaled = hadamard_transform_scaled_saved(
            values.contiguous(), vector, 1. / math.sqrt(width),
        )
        ctx.save_for_backward(vector, unscaled)
        return scaled

    @staticmethod
    def backward(ctx, grad_output):
        vector, unscaled = ctx.saved_tensors
        grad_values = hadamard_transform_reverse_scaled(
            grad_output.contiguous(), vector, 1. / math.sqrt(ctx.width),
        )
        grad_vector = (grad_output * unscaled).sum(0)
        return grad_values, grad_vector


def _training_hadamard(values, fast_hadamard):
    if not fast_hadamard:
        return matmul_hadU(values)
    return _ExactTrainingHadamard.apply(values)


class _SparseCandidateMatrixMixture(torch.autograd.Function):
    """Exact P32 mixture directly in the Hadamard matrix layout."""

    @staticmethod
    def forward(ctx, probabilities, baseline_matrix, matrix_indices, sparse_deltas):
        contributions = probabilities[:, 1:, None].to(sparse_deltas.dtype) * sparse_deltas
        matrix = baseline_matrix.flatten().clone().scatter_add(
            0, matrix_indices.flatten(), contributions.flatten(),
        ).reshape_as(baseline_matrix)
        ctx.save_for_backward(matrix_indices, sparse_deltas)
        ctx.probability_dtype = probabilities.dtype
        return matrix

    @staticmethod
    def backward(ctx, grad_matrix):
        matrix_indices, sparse_deltas = ctx.saved_tensors
        selected = grad_matrix.flatten().gather(0, matrix_indices.flatten()).reshape_as(sparse_deltas)
        candidate_gradient = (selected * sparse_deltas).sum(-1)
        probability_gradient = torch.nn.functional.pad(candidate_gradient, (1, 0))
        return probability_gradient.to(ctx.probability_dtype), None, None, None


class _FusedSparseCandidateMatrixMixture(torch.autograd.Function):
    """Exact P32 mixture with fused CUDA materialization and adjoint."""

    @staticmethod
    def forward(ctx, probabilities, baseline_matrix, matrix_indices, sparse_deltas,
                baseline_tiles, position_indices, position_choices, position_deltas,
                compact_forward, output_dtype, transposed_output):
        from .qvq_gsq_triton import compact_sparse_mixture

        if compact_forward:
            shape = (baseline_matrix.shape[1], baseline_matrix.shape[0]) if transposed_output else baseline_matrix.shape
            matrix = torch.empty(shape, dtype=output_dtype, device=baseline_matrix.device)
            compact_sparse_mixture(
                probabilities, baseline_tiles, position_indices,
                position_choices, position_deltas, matrix,
                in_features=baseline_matrix.shape[0],
                out_features=baseline_matrix.shape[1],
                transposed=transposed_output,
            )
        else:
            contributions = probabilities[:, 1:, None].to(sparse_deltas.dtype) * sparse_deltas
            matrix = baseline_matrix.flatten().clone().scatter_add(
                0, matrix_indices.flatten(), contributions.flatten(),
            ).reshape_as(baseline_matrix)
        ctx.save_for_backward(matrix_indices, sparse_deltas)
        ctx.probability_dtype = probabilities.dtype
        return matrix

    @staticmethod
    def backward(ctx, grad_matrix):
        from .qvq_gsq_triton import candidate_probability_gradient

        matrix_indices, sparse_deltas = ctx.saved_tensors
        probability_gradient = torch.empty(
            (matrix_indices.shape[0], matrix_indices.shape[1] + 1),
            dtype=ctx.probability_dtype,
            device=grad_matrix.device,
        )
        candidate_probability_gradient(
            grad_matrix.contiguous(), matrix_indices, sparse_deltas,
            probability_gradient,
        )
        return probability_gradient, None, None, None, None, None, None, None, None, None, None


class _TransposeView(torch.autograd.Function):
    """Rank-two transpose view with its exact transpose adjoint.

    The caller immediately converts reconstructed FP32 weights to the model's
    BF16 dtype. Keeping this as a view lets that conversion consume the
    transposed layout directly instead of copying a full FP32 matrix first.
    """

    @staticmethod
    def forward(ctx, values):
        return values.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.transpose(0, 1)


def _rht_reconstruct_differentiable(
    inner_weight,
    SU,
    SV,
    *,
    input_hadamard=True,
    output_hadamard=True,
    fast_hadamard=False,
    inner_transposed=False,
):
    """Unchecked differentiable form of ``rht_reconstruct_weight``.

    Shapes and finiteness are validated once by the owning module. Avoiding the
    public helper's runtime guards here prevents device synchronizations on
    every training microbatch.
    """
    work = inner_weight
    if input_hadamard:
        if not fast_hadamard:
            work = matmul_hadU(work.transpose(0, 1), transpose=True).transpose(0, 1)
        else:
            work = _ExactTrainingHadamardFixedScale.apply(
                work if inner_transposed else work.transpose(0, 1),
                SU.to(work.dtype),
            ).transpose(0, 1)
    if not (input_hadamard and fast_hadamard):
        work = work * SU.to(work.dtype).unsqueeze(1)
    if output_hadamard and fast_hadamard:
        work = _ExactTrainingHadamardTrainableScale.apply(
            work, SV.to(work.dtype),
        )
    else:
        if output_hadamard:
            work = _training_hadamard(work, fast_hadamard)
        work = work * SV.to(work.dtype).unsqueeze(0)
    return _TransposeView.apply(work)


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
        sparse_values,
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
        fast_hadamard=True,
        training_dtype=torch.float32,
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
            or sparse_values.shape != sparse_indices.shape
            or sparse_shifts.shape != expected_sparse
        ):
            raise ValueError("P32 GSQ sparse candidate metadata does not match candidates")
        if sparse_indices.dtype != torch.int64 or sparse_shifts.dtype != torch.int64:
            raise ValueError("P32 GSQ sparse indices and shifts must be int64")
        if (not sparse_deltas.is_floating_point()
                or not sparse_values.is_floating_point()
                or not baseline_tiles.is_floating_point()):
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
            sparse_values,
            sparse_shifts,
            bank_ids,
            bank_alt_id,
            SU,
            SV,
        )
        if any(value.device != candidates.device for value in tensors):
            raise ValueError("P32 GSQ training tensors must share one device")
        if not all(torch.isfinite(value).all() for value in (
            baseline_tiles, sparse_deltas, sparse_values, SU, SV,
        )):
            raise ValueError("P32 GSQ training tensors must be finite")
        if not math.isfinite(std) or std <= 0 or not math.isfinite(strength) or strength < 0:
            raise ValueError("P32 GSQ initialization controls are invalid")
        if training_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("P32 GSQ training dtype must be float32 or bfloat16")
        if training_dtype == torch.bfloat16 and candidates.device.type != "cuda":
            raise ValueError("P32 GSQ bfloat16 training requires CUDA")

        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features
        self.input_hadamard = input_hadamard
        self.output_hadamard = output_hadamard
        self.training_dtype = training_dtype
        fast_widths = {
            width
            for enabled, width in (
                (input_hadamard, in_features),
                (output_hadamard, out_features),
            )
            if enabled
        }
        fast_compatible = (
            candidates.device.type == "cuda"
            and baseline_tiles.dtype == torch.float32
            and all(width >= 8 and width <= 32768 and not width & (width - 1)
                    for width in fast_widths)
        )
        self.fast_hadamard = bool(
            fast_hadamard and fast_compatible and hadamard_available()
        )
        self.training_hadamard_backend = (
            "fused_cuda_exact" if self.fast_hadamard else "eager"
        )
        self.compact_forward = max(in_features, out_features) > 2048
        self.adapter = TrellisCandidateAdapter("p32_window", bits)
        self.register_buffer("candidates", candidates.detach().clone().contiguous())
        self.register_buffer("baseline_tiles", baseline_tiles.detach().reshape(tile_count, 256).clone())
        # Candidate metadata is tile-major; matrix indices below remove the
        # tile-layout permutation from every differentiable reconstruction.
        self.register_buffer("sparse_indices", sparse_indices.permute(1, 0, 2).contiguous())
        self.register_buffer("sparse_deltas", sparse_deltas.permute(1, 0, 2).contiguous())
        self.register_buffer("sparse_values", sparse_values.permute(1, 0, 2).contiguous())
        input_tiles = in_features // 16
        output_tiles = out_features // 16
        tile_ids = torch.arange(tile_count, device=candidates.device)
        input_tile_ids = tile_ids // output_tiles
        output_tile_ids = tile_ids % output_tiles
        local_indices = self.sparse_indices
        matrix_indices = (
            (input_tile_ids[:, None, None] * 16 + local_indices // 16) * out_features
            + output_tile_ids[:, None, None] * 16
            + local_indices % 16
        )
        self.register_buffer("matrix_sparse_indices", matrix_indices.contiguous())
        transposed_indices = (
            (matrix_indices % out_features) * in_features
            + matrix_indices // out_features
        )
        self.register_buffer(
            "matrix_sparse_indices_transposed", transposed_indices.contiguous(),
        )
        if self.compact_forward:
            from .qvq_gsq_triton import (
                build_compact_position_map,
                transpose_compact_position_map,
            )

            position_indices, position_choices, position_deltas = build_compact_position_map(
                self.sparse_indices, self.sparse_deltas, matrix_indices,
            )
            (transposed_position_indices, transposed_position_choices,
             transposed_position_deltas) = transpose_compact_position_map(
                position_indices, position_choices, position_deltas,
            )
        else:
            position_indices = torch.empty(
                (0, 0), dtype=torch.uint8, device=candidates.device,
            )
            position_choices = torch.empty(
                (0, 0, 0), dtype=torch.uint8, device=candidates.device,
            )
            position_deltas = torch.empty(
                (0, 0, 0), dtype=self.sparse_deltas.dtype, device=candidates.device,
            )
            transposed_position_indices = position_indices
            transposed_position_choices = position_choices
            transposed_position_deltas = position_deltas
        self.register_buffer("position_indices", position_indices)
        self.register_buffer("position_choices", position_choices)
        self.register_buffer("position_deltas", position_deltas)
        self.register_buffer("transposed_position_indices", transposed_position_indices)
        self.register_buffer("transposed_position_choices", transposed_position_choices)
        self.register_buffer("transposed_position_deltas", transposed_position_deltas)
        baseline_matrix = (
            self.baseline_tiles.reshape(input_tiles, output_tiles, 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(in_features, out_features)
            .contiguous()
        )
        self.register_buffer("baseline_matrix", baseline_matrix)
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

    def _inner_from_probabilities(self, probabilities, output_dtype=None,
                                  transposed_output=False):
        if probabilities.shape != self.logits.shape:
            raise ValueError("P32 GSQ probabilities do not match logits")
        if output_dtype is None:
            output_dtype = self.baseline_matrix.dtype
        if transposed_output and not self.compact_forward:
            raise ValueError("transposed P32 mixtures require compact materialization")
        if probabilities.device.type == "cuda":
            matrix_indices = (
                self.matrix_sparse_indices_transposed
                if transposed_output else self.matrix_sparse_indices
            )
            position_indices = (
                self.transposed_position_indices
                if transposed_output else self.position_indices
            )
            position_choices = (
                self.transposed_position_choices
                if transposed_output else self.position_choices
            )
            position_deltas = (
                self.transposed_position_deltas
                if transposed_output else self.position_deltas
            )
            return _FusedSparseCandidateMatrixMixture.apply(
                probabilities,
                self.baseline_matrix,
                matrix_indices,
                self.sparse_deltas,
                self.baseline_tiles,
                position_indices,
                position_choices,
                position_deltas,
                self.compact_forward,
                output_dtype,
                transposed_output,
            )
        return _SparseCandidateMatrixMixture.apply(
            probabilities,
            self.baseline_matrix,
            self.matrix_sparse_indices,
            self.sparse_deltas,
        )

    def forward(self, *, uniform, temperature, multiplier):
        probabilities = _candidate_probabilities(
            self.logits,
            uniform.clamp(1e-6, 1 - 1e-6),
            temperature,
            multiplier,
        )
        inner_transposed = (
            self.compact_forward and self.input_hadamard and self.fast_hadamard
        )
        inner = self._inner_from_probabilities(
            probabilities, self.training_dtype,
            transposed_output=inner_transposed,
        )
        return _rht_reconstruct_differentiable(
            inner,
            self.SU,
            self.scales,
            input_hadamard=self.input_hadamard,
            output_hadamard=self.output_hadamard,
            fast_hadamard=self.fast_hadamard,
            inner_transposed=inner_transposed,
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

    @torch.no_grad()
    def hard_weight_for_evaluation(self):
        """Materialize a hard training checkpoint from its validated bank.

        Candidate construction proves that every row in ``self.candidates``
        is a legal round-trippable P32 payload. Repeating that payload audit at
        every held-out checkpoint does not strengthen the invariant and is
        particularly expensive for the 2,000-update Q/K stages. The public
        ``hard_state`` and ``hard_weight`` paths retain the audit for export.
        """
        choices = self.logits.argmax(-1)
        tile_ids = torch.arange(len(choices), device=choices.device)
        candidate_ids = (choices - 1).clamp_min(0)
        indices = self.sparse_indices[tile_ids, candidate_ids]
        matrix_indices = self.matrix_sparse_indices[tile_ids, candidate_ids]
        selected_values = self.sparse_values[tile_ids, candidate_ids]
        baseline_values = self.baseline_tiles.gather(1, indices)
        values = torch.where(choices[:, None] == 0, baseline_values, selected_values)
        inner = self.baseline_matrix.flatten().clone().scatter_(
            0, matrix_indices.flatten(), values.flatten(),
        ).reshape_as(self.baseline_matrix)
        return _rht_reconstruct_differentiable(
            inner,
            self.SU,
            self.scales,
            input_hadamard=self.input_hadamard,
            output_hadamard=self.output_hadamard,
            fast_hadamard=self.fast_hadamard,
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
    fast_hadamard=True,
    training_dtype=torch.float32,
    fast_identity_metric=True,
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
        fast_hadamard=fast_hadamard,
        training_dtype=training_dtype,
        fast_identity_metric=fast_identity_metric,
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
    fast_hadamard=True,
    training_dtype=torch.float32,
    fast_identity_metric=True,
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
    identity_metric = (
        fast_identity_metric and input_metric is None and output_metric is None
    )
    if not identity_metric:
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
        identity_metric=identity_metric,
    )
    candidate_words, decoded, indices, deltas, shifts = values
    sparse_values = decoded[1:].reshape(len(decoded) - 1, len(baseline), 256).gather(
        2, indices,
    )
    return GSQP32TrainingModule(
        candidate_words,
        decoded[0],
        indices,
        deltas,
        sparse_values,
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
        fast_hadamard=fast_hadamard,
        training_dtype=training_dtype,
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
