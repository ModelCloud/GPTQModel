import pytest
import torch

from gptqmodel.quantization.gsq_training_qvq import (
    GSQP32TrainingModule,
    _ContiguousTranspose,
    _rht_reconstruct_differentiable,
    _SparseCandidateMatrixMixture,
    p32_training_module_from_words,
)
from gptqmodel.quantization.qvq import rht_reconstruct_weight
from gptqmodel.quantization.qvq_gsq import (
    TrellisCandidateAdapter,
    fisher_screened_trellis_candidates,
)


def _training_module(device="cpu"):
    bits = 3
    baseline = torch.zeros((1, 24), dtype=torch.int32, device=device)
    bank = torch.zeros(1, dtype=torch.uint8, device=device)
    alt = torch.ones(1, dtype=torch.uint8, device=device)
    adapter = TrellisCandidateAdapter("p32_window", bits)
    teacher = adapter.inner(baseline, 16, 16, bank, alt)
    candidates, decoded, indices, deltas, shifts = fisher_screened_trellis_candidates(
        baseline,
        count=5,
        seed=7,
        bits=bits,
        layout="p32_window",
        target=teacher + torch.eye(16, device=device),
        input_hessian=torch.eye(16, device=device),
        output_hessian=torch.eye(16, device=device),
        bank_ids=bank,
        bank_alt_id=alt,
        return_decoded=True,
        return_sparse=True,
        return_shifts=True,
    )
    module = GSQP32TrainingModule(
        candidates,
        decoded[0],
        indices,
        deltas,
        decoded[1:].reshape(len(decoded) - 1, 1, 256).gather(2, indices),
        shifts,
        bits=bits,
        bank_ids=bank,
        bank_alt_id=alt,
        in_features=16,
        out_features=16,
        SU=torch.ones(16, device=device),
        SV=torch.ones(16, device=device),
        seed=7,
    )
    return module, candidates, decoded


def test_p32_staged_soft_weight_has_assignment_and_scale_gradients():
    module, _, decoded = _training_module()
    assert torch.equal(module.logits.argmax(-1), torch.zeros(len(module.logits), dtype=torch.long))
    uniform = torch.full_like(module.logits, .5)
    weight = module(uniform=uniform, temperature=2., multiplier=100.)
    loss = torch.nn.functional.mse_loss(weight, decoded[1].reshape(16, 16).T)
    loss.backward()
    assert module.logits.grad is not None and torch.isfinite(module.logits.grad).all()
    assert module.logits.grad.abs().sum() > 0
    assert module.scales.grad is not None and torch.isfinite(module.scales.grad).all()
    assert module.scales.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
)])
def test_explicit_layout_adjoints_are_bitwise_exact(device):
    generator = torch.Generator(device=device).manual_seed(23)
    tiles_reference = torch.randn((6, 256), device=device, generator=generator,
                                  requires_grad=True)
    tiles_exact = tiles_reference.detach().clone().requires_grad_()
    upstream = torch.randn((32, 48), device=device, generator=generator)
    reference = (
        tiles_reference.reshape(2, 3, 16, 16).permute(0, 2, 1, 3)
        .reshape(32, 48).contiguous()
    )
    exact = (
        tiles_exact.reshape(2, 3, 16, 16).permute(0, 2, 1, 3)
        .reshape(32, 48).contiguous()
    )
    reference.backward(upstream)
    exact.backward(upstream)
    assert torch.equal(exact, reference)
    assert torch.equal(tiles_exact.grad, tiles_reference.grad)

    matrix_reference = torch.randn((32, 48), device=device, generator=generator,
                                   requires_grad=True)
    matrix_exact = matrix_reference.detach().clone().requires_grad_()
    transpose_upstream = torch.randn((48, 32), device=device, generator=generator)
    reference = matrix_reference.transpose(0, 1).contiguous()
    exact = _ContiguousTranspose.apply(matrix_exact)
    reference.backward(transpose_upstream)
    exact.backward(transpose_upstream)
    assert torch.equal(exact, reference)
    assert torch.equal(matrix_exact.grad, matrix_reference.grad)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matrix_layout_sparse_mixture_is_bitwise_exact_for_real_p32_metadata():
    module, _, _ = _training_module("cuda")
    generator = torch.Generator(device="cuda").manual_seed(29)
    probabilities_reference = torch.randn(
        module.logits.shape, device="cuda", generator=generator, requires_grad=True,
    )
    probabilities_matrix = probabilities_reference.detach().clone().requires_grad_()
    upstream_tiles = torch.randn(module.baseline_tiles.shape, device="cuda", generator=generator)
    upstream_matrix = upstream_tiles.reshape(16, 16)

    contributions = probabilities_reference[:, 1:, None] * module.sparse_deltas
    tile_output = module.baseline_tiles.clone().scatter_add(
        1, module.sparse_indices.flatten(1), contributions.flatten(1),
    )
    reference = tile_output.reshape(16, 16)
    matrix = _SparseCandidateMatrixMixture.apply(
        probabilities_matrix, module.baseline_matrix,
        module.matrix_sparse_indices, module.sparse_deltas,
    )
    reference.backward(upstream_matrix)
    matrix.backward(upstream_matrix)

    assert torch.equal(matrix, reference)
    assert torch.equal(probabilities_matrix.grad, probabilities_reference.grad)


def test_p32_staged_hard_state_is_exact_legal_candidate():
    module, candidates, _ = _training_module()
    for choice in range(len(candidates)):
        with torch.no_grad():
            module.logits.fill_(-1)
            module.logits[:, choice] = 1
        state = module.hard_state()
        assert torch.equal(state["words"], candidates[choice])
        assert torch.equal(module.adapter.pack(module.adapter.unpack(state["words"])), state["words"])
        expected_inner = module.adapter.inner(
            candidates[choice], 16, 16, module.bank_ids, module.bank_alt_id,
        )
        torch.testing.assert_close(
            module.hard_weight(),
            rht_reconstruct_weight(expected_inner, module.SU, module.scales),
        )
        assert torch.equal(module.hard_weight_for_evaluation(), module.hard_weight())


def test_p32_staged_rejects_inconsistent_sparse_metadata():
    module, candidates, decoded = _training_module()
    with pytest.raises(ValueError, match="metadata"):
        GSQP32TrainingModule(
            candidates,
            decoded[0],
            module.sparse_indices.permute(1, 0, 2)[:, :, :-1],
            module.sparse_deltas.permute(1, 0, 2),
            module.sparse_values.permute(1, 0, 2),
            module.sparse_shifts.T,
            bits=3,
            bank_ids=module.bank_ids,
            bank_alt_id=module.bank_alt_id,
            in_features=16,
            out_features=16,
            SU=module.SU,
            SV=module.scales,
        )


def test_p32_next_round_uses_prior_hard_words_and_scales_as_baseline():
    first, _, _ = _training_module()
    with torch.no_grad():
        first.logits.fill_(-1)
        first.logits[:, 1] = 1
        first.scales.add_(.125)
    accepted = first.hard_state()
    teacher = first.hard_weight()
    second = p32_training_module_from_words(
        accepted["words"],
        first.SU,
        accepted["SV"],
        first.bank_ids,
        first.bank_alt_id,
        teacher,
        candidates=5,
        seed=8,
    )
    with torch.no_grad():
        second.logits.fill_(-1)
        second.logits[:, 0] = 1
    state = second.hard_state()
    assert torch.equal(state["words"], accepted["words"])
    torch.testing.assert_close(state["SV"], accepted["SV"])
    torch.testing.assert_close(second.hard_weight(), teacher)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_training_hadamard_is_bitwise_exact_forward_and_backward():
    generator = torch.Generator(device="cuda").manual_seed(41)
    # Llama 3.2 1B K-projection geometry exercises both transform widths used
    # by the measured Q/K stage without making this a model-backed test.
    inner_eager = torch.randn((2048, 512), device="cuda", generator=generator,
                              requires_grad=True)
    inner_fused = inner_eager.detach().clone().requires_grad_()
    su = torch.randn((2048,), device="cuda", generator=generator)
    sv_eager = torch.randn((512,), device="cuda", generator=generator,
                           requires_grad=True)
    sv_fused = sv_eager.detach().clone().requires_grad_()
    upstream = torch.randn((512, 2048), device="cuda", generator=generator)

    eager = _rht_reconstruct_differentiable(inner_eager, su, sv_eager)
    fused = _rht_reconstruct_differentiable(
        inner_fused,
        su,
        sv_fused,
        fast_hadamard=True,
    )
    eager.backward(upstream)
    fused.backward(upstream)

    assert torch.equal(fused, eager)
    assert torch.equal(inner_fused.grad, inner_eager.grad)
    assert torch.equal(sv_fused.grad, sv_eager.grad)
