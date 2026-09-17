import pytest
import torch

from gptqmodel.quantization.gsq_training import GSQLion
from gptqmodel.quantization.gsq_training_qvq import (
    GSQP32TrainingModule,
    _ExactTrainingHadamard,
    _rht_reconstruct_differentiable,
    _SparseCandidateMatrixMixture,
    _TransposeView,
    p32_training_module_from_words,
)
from gptqmodel.quantization.qvq import rht_preprocess_weight, rht_reconstruct_weight
from gptqmodel.quantization.qvq_gsq import (
    TrellisCandidateAdapter,
    fisher_screened_trellis_candidates,
)
from scripts.validate_qvq_gsq_staged_layer import (
    closed_form_qk_scales,
    select_qk_pair,
    two_sided_normalized_hadamard,
    unique_qk_alternatives,
)


def _training_module(device="cpu", candidates=5):
    bits = 3
    baseline = torch.zeros((1, 24), dtype=torch.int32, device=device)
    bank = torch.zeros(1, dtype=torch.uint8, device=device)
    alt = torch.ones(1, dtype=torch.uint8, device=device)
    adapter = TrellisCandidateAdapter("p32_window", bits)
    teacher = adapter.inner(baseline, 16, 16, bank, alt)
    candidates, decoded, indices, deltas, shifts = fisher_screened_trellis_candidates(
        baseline,
        count=candidates,
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
    exact = _TransposeView.apply(matrix_exact)
    reference.backward(transpose_upstream)
    exact.backward(transpose_upstream)
    assert torch.equal(exact, reference)
    assert not exact.is_contiguous()
    assert torch.equal(matrix_exact.grad, matrix_reference.grad)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matrix_layout_sparse_mixture_is_bitwise_exact_for_real_p32_metadata():
    module, _, _ = _training_module("cuda", candidates=33)
    generator = torch.Generator(device="cuda").manual_seed(29)
    probabilities_matrix = torch.randn(
        module.logits.shape, device="cuda", generator=generator, requires_grad=True,
    )
    probabilities_fused = probabilities_matrix.detach().clone().requires_grad_()
    probabilities_transposed = probabilities_matrix.detach().clone().requires_grad_()
    from gptqmodel.quantization.qvq_gsq_triton import (
        build_compact_position_map,
        transpose_compact_position_map,
    )

    position_map = build_compact_position_map(
        module.sparse_indices, module.sparse_deltas, module.matrix_sparse_indices,
    )
    module.position_indices, module.position_choices, module.position_deltas = position_map
    transposed_map = transpose_compact_position_map(*position_map)
    (module.transposed_position_indices, module.transposed_position_choices,
     module.transposed_position_deltas) = transposed_map
    module.matrix_sparse_indices_transposed = (
        (module.matrix_sparse_indices % module.out_features) * module.in_features
        + module.matrix_sparse_indices // module.out_features
    )
    module.compact_forward = True
    upstream_matrix = torch.randn(
        module.baseline_matrix.shape, device="cuda", generator=generator,
    )
    matrix = _SparseCandidateMatrixMixture.apply(
        probabilities_matrix, module.baseline_matrix,
        module.matrix_sparse_indices, module.sparse_deltas,
    )
    fused = module._inner_from_probabilities(probabilities_fused)
    transposed = module._inner_from_probabilities(
        probabilities_transposed, transposed_output=True,
    )
    matrix.backward(upstream_matrix)
    fused.backward(upstream_matrix)
    transposed.backward(upstream_matrix.T.contiguous())

    torch.testing.assert_close(fused, matrix, rtol=0, atol=5e-7)
    assert torch.equal(transposed, fused.T.contiguous())
    assert torch.equal(probabilities_fused.grad, probabilities_matrix.grad)
    assert torch.equal(probabilities_transposed.grad, probabilities_matrix.grad)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_mixture_writes_bfloat16_directly_without_changing_gradients():
    module, _, _ = _training_module("cuda", candidates=33)
    from gptqmodel.quantization.qvq_gsq_triton import build_compact_position_map

    position_map = build_compact_position_map(
        module.sparse_indices, module.sparse_deltas, module.matrix_sparse_indices,
    )
    module.position_indices, module.position_choices, module.position_deltas = position_map
    module.compact_forward = True
    generator = torch.Generator(device="cuda").manual_seed(31)
    probabilities_reference = torch.randn(
        module.logits.shape, device="cuda", generator=generator, requires_grad=True,
    )
    probabilities_direct = probabilities_reference.detach().clone().requires_grad_()
    upstream = torch.randn(
        module.baseline_matrix.shape, dtype=torch.bfloat16,
        device="cuda", generator=generator,
    )

    reference = module._inner_from_probabilities(
        probabilities_reference, torch.float32,
    ).to(torch.bfloat16)
    direct = module._inner_from_probabilities(
        probabilities_direct, torch.bfloat16,
    )
    reference.backward(upstream)
    direct.backward(upstream)

    assert direct.dtype == torch.bfloat16
    assert torch.equal(direct, reference)
    assert torch.equal(probabilities_direct.grad, probabilities_reference.grad)


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_hard_evaluation_is_bitwise_equal_to_export_weight():
    module, candidates, _ = _training_module("cuda", candidates=33)
    assert module.fast_hadamard is True
    with torch.no_grad():
        module.logits.fill_(-1)
        module.logits[:, 17] = 1
        module.scales.copy_(torch.linspace(.5, 1.5, len(module.scales), device="cuda"))

    evaluated = module.hard_weight_for_evaluation()
    exported = module.hard_weight()

    assert torch.equal(evaluated, exported)
    assert torch.equal(module.hard_state()["words"], candidates[17])


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


def test_closed_form_qk_scales_minimize_the_masked_quadratic():
    generator = torch.Generator().manual_seed(43)
    unscaled = torch.randn((7, 16), generator=generator)
    expected = torch.linspace(.5, 1.5, 7)
    target = expected[:, None] * unscaled
    factor = torch.randn((16, 16), generator=generator)
    dead = torch.zeros(16, dtype=torch.bool)
    dead[[2, 11]] = True

    actual = closed_form_qk_scales(target, unscaled, factor, dead)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_qk_pair_selector_uses_joint_downstream_loss_and_stable_ties():
    q_alternatives = [
        {"selection": "original", "value": 0},
        {"selection": "local_q", "value": 2},
    ]
    k_alternatives = [
        {"selection": "original", "value": 0},
        {"selection": "local_k", "value": 3},
    ]

    q_selected, k_selected, measurements = select_qk_pair(
        q_alternatives,
        k_alternatives,
        lambda q, k: abs(q["value"] + k["value"] - 3),
    )

    assert q_selected["selection"] == "original"
    assert k_selected["selection"] == "local_k"
    assert len(measurements) == 4
    assert measurements[-1] == {
        "q_selection": "local_q",
        "k_selection": "local_k",
        "loss": 2,
    }


def test_qk_pair_selector_rejects_nonfinite_replay():
    alternatives = [{"selection": "original"}]
    with pytest.raises(ValueError, match="nonfinite"):
        select_qk_pair(alternatives, alternatives, lambda _q, _k: float("nan"))


def test_qk_pair_selector_batches_scalar_tensor_losses_and_preserves_ties():
    alternatives = [
        {"selection": "first", "value": 0},
        {"selection": "second", "value": 1},
    ]
    q_selected, k_selected, measurements = select_qk_pair(
        alternatives,
        alternatives,
        lambda q, k: torch.tensor(float(q["value"] + k["value"])),
    )

    assert q_selected["selection"] == "first"
    assert k_selected["selection"] == "first"
    assert [measurement["loss"] for measurement in measurements] == [0., 1., 1., 2.]


def test_qk_pair_alternatives_deduplicate_identical_serialized_states():
    original = {
        "selection": "original",
        "state": {"words": torch.zeros(2, 3), "SV": torch.ones(2)},
    }
    duplicate = {
        "selection": "duplicate",
        "state": {"words": original["state"]["words"].clone(), "SV": torch.ones(2)},
    }
    changed = {
        "selection": "changed",
        "state": {"words": torch.ones(2, 3), "SV": torch.ones(2)},
    }

    unique = unique_qk_alternatives([original, duplicate, changed])

    assert [alternative["selection"] for alternative in unique] == ["original", "changed"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inner_fisher_metric_matches_dense_qk_quadratic():
    generator = torch.Generator(device="cuda").manual_seed(47)
    inner = torch.randn((16, 16), device="cuda", generator=generator)
    target = torch.randn((16, 16), device="cuda", generator=generator)
    su = torch.rand(16, device="cuda", generator=generator) + .5
    sv = torch.rand(16, device="cuda", generator=generator) + .5
    factor = torch.randn((16, 16), device="cuda", generator=generator)

    weight = rht_reconstruct_weight(inner, su, sv)
    target_inner = rht_preprocess_weight(
        target, su.reciprocal(), sv.reciprocal(),
    ).float()
    error = inner - target_inner
    input_metric = two_sided_normalized_hadamard(
        su[:, None] * (factor @ factor.T) * su[None, :],
    )
    output_metric = two_sided_normalized_hadamard(torch.diag(sv.square()))

    dense_loss = ((target - weight) @ factor).square().sum()
    inner_loss = (error * (input_metric @ error @ output_metric)).sum()

    torch.testing.assert_close(inner_loss, dense_loss, rtol=2e-6, atol=2e-3)


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scaled_hadamard_matches_prior_bfloat16_native_operation_order():
    generator = torch.Generator(device="cuda").manual_seed(43)
    inner_prior = torch.randn(
        (2048, 512), device="cuda", dtype=torch.bfloat16,
        generator=generator, requires_grad=True,
    )
    inner_fused = inner_prior.detach().clone().requires_grad_()
    su = torch.randn(
        (2048,), device="cuda", dtype=torch.bfloat16, generator=generator,
    )
    sv_prior = torch.randn(
        (512,), device="cuda", dtype=torch.bfloat16,
        generator=generator, requires_grad=True,
    )
    sv_fused = sv_prior.detach().clone().requires_grad_()
    upstream = torch.randn(
        (512, 2048), device="cuda", dtype=torch.bfloat16, generator=generator,
    )

    prior = _ExactTrainingHadamard.apply(inner_prior.T.contiguous()).T
    prior = prior * su.unsqueeze(1)
    prior = _ExactTrainingHadamard.apply(prior)
    prior = prior * sv_prior.unsqueeze(0)
    prior = _TransposeView.apply(prior)
    fused = _rht_reconstruct_differentiable(
        inner_fused, su, sv_fused, fast_hadamard=True,
    )
    prior.backward(upstream)
    fused.backward(upstream)

    assert torch.equal(fused, prior)
    assert torch.equal(inner_fused.grad, inner_prior.grad)
    assert torch.equal(sv_fused.grad, sv_prior.grad)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_gsq_lion_is_bitwise_exact():
    generator = torch.Generator(device="cuda").manual_seed(47)
    initial = torch.randn(12345, device="cuda", generator=generator)
    fused = torch.nn.Parameter(initial.clone())
    reference = initial.clone()
    reference_momentum = torch.zeros_like(reference)
    optimizer = GSQLion(
        [{"params": [fused], "lr": 2e-4, "weight_decay": 1.0}],
        betas=(0.9, 0.95),
    )

    for step in range(20):
        gradient = torch.randn(
            fused.shape, device="cuda", generator=generator,
        )
        learning_rate = 2e-4 * (1.0 - step / 20)
        optimizer.param_groups[0]["lr"] = learning_rate
        fused.grad = gradient.clone()
        optimizer.step()

        direction = reference_momentum.clone().mul_(0.9).add_(
            gradient, alpha=0.1,
        ).sign_()
        reference.mul_(1.0 - learning_rate).add_(
            direction, alpha=-learning_rate,
        )
        reference_momentum.mul_(0.95).add_(gradient, alpha=0.05)
        assert torch.equal(fused, reference)
        assert torch.equal(
            optimizer.state[fused]["exp_avg"], reference_momentum,
        )
