import pytest
import torch

from gptqmodel.quantization.gsq_training_qvq import (
    GSQP32TrainingModule,
    _rht_reconstruct_differentiable,
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


def test_p32_staged_hard_state_is_exact_legal_candidate():
    module, candidates, _ = _training_module()
    with torch.no_grad():
        module.logits.fill_(-1)
        module.logits[:, 1] = 1
    state = module.hard_state()
    assert torch.equal(state["words"], candidates[1])
    assert torch.equal(module.adapter.pack(module.adapter.unpack(state["words"])), state["words"])
    expected_inner = module.adapter.inner(
        candidates[1], 16, 16, module.bank_ids, module.bank_alt_id,
    )
    torch.testing.assert_close(
        module.hard_weight(),
        rht_reconstruct_weight(expected_inner, module.SU, module.scales),
    )


def test_p32_staged_rejects_inconsistent_sparse_metadata():
    module, candidates, decoded = _training_module()
    with pytest.raises(ValueError, match="metadata"):
        GSQP32TrainingModule(
            candidates,
            decoded[0],
            module.sparse_indices.permute(1, 0, 2)[:, :, :-1],
            module.sparse_deltas.permute(1, 0, 2),
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
