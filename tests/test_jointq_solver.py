# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Numerical oracles for the JointQ refinement.

Every expected value here is computed independently of the implementation:
closed-form scale solutions, the paper's separation-instance bound, or a
direct objective evaluation on a fixed Hessian.
"""

import copy

import pytest
import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear import _scales_for_storage
from gptqmodel.quantization.config import GPTQConfig, JointQConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.jointq import (
    JointQ,
    _build_group_plan,
    _clip_candidate,
    _codes_from_reconstruction,
    _dequantize,
    _integer_search,
    _joint_scale_solve,
    _local_search,
    _row_objectives,
    _solve_scales,
    paper_regularization,
)
from gptqmodel.quantization.quantizer import SCALE_STORAGE_FLOOR


def test_paper_regularization_matches_table_6():
    assert paper_regularization(4, 128, 4096, "model.layers.0.self_attn.v_proj") == pytest.approx(0.01)
    assert paper_regularization(4, 128, 4096, "model.layers.0.mlp.down_proj") == 0.0
    assert paper_regularization(4, -1, 4096, "model.layers.0.self_attn.v_proj") == pytest.approx(0.15)
    assert paper_regularization(4, 4096, 4096, "model.layers.0.mlp.gate_proj") == pytest.approx(0.15)
    assert paper_regularization(3, 128, 4096, "model.layers.0.mlp.o_proj") == 0.0
    assert paper_regularization(3, -1, 4096, "model.layers.0.mlp.down_proj") == pytest.approx(0.01)
    assert paper_regularization(2, 32, 4096, "model.layers.0.self_attn.q_proj") == pytest.approx(0.1)
    assert paper_regularization(2, 32, 4096, "model.layers.0.mlp.gate_proj") == 0.0
    # Settings the authors did not tune fall back to no regularization.
    assert paper_regularization(4, 64, 4096, "model.layers.0.self_attn.v_proj") == 0.0
    assert paper_regularization(8, 128, 4096, "model.layers.0.self_attn.v_proj") == 0.0


def test_group_plan_groups_by_g_idx_not_by_position():
    g_idx = torch.tensor([2, 0, 2, 1, 0, 2], dtype=torch.int32)

    plan = _build_group_plan(g_idx, 3)

    assert [columns.tolist() for columns in plan.columns] == [[1, 4], [3], [0, 2, 5]]
    scales = torch.tensor([[10.0, 20.0, 30.0]])
    assert plan.expand(scales).tolist() == [[30.0, 10.0, 30.0, 20.0, 10.0, 30.0]]


def test_group_plan_rejects_missing_and_empty_groups():
    with pytest.raises(ValueError, match="empty group"):
        _build_group_plan(torch.tensor([0, 0, 2], dtype=torch.int32), 3)
    with pytest.raises(ValueError, match="must lie in"):
        _build_group_plan(torch.tensor([0, 3], dtype=torch.int32), 3)


def test_codes_from_reconstruction_mirrors_the_packer():
    plan = _build_group_plan(torch.tensor([0, 0, 1, 1], dtype=torch.int32), 2)
    scales = torch.tensor([[0.5, 2.0]])
    zero = 8.0
    reconstruction = torch.tensor([[0.0, 0.0, -2.0, 2.0]])

    codes = _codes_from_reconstruction(reconstruction, scales, zero, plan, 15)

    # The packer's exact FP32 arithmetic, including the zero-scale lane.
    scale_cols = plan.expand(scales)
    expected = torch.round((reconstruction + zero * scale_cols) / scale_cols)
    assert codes.tolist() == [[8, 8, 7, 9]]
    assert torch.equal(codes, expected.clamp(0, 15).to(torch.int64))


def test_integer_search_follows_algorithm_2_on_the_separation_instance():
    # Proposition 3.5: d = 3, b = 2, H = I, w = (M, 1, 1), 2 < M < 1 + sqrt(2).
    # Algorithm 2 only accepts strictly improving +-1 moves, so it stops at
    # q = (1, 1, 1) even though a non-adjacent codebook point is better. This
    # checks the algorithm's own endpoint, not the global minimum.
    zero, max_code = 2.0, 3
    weight = torch.tensor([[2.2, 1.0, 1.0]], dtype=torch.float64)
    hessian = torch.eye(3, dtype=torch.float64)

    # Clip-Optimize initialization: s = max|w| / (L - 1) = 2.2.
    scale = weight.abs().amax() / (zero - 1.0)
    raw = torch.round(weight / scale).clamp(-zero, zero - 1) + zero
    raw = raw.to(torch.int64)

    codes, sigma = _integer_search(
        weight @ hessian,
        raw.clone(),
        hessian,
        zero=zero,
        max_code=max_code,
        tolerance=1e-12,
        stall_fraction=0.1,
        iteration_cap=10 * 3,
    )

    centered = codes.to(torch.float64) - zero
    assert centered.tolist() == [[1.0, 1.0, 1.0]]
    # Best-response scale at that point: (M + 1 + 1) / (1 + 1 + 1).
    assert sigma.item() == pytest.approx(4.2 / 3.0, abs=1e-12)
    error = float(((weight - sigma * centered) ** 2).sum())
    # Closed form: (M - sigma)^2 + (d - 1) * (1 - sigma)^2.
    assert error == pytest.approx(0.96, abs=1e-12)

    def best_response_error(candidate):
        candidate = candidate.to(torch.float64)
        gram = (candidate * candidate).sum()
        projection = ((weight * candidate).sum() / gram) * candidate
        return float(((weight - projection) ** 2).sum())

    # Local optimality: no single feasible +-1 move improves the objective.
    for index in range(centered.shape[1]):
        for delta in (-1, 1):
            neighbor = centered.clone()
            neighbor[0, index] += delta
            if not (-zero <= neighbor[0, index] <= zero - 1):
                continue
            assert best_response_error(neighbor) >= error - 1e-12


def test_solve_scales_uses_the_exact_solution_and_damps_failures():
    gram = torch.tensor([[[4.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    rhs = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    solution, solved = _solve_scales(gram, rhs, scale_bound=65504.0)

    expected = torch.linalg.solve(gram[0], rhs[0].unsqueeze(-1)).squeeze(-1)
    assert bool(solved.all())
    assert torch.allclose(solution[0], expected, atol=1e-12)

    # A singular row is retried with damping; an unreachable scale bound leaves
    # the row unsolved so the caller keeps its previous scale.
    singular = torch.zeros((1, 2, 2), dtype=torch.float64)
    unsolved, solved = _solve_scales(singular, torch.tensor([[1.0, 0.0]], dtype=torch.float64), scale_bound=1e-3)
    assert not bool(solved.any())
    assert torch.isfinite(unsolved).all()


def test_joint_scale_solve_matches_the_closed_form_for_active_groups():
    torch.manual_seed(3)
    weight = torch.randn(4, 6, dtype=torch.float64)
    hessian = torch.randn(6, 6, dtype=torch.float64)
    hessian = hessian @ hessian.T + 0.5 * torch.eye(6, dtype=torch.float64)
    plan = _build_group_plan(torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32), 2)
    zero = 8.0
    codes = torch.randint(0, 16, (4, 6), dtype=torch.int64)
    scales = torch.rand(4, 2, dtype=torch.float64) + 0.1

    new_scales, new_phi = _joint_scale_solve(
        weight, codes, scales, hessian, plan, zero, tolerance=1e-12, nsamples=16, scale_bound=65504.0
    )

    # Independent closed form: s = w H A^T (A H A^T)^-1 per row.
    centered = codes.to(torch.float64) - zero
    A = torch.zeros(2, 6, dtype=torch.float64)
    A[0, :3] = centered[0, :3]
    A[1, 3:] = centered[0, 3:]
    # Use a row whose groups are both active.
    for row in range(4):
        A = torch.zeros(2, 6, dtype=torch.float64)
        A[0, :3] = centered[row, :3]
        A[1, 3:] = centered[row, 3:]
        if not bool(A.any(dim=1).all()):
            continue
        gram = A @ hessian @ A.T
        expected = torch.linalg.solve(gram, (A @ hessian @ weight[row]))
        assert torch.allclose(new_scales[row], expected, atol=1e-8)

    reconstruction = _dequantize(codes, new_scales, zero, plan)
    phi = _row_objectives(weight, reconstruction, hessian)
    assert torch.allclose(phi, new_phi, atol=1e-10)


def test_joint_scale_solve_zeros_inactive_groups():
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    hessian = torch.eye(4, dtype=torch.float64)
    plan = _build_group_plan(torch.tensor([0, 0, 1, 1], dtype=torch.int32), 2)
    codes = torch.tensor([[8, 9, 8, 8]], dtype=torch.int64)
    scales = torch.tensor([[0.5, 0.7]], dtype=torch.float64)

    new_scales, _ = _joint_scale_solve(
        weight, codes, scales, hessian, plan, 8.0, tolerance=1e-12, nsamples=1, scale_bound=65504.0
    )

    assert new_scales[0, 1].item() == 0.0
    reconstruction = _dequantize(codes, new_scales, 8.0, plan)
    assert reconstruction[0, 2].item() == 0.0 and reconstruction[0, 3].item() == 0.0


def test_clip_candidate_maps_the_largest_entry_to_the_last_level():
    weight = torch.tensor([[2.0, -1.0, 0.5, 0.0]], dtype=torch.float64)
    hessian = torch.eye(4, dtype=torch.float64)
    plan = _build_group_plan(torch.tensor([0, 0, 1, 1], dtype=torch.int32), 2)

    codes, scales = _clip_candidate(
        weight, hessian, plan, 8.0, 15, tolerance=1e-12, stall_fraction=0.1, iteration_cap=40
    )

    centered = codes.to(torch.float64) - 8.0
    # The largest magnitude entry saturates the positive level.
    assert codes[0, 0].item() == 15
    assert centered[0, 0].item() == 7.0
    # The returned scale is the best response for the returned integers, not
    # the min-max initialization: sigma = (w . q) / (q . q).
    assert scales[0, 0].item() == pytest.approx(18.0 / 65.0, abs=1e-12)
    assert scales[0, 1].item() == pytest.approx(3.5 / 49.0, abs=1e-12)


def test_local_search_never_increases_the_row_objective():
    torch.manual_seed(11)
    weight = torch.randn(5, 8, dtype=torch.float64)
    hessian = torch.randn(8, 8, dtype=torch.float64)
    hessian = hessian @ hessian.T + 0.25 * torch.eye(8, dtype=torch.float64)
    plan = _build_group_plan(torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32), 2)
    zero = 8.0
    codes = torch.randint(0, 16, (5, 8), dtype=torch.int64)
    scales = torch.rand(5, 2, dtype=torch.float64) + 0.05

    before = _row_objectives(weight, _dequantize(codes, scales, zero, plan), hessian)
    new_codes, new_scales, phi, passes = _local_search(
        weight,
        hessian,
        codes.clone(),
        scales.clone(),
        plan,
        zero,
        15,
        tolerance=1e-12,
        stall_fraction=0.1,
        active_fraction=0.1,
        re_solve_low=0.2,
        re_solve_high=0.6,
        iteration_cap=40,
        max_passes=10,
        nsamples=16,
        scale_bound=65504.0,
    )

    assert passes >= 1
    assert (new_codes >= 0).all() and (new_codes <= 15).all()
    assert torch.all(phi <= before + 1e-9)
    after = _row_objectives(weight, _dequantize(new_codes, new_scales, zero, plan), hessian)
    assert torch.allclose(phi, after, atol=1e-10)


def test_finalize_storage_pins_zero_scale_groups_and_keeps_fp16_representable():
    plan = _build_group_plan(torch.tensor([0, 0, 1, 1], dtype=torch.int32), 2)
    codes = torch.tensor([[8, 9, 3, 4]], dtype=torch.int64)
    scales = torch.tensor([[0.0, 0.25]], dtype=torch.float32)
    fallback_codes = torch.tensor([[8, 9, 8, 8]], dtype=torch.int64)
    fallback_scales = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

    out_codes, out_scales = JointQ._finalize_storage(
        codes, scales, fallback_codes, fallback_scales, plan, 8.0, 15
    )

    assert out_scales[0, 0].item() == SCALE_STORAGE_FLOOR
    assert out_codes[0, 0].item() == 8 and out_codes[0, 1].item() == 8
    # The representable group keeps its refined values.
    assert out_scales[0, 1].item() == pytest.approx(0.25)
    assert out_codes[0, 2].item() == 3
    # The returned scales are exactly what the checkpoint writer can store.
    _scales_for_storage(out_scales, name="test")


def _run_quantizer(cls, linear, qcfg, batches):
    named = NamedModule(linear, "proj", "model.layers.0.proj", 0)
    task = cls(named, qcfg)
    task.quantizer.configure(perchannel=True)
    for batch in batches:
        task.add_batch(batch, None)
    return task, task.quantize(blocksize=8)


def test_jointq_beats_gptq_on_the_calibration_objective_and_packs_consistently():
    torch.manual_seed(7)
    in_features, out_features, group_size = 32, 8, 8
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    batches = [torch.randn(6, in_features) for _ in range(5)]
    hessian = sum(batch.T @ batch for batch in batches)
    qcfg = GPTQConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        damp_percent=0.01,
        jointq=JointQConfig(regularization=0.0, dtype="float64"),
    )

    def prepared_linear():
        module = torch.nn.Linear(in_features, out_features, bias=False)
        module.weight.data.copy_(linear.weight.data)
        return module

    # `quantize` reads the module weight, so the reference values must be in
    # place before either task runs.
    gptq_task, gptq_result = _run_quantizer(
        GPTQ, prepared_linear(), copy.deepcopy(qcfg), batches
    )
    jointq_task, jointq_result = _run_quantizer(
        JointQ, prepared_linear(), copy.deepcopy(qcfg), batches
    )

    def objective(result):
        weight, scales, _, g_idx, *_ = result
        plan = _build_group_plan(g_idx.cpu(), scales.shape[1])
        codes = _codes_from_reconstruction(weight.float(), scales.float(), 8.0, plan, 15)
        reconstruction = _dequantize(codes, scales.float(), 8.0, plan)
        difference = linear.weight.data.float() - reconstruction
        return float((difference @ hessian @ difference.T).trace())

    assert objective(jointq_result) <= objective(gptq_result) + 1e-6

    scales = jointq_result[1]
    _scales_for_storage(scales, name="jointq")
    plan = _build_group_plan(jointq_result[3].cpu(), scales.shape[1])
    codes = _codes_from_reconstruction(jointq_result[0].float(), scales.float(), 8.0, plan, 15)
    scale_cols = plan.expand(scales.float())
    packer = torch.round((jointq_result[0].float() + 8.0 * scale_cols) / scale_cols).clamp_(0, 15)
    assert torch.equal(packer.to(torch.int64), codes)
