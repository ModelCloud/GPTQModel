import pytest
import torch

import gptqmodel.quantization.awq.modules.triton.scheduler as scheduler
from gptqmodel.quantization.awq.modules.triton.scheduler import (
    AwqTritonPlan,
    candidate_plans,
    clear_awq_triton_rules,
    mark_awq_triton_plan_warmed,
    legacy_plan,
    register_awq_triton_rule,
    resolve_group_size,
    select_awq_triton_plan,
    validate_fused_config,
)


def teardown_function(_function):
    clear_awq_triton_rules()


def test_group_and_split_constraints_are_pruned():
    ok, reason = validate_fused_config(
        M=1, N=512, K=128, group_size=16,
        block_size_m=32, block_size_n=32, block_size_k=32, split_k_iters=8,
    )
    assert not ok and "group_size" in reason
    ok, reason = validate_fused_config(
        M=1, N=512, K=128, group_size=128,
        block_size_m=32, block_size_n=32, block_size_k=32, split_k_iters=8,
    )
    assert not ok and "empty" in reason
    plans, rejected = candidate_plans(1, 512, 128, 128)
    fused = [plan for plan in plans if plan.fused]
    assert all(plan.block_size_k <= 128 for plan in fused)
    assert all(plan.split_k_iters <= 4 for plan in fused)
    assert any(plan.path == "dense" for plan in plans)
    assert rejected


@pytest.mark.parametrize(
    "overrides,reason_fragment",
    [
        ({"N": 510}, "divisible by 8"),
        ({"block_size_m": 8}, "BLOCK_SIZE_M"),
        ({"block_size_n": 16}, "BLOCK_SIZE_N"),
        ({"block_size_k": 16}, "BLOCK_SIZE_K"),
        ({"block_size_k": 64, "group_size": 32}, "BLOCK_SIZE_K"),
        ({"split_k_iters": 3}, "power of two"),
        ({"split_k_iters": 16}, "empty"),
    ],
)
def test_invalid_fused_layouts_are_rejected(overrides, reason_fragment):
    kwargs = {
        "M": 1, "N": 512, "K": 256, "group_size": 128,
        "block_size_m": 32, "block_size_n": 32, "block_size_k": 32,
        "split_k_iters": 8,
    }
    kwargs.update(overrides)
    ok, reason = validate_fused_config(**kwargs)
    assert not ok
    assert reason_fragment in reason


def test_full_k_group_is_resolved_before_validation():
    assert resolve_group_size(-1, 256) == 256
    ok, reason = validate_fused_config(
        M=1, N=264, K=256, group_size=-1,
        block_size_m=16, block_size_n=32, block_size_k=64,
        split_k_iters=4, num_warps=4, num_stages=2,
    )
    assert ok and reason is None


@pytest.mark.parametrize("K,expected", [(16, False), (48, False), (96, True)])
def test_full_k_group_respects_bk_alignment(K, expected):
    ok, _ = validate_fused_config(
        M=1, N=264, K=K, group_size=-1,
        block_size_m=16, block_size_n=32, block_size_k=32,
        split_k_iters=1, num_warps=4, num_stages=2,
    )
    assert ok is expected


def test_legacy_boundary_and_short_k_are_safe():
    assert legacy_plan(128, 256, 512, 128).path == "fused"
    assert legacy_plan(129, 256, 512, 128).path == "dense"
    assert legacy_plan(1, 128, 512, 128).path == "fused"
    assert legacy_plan(1, 128, 512, 128).split_k_iters == 8


def test_auto_unknown_device_falls_back_and_exact_rule_is_isolated():
    baseline = select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=True,
    )
    assert baseline == legacy_plan(1, 256, 512, 128)
    candidate = AwqTritonPlan("fused", 16, 64, 64, 2, 4, 2)
    register_awq_triton_rule(
        "cpu", candidate, M=1, N=512, K=256, group_size=128,
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=True,
    )
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=True,
    ) == candidate
    # Different precision and M must not reuse the measured entry.
    assert select_awq_triton_plan(
        M=2, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=True,
    ) == legacy_plan(2, 256, 512, 128)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=False,
    ) == legacy_plan(1, 256, 512, 128)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=64, device="cpu",
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=True,
    ) == legacy_plan(1, 256, 512, 64)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.bfloat16, output_dtype=torch.float16, fp32_accum=True,
    ) == legacy_plan(1, 256, 512, 128)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, output_dtype=torch.float32, fp32_accum=True,
    ) == legacy_plan(1, 256, 512, 128)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, compute_dtype=torch.bfloat16,
        output_dtype=torch.float16, fp32_accum=True,
    ) == legacy_plan(1, 256, 512, 128)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu:1",
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=True,
    ) == legacy_plan(1, 256, 512, 128)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, output_dtype=torch.float16, fp32_accum=True,
        mode="legacy",
    ) == legacy_plan(1, 256, 512, 128)


def test_builtin_rule_requires_exact_verified_device_and_precision(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "device_identity",
        lambda _device: (
            "cuda", 0, "NVIDIA GeForce RTX 4090", (8, 9), 128, 50_950_569_984,
        ),
    )
    expected = AwqTritonPlan("fused", 64, 64, 64, 1, 4, 2)
    kwargs = {
        "M": 64, "N": 8192, "K": 2048, "group_size": 128,
        "device": "cuda:0", "input_dtype": torch.float16,
        "compute_dtype": torch.float16, "output_dtype": torch.float16,
        "fp32_accum": True,
    }
    assert select_awq_triton_plan(**kwargs) == expected
    clear_awq_triton_rules()
    assert select_awq_triton_plan(**{**kwargs, "output_dtype": torch.float32}) == legacy_plan(64)
    clear_awq_triton_rules()
    monkeypatch.setattr(
        scheduler,
        "device_identity",
        lambda _device: (
            "cuda", 0, "NVIDIA GeForce RTX 4090", (8, 9), 128, 25_745_473_536,
        ),
    )
    assert select_awq_triton_plan(**kwargs) == legacy_plan(64)


def test_explicit_and_training_modes():
    explicit = AwqTritonPlan("fused", 16, 32, 32, 2, 4, 2)
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu", explicit=explicit,
    ) == explicit
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu", explicit=explicit,
        training=True,
    ) == legacy_plan(1, 256, 512, 128)
    with pytest.raises(RuntimeError, match="eager prewarm"):
        select_awq_triton_plan(
            M=1, N=512, K=256, group_size=128, device="cpu", explicit=explicit,
            cuda_graph=True,
        )
    mark_awq_triton_plan_warmed(
        explicit, M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, compute_dtype=torch.float16,
        output_dtype=torch.float16, fp32_accum=True, explicit=explicit,
    )
    assert select_awq_triton_plan(
        M=1, N=512, K=256, group_size=128, device="cpu",
        input_dtype=torch.float16, compute_dtype=torch.float16,
        output_dtype=torch.float16, fp32_accum=True, explicit=explicit,
        cuda_graph=True,
    ) == explicit
    with pytest.raises(ValueError, match="invalid AWQ Triton explicit schedule"):
        select_awq_triton_plan(
            M=1, N=512, K=256, group_size=128, device="cpu",
            explicit=AwqTritonPlan("fused", block_size_k=16),
        )
    with pytest.raises(ValueError, match="unknown AWQ Triton path"):
        select_awq_triton_plan(
            M=1, N=512, K=256, group_size=128, device="cpu",
            explicit=AwqTritonPlan("unknown"),
        )


def test_cuda_graph_warmup_is_isolated_by_request_and_rule_changes():
    candidate = AwqTritonPlan("fused", 16, 64, 64, 2, 4, 2)
    common = {
        "N": 512, "K": 256, "group_size": 128, "device": "cpu",
        "input_dtype": torch.float16, "compute_dtype": torch.float16,
        "output_dtype": torch.float16, "fp32_accum": True,
    }
    mark_awq_triton_plan_warmed(candidate, M=1, **common)
    with pytest.raises(RuntimeError, match="eager prewarm"):
        select_awq_triton_plan(M=1, mode="legacy", cuda_graph=True, **common)

    mark_awq_triton_plan_warmed(legacy_plan(2), M=2, mode="legacy", **common)
    with pytest.raises(RuntimeError, match="eager prewarm"):
        select_awq_triton_plan(M=2, explicit=candidate, cuda_graph=True, **common)

    mark_awq_triton_plan_warmed(legacy_plan(3), M=3, training=True, **common)
    with pytest.raises(RuntimeError, match="eager prewarm"):
        select_awq_triton_plan(M=3, cuda_graph=True, **common)

    mark_awq_triton_plan_warmed(legacy_plan(4), M=4, **common)
    register_awq_triton_rule(
        "cpu", candidate, M=4, N=512, K=256, group_size=128,
        input_dtype=torch.float16, compute_dtype=torch.float16,
        output_dtype=torch.float16, fp32_accum=True,
    )
    with pytest.raises(RuntimeError, match="eager prewarm"):
        select_awq_triton_plan(M=4, cuda_graph=True, **common)


def test_unindexed_cuda_cache_resolves_current_device(monkeypatch):
    current = {"index": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])
    monkeypatch.setattr(
        scheduler, "device_identity",
        lambda _device: ("cuda", current["index"], None, None, None, None),
    )
    kwargs = {
        "M": 1, "N": 512, "K": 256, "group_size": 128,
        "device": "cuda", "input_dtype": torch.float16,
        "compute_dtype": torch.float16, "output_dtype": torch.float16,
    }
    select_awq_triton_plan(**kwargs)
    current["index"] = 1
    select_awq_triton_plan(**kwargs)
    assert {key[1] for key in scheduler._PLAN_CACHE} == {0, 1}


def test_plan_cache_is_bounded_and_clearable():
    for M in range(1, scheduler.MAX_PLAN_CACHE + 20):
        select_awq_triton_plan(
            M=M, N=512, K=256, group_size=128, device="cpu",
            input_dtype=torch.float16, output_dtype=torch.float16,
        )
    assert len(scheduler._PLAN_CACHE) == scheduler.MAX_PLAN_CACHE
    scheduler.clear_awq_triton_plan_cache()
    assert not scheduler._PLAN_CACHE


def test_empty_input_is_rejected_before_scheduling():
    with pytest.raises(ValueError, match="empty"):
        select_awq_triton_plan(M=0, N=512, K=256, group_size=128, device="cpu")
