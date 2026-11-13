# SPDX-License-Identifier: Apache-2.0
"""Qwen's nested target layer 45 must never inherit GLM's MTP identity."""

from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.utils.exl3_capture_frontier import (
    EXL3CaptureFrontierStore,
    EXL3CaptureState,
)
from gptqmodel.utils.exl3_error_ledger import (
    build_projection_record,
    routed_expert_identity,
)
from gptqmodel.utils.exl3_inline_mixed import (
    INLINE_MIXED_META_KEY,
    INLINE_MIXED_NAMESPACES_META_KEY,
    InlineMixedPolicy,
    inline_mixed_policy,
)


FAMILY = {
    "model_type": "qwen4_exp",
    "target_layer_count": 48,
    "source_revision": "qwen-test-source",
}
BASE = "model.language_model.layers.45.mlp.experts.7.gate_proj"
MTP = "mtp.layers.0.mlp.experts.7.gate_proj"


def _processor(meta):
    processor = object.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(meta=meta)
    return processor


def _policies(tmp_path):
    return {
        namespace: dict(
            InlineMixedPolicy(
                namespace=namespace,
                base_bits=4,
                upgrade_bits=5,
                extra_bits=Fraction(1, 4),
                projection_ratio=(3, 5, 8),
                tier_plan_root=tmp_path / namespace,
                logical_layer_count=count,
            ).policy_body,
            tier_plan_root=str(tmp_path / namespace),
        )
        for namespace, count in (("base", 48), ("mtp", 1))
    }


def test_qwen_identity_preserves_legacy_glm_checkpoints():
    assert routed_expert_identity(BASE)["block_namespace"] == "mtp"
    assert (
        routed_expert_identity(BASE, family_join={"model_type": "glm5_next"})[
            "block_namespace"
        ]
        == "mtp"
    )
    assert routed_expert_identity(BASE, family_join=FAMILY) == {
        "block_namespace": "base",
        "logical_layer": 45,
        "expert": 7,
        "projection": "w1",
    }
    assert routed_expert_identity(MTP, family_join=FAMILY) == {
        "block_namespace": "mtp",
        "logical_layer": 0,
        "expert": 7,
        "projection": "w1",
    }
    processor = _processor({"ds4rt_error_ledger": {"family_join": FAMILY}})
    assert processor._routed_expert_identity(BASE)["block_namespace"] == "base"
    phase, identities = processor._subset_capture_phase(
        {"gate": SimpleNamespace(full_name=BASE)}
    )
    assert phase == "gate-up" and identities["gate"]["block_namespace"] == "base"
    record = build_projection_record(
        module_full_name=BASE,
        layer_index=45,
        bits=4,
        codebook="mcg",
        sample_count=32,
        duration_seconds=1,
        encoded_bytes=128,
        device_names=["test"],
        quantizer_metrics={},
        provenance={"family_join": FAMILY},
    )
    assert record["block_namespace"] == "base" and record["logical_layer"] == 45


def test_qwen_frontier_round_trip_and_scoped_discard(tmp_path):
    store = EXL3CaptureFrontierStore(tmp_path, family_join=FAMILY)
    for index, name, namespace in ((45, BASE, "base"), (48, MTP, "mtp")):
        subset = {"gate": SimpleNamespace(full_name=name)}
        manifest = store.commit(
            layer_index=index,
            subset_index=0,
            subset_total=1,
            subset=subset,
            states=[
                EXL3CaptureState(
                    module=name,
                    hessian=torch.eye(4),
                    sample_count=32,
                    route_evidence=None,
                )
            ],
        )
        assert (
            manifest["captures"][0]["expert_identity"]["block_namespace"] == namespace
        )
        restored = store.restore(
            layer_index=index, subset_index=0, subset_total=1, subset=subset
        )
        torch.testing.assert_close(restored[name].hessian, torch.eye(4), rtol=0, atol=0)
    store.discard_through(48, block_namespace="base")
    assert (
        store.restore(
            layer_index=45,
            subset_index=0,
            subset_total=1,
            subset={"gate": SimpleNamespace(full_name=BASE)},
        )
        is None
    )
    assert (
        store.restore(
            layer_index=48,
            subset_index=0,
            subset_total=1,
            subset={"gate": SimpleNamespace(full_name=MTP)},
        )
        is not None
    )


def test_integrated_qwen_policies_select_both_namespaces_and_exact_425_358_quotas(
    tmp_path,
):
    policies = _policies(tmp_path)
    meta = {
        "ds4rt_error_ledger": {"family_join": FAMILY},
        INLINE_MIXED_NAMESPACES_META_KEY: policies,
    }
    processor = _processor(meta)
    base = processor._inline_mixed_policy(BASE)
    mtp = processor._inline_mixed_policy(MTP)
    assert base.namespace == "base" and mtp.namespace == "mtp"
    assert base.target_bpw == mtp.target_bpw == Fraction(17, 4)
    assert base.namespace_quotas(layer_count=48, experts_per_layer=512) == {
        "w1": 3456,
        "w3": 5760,
        "w2": 9216,
    }
    assert mtp.namespace_quotas(layer_count=1, experts_per_layer=512) == {
        "w1": 72,
        "w3": 120,
        "w2": 192,
    }
    for index in range(48):
        assert base.layer_quotas(
            layer_index=index, layer_count=49, experts_per_layer=512
        ) == {"w1": 72, "w3": 120, "w2": 192}
    assert mtp.layer_quotas(layer_index=0, layer_count=49, experts_per_layer=512) == {
        "w1": 72,
        "w3": 120,
        "w2": 192,
    }
    assert (
        inline_mixed_policy({INLINE_MIXED_META_KEY: policies["base"]}, namespace="mtp")
        is None
    )


def test_integrated_policies_reject_ambiguous_or_mismatched_contracts(tmp_path):
    policies = _policies(tmp_path)
    with pytest.raises(ValueError, match="invalid"):
        inline_mixed_policy(
            {
                INLINE_MIXED_NAMESPACES_META_KEY: policies,
                INLINE_MIXED_META_KEY: policies["base"],
            }
        )
    policies["mtp"]["namespace"] = "base"
    with pytest.raises(ValueError, match="identity differs"):
        inline_mixed_policy({INLINE_MIXED_NAMESPACES_META_KEY: policies})


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("tied", [False, True])
def test_qwen_softmax_recovery_preserves_live_top10_and_ranks_11_through_20(
    device, tied
):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

    from gptqmodel.looper.exllamav3_processor import _router_recovery_candidates
    from gptqmodel.utils.exl3_router_candidates import bind_softmax_router_recovery

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(11)
    router = Qwen4ExpTextTopKRouter(
        SimpleNamespace(
            num_experts_per_tok=10,
            num_experts=32,
            norm_topk_prob=True,
            hidden_size=16,
        )
    ).to(device=device, dtype=torch.bfloat16)
    if not tied:
        torch.nn.init.normal_(router.weight, std=0.1)
    inputs = torch.randn(17, 16, dtype=torch.bfloat16, device=device)
    logits, weights, selected = router(inputs)
    with pytest.raises(RuntimeError, match="requires a learned"):
        _router_recovery_candidates(
            router, logits, selected, candidate_rank_min=11, candidate_rank_max=20
        )
    original_keys = set(router.state_dict())
    bind_softmax_router_recovery(router)
    candidates, gaps = _router_recovery_candidates(
        router,
        logits,
        selected,
        candidate_rank_min=11,
        candidate_rank_max=20,
    )
    assert original_keys == set(router.state_dict())
    logits_after, weights_after, selected_after = router(inputs)
    for actual, expected in (
        (logits_after, logits),
        (weights_after, weights),
        (selected_after, selected),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probabilities = logits.float().softmax(dim=-1)
    remaining = probabilities.clone().scatter_(1, selected, -torch.inf)
    next_scores, expected_candidates = remaining.topk(10, dim=-1)
    torch.testing.assert_close(candidates, expected_candidates, rtol=0, atol=0)
    boundary = probabilities.gather(1, selected).min(dim=-1).values
    torch.testing.assert_close(gaps, boundary[:, None] - next_scores, rtol=0, atol=0)
