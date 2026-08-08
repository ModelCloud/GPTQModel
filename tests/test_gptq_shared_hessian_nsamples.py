# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for shared-Hessian sample accounting and ownership."""

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import (
    AdaptiveClippingConfig,
    AdaptiveDampingConfig,
    ExpertsRoutingBypass,
    HessianConfig,
    LengthAwareConfig,
    LengthAwareMode,
    MoEConfig,
    MoEExecutionConfig,
    QuantizeConfig,
    VramStrategy,
)


def _make_processor(*, group_size: int = 64, module_count: int = 4):
    """Build a real shared-Hessian subset with the reported MaCa crash config."""

    calibration_lengths = [length for length in (64, 128, 256, 512, 1024, 2048) for _ in range(16)]
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        desc_act=False,
        act_group_aware=True,
        scale_search="activation",
        moe=MoEConfig(routing=ExpertsRoutingBypass(), execution=MoEExecutionConfig(batch_size=None)),
        dense_vram_strategy=VramStrategy.BALANCED,
        moe_vram_strategy=VramStrategy.BALANCED,
        adaptive_damping=AdaptiveDampingConfig(enabled=False),
        adaptive_clipping=AdaptiveClippingConfig(enabled=False),
        hessian=HessianConfig(
            length_aware=LengthAwareConfig.from_lengths(
                calibration_lengths,
                mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
                target_bucket_count=6,
                bucket_weight_exponent=0.2,
            )
        ),
    )
    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    subset = {}
    for index in range(module_count):
        expert_index = index // 2
        projection = "gate_proj" if index % 2 == 0 else "up_proj"
        name = f"mlp.experts.{expert_index}.{projection}"
        named = NamedModule(
            nn.Linear(8, 8, bias=False),
            name=name,
            full_name=f"model.layers.0.{name}",
            layer_index=0,
        )
        named.state["module_tree_flags"] = frozenset(
            {"routed", projection.removesuffix("_proj"), "expert_gate=experts._apply_gate"}
        )
        named.state["module_tree_expert_group"] = f"routed-expert-{expert_index}"
        processor.preprocess(named)
        subset[name] = named

    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    shared_states = [processor.tasks[name]._shared_hessian_state for name in subset]
    assert shared_states[0] is not None
    assert all(state is shared_states[0] for state in shared_states)
    return processor, subset, list(subset)


def _capture_active_modules(processor, subset, active_names, *, token_count: int = 10):
    """Populate shared state through the production forward-hook path."""

    activations = torch.arange(token_count * 8, dtype=torch.float32).reshape(1, token_count, 8) / 100
    activations[..., -1] = 0  # Real dead input column used by the ownership regression.
    for name in active_names:
        module = subset[name].module
        processor.pre_process_fwd_hook(name)(module, (activations,), module(activations))
    return activations


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("materialization_order", ["inactive_first", "cached_after_active"])
def test_inactive_shared_hessian_task_keeps_zero_samples_and_uses_rtn(group_size, materialization_order):
    """An inactive module must never inherit a peer's shared sample count."""

    processor, subset, names = _make_processor(group_size=group_size)
    active_names = names[:2]
    inactive_name = names[2]
    _capture_active_modules(processor, subset, active_names)

    inactive_task = processor.tasks[inactive_name]
    shared_state = inactive_task._shared_hessian_state
    assert shared_state["sample_counts"][torch.device("cpu")] == 10
    assert processor.shared_hessian_stats()["batch_misses"] == 1
    assert processor.shared_hessian_stats()["batch_hits"] == 1
    assert all(processor.tasks[name].nsamples == 10 for name in active_names)
    assert inactive_task.nsamples == 0
    assert inactive_task.fwd_counter == 0

    # Cover both production materialization paths without patching state: the
    # inactive task either builds the group Hessian or borrows the clean cache.
    if materialization_order == "cached_after_active":
        active_task = processor.tasks[active_names[0]]
        active_task.finalize_hessian(target_device=torch.device("cpu"))
        assert active_task.nsamples == 10
    inactive_task.finalize_hessian(target_device=torch.device("cpu"))

    assert inactive_task.nsamples == 0
    assert inactive_task.fwd_counter == 0

    quantized, scales, zeros, g_idx, _duration, loss, _damp, nsamples = inactive_task.quantize(blocksize=8)
    assert nsamples == 0
    assert loss.startswith("fallback(rtn):")
    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scales).all()
    assert torch.isfinite(zeros).all()
    assert g_idx.numel() == inactive_task.columns

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)


def test_shared_hessian_dead_column_repair_does_not_mutate_borrowed_state():
    """Task-local dead-column repair must leave the processor-owned Hessian immutable."""

    processor, subset, names = _make_processor(group_size=64, module_count=2)
    _capture_active_modules(processor, subset, names)
    task = processor.tasks[names[0]]
    task.finalize_hessian(target_device=torch.device("cpu"))

    shared_state = task._shared_hessian_state
    shared_hessian = shared_state["H"]
    expected = shared_hessian.clone()
    assert shared_hessian.diagonal()[-1].item() == 0

    quantized, *_ = task.quantize(blocksize=8)

    torch.testing.assert_close(shared_state["H"], expected, rtol=0, atol=0)
    assert shared_state["H"].data_ptr() == shared_hessian.data_ptr()
    assert torch.isfinite(quantized).all()
    assert task.nsamples == 10

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)
