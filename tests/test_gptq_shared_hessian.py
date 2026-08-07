# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig


pytestmark = pytest.mark.cuda


def _test_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for shared Hessian GPTQ test")

    requested = int(os.environ.get("GPTQMODEL_TEST_CUDA_INDEX", "6"))
    fallback = int(os.environ.get("GPTQMODEL_TEST_CUDA_FALLBACK_INDEX", "7"))
    count = torch.cuda.device_count()
    if requested < count:
        return torch.device("cuda", requested)
    if fallback < count:
        return torch.device("cuda", fallback)
    return torch.device("cuda", 0)


def _make_qkv_processor(device: torch.device, *, enable_shared_hessian_cache: bool, act_group_aware: bool = True):
    qcfg = QuantizeConfig(
        bits=4,
        group_size=8,
        sym=True,
        desc_act=False,
        act_group_aware=act_group_aware,
        hessian=HessianConfig(staging_dtype=torch.float32),
        enable_shared_hessian_cache=enable_shared_hessian_cache,
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

    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    out_features = [12, 10, 14]
    subset = {}
    for name, out_feature in zip(names, out_features):
        module = torch.nn.Linear(16, out_feature, bias=False, dtype=torch.float16).to(device).eval()
        named = NamedModule(
            module,
            name=name,
            full_name=f"model.layers.0.{name}",
            layer_index=0,
        )
        processor.preprocess(named)
        subset[name] = named

    return processor, subset, names


def test_llama_same_input_modules_share_hessian_accumulation_and_inverse():
    llama_blocks = LlamaQModel.build_layer_modules(LlamaQModel.module_tree)
    assert ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] in llama_blocks

    device = _test_device()
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    processor, subset, names = _make_qkv_processor(device, enable_shared_hessian_cache=True)
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    x = torch.randn(2, 8, 16, device=device, dtype=torch.float16)
    for name in names:
        output = subset[name].module(x)
        hook = processor.pre_process_fwd_hook(name)
        hook(subset[name].module, (x,), output)

    stats = processor.shared_hessian_stats()
    assert stats["batch_misses"] == 1
    assert stats["batch_hits"] == len(names) - 1

    shared_state = None
    for name in names:
        task = processor.tasks[name]
        assert task.fwd_counter == 1
        assert task.nsamples == x.shape[0] * x.shape[1]
        assert task._device_hessian_partials == {}
        if shared_state is None:
            shared_state = task._shared_hessian_state
        assert task._shared_hessian_state is shared_state

    assert shared_state is not None
    assert list(shared_state["partials"].keys()) == [device]
    assert shared_state["sample_counts"][device] == x.shape[0] * x.shape[1]
    assert shared_state["partials"][device].shape == (16, 16)

    materialized = [processor.tasks[name].finalize_hessian(target_device=device) for name in names]
    for hessian in materialized[1:]:
        assert hessian.data_ptr() == materialized[0].data_ptr()

    for name in names:
        task = processor.tasks[name]
        quantized, scales, zeros, g_idx, *_ = task.quantize(blocksize=8)
        assert quantized.shape == subset[name].module.weight.shape
        assert scales.numel() > 0
        assert zeros.numel() > 0
        assert g_idx.numel() == 16

    stats = processor.shared_hessian_stats()
    assert stats["inverse_misses"] == 1
    assert stats["inverse_hits"] == len(names) - 1
    assert processor._shared_hessian_inverse_cache == {}
    assert processor._shared_hessian_inverse_ref_counts == {}

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)


def test_gptq_shared_hessian_toggle_disabled_uses_isolated_hessians():
    device = _test_device()
    torch.cuda.set_device(device)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    processor, subset, names = _make_qkv_processor(device, enable_shared_hessian_cache=False)
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    x = torch.randn(2, 8, 16, device=device, dtype=torch.float16)
    for name in names:
        output = subset[name].module(x)
        processor.pre_process_fwd_hook(name)(subset[name].module, (x,), output)

    stats = processor.shared_hessian_stats()
    assert stats["batch_requests"] == 0
    assert stats["batch_hits"] == 0
    assert stats["batch_misses"] == 0

    materialized = [processor.tasks[name].finalize_hessian(target_device=device) for name in names]
    assert len({hessian.data_ptr() for hessian in materialized}) == len(names)
    for hessian in materialized[1:]:
        torch.testing.assert_close(hessian, materialized[0], rtol=0.0, atol=0.0)

    for name in names:
        task = processor.tasks[name]
        assert task._shared_hessian_state is None
        assert task._shared_hessian_inverse_cache is None
        assert task.fwd_counter == 1
        assert task.nsamples == x.shape[0] * x.shape[1]

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)


def test_gptq_shared_hessian_rejects_overlapping_subset_setup():
    """A new subset cannot invalidate processor-wide caches used by active workers."""

    device = _test_device()
    processor, subset, _ = _make_qkv_processor(device, enable_shared_hessian_cache=True)
    processor.prepare_subset(subset, subset_index=0, subset_total=2)

    with pytest.raises(RuntimeError, match="cannot overlap an active subset"):
        processor.prepare_subset(subset, subset_index=1, subset_total=2)

    processor.cleanup_subset(subset, subset_index=0, subset_total=2)


def test_gptq_shared_hessian_cleanup_accepts_in_place_subset_pruning():
    """Stage pruning mutates the subset mapping without changing its lifecycle identity."""

    device = _test_device()
    processor, subset, names = _make_qkv_processor(device, enable_shared_hessian_cache=True)
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    subset.pop(names[0])
    processor.cleanup_subset(subset, subset_index=0, subset_total=1)

    assert processor._active_shared_hessian_subset is None
    assert processor._shared_hessian_states == {}
    assert processor._shared_hessian_inverse_cache == {}


def test_gptq_shared_hessian_mismatched_cleanup_fails_after_releasing_state():
    """Misuse is reported without leaking the active subset's shared pointers."""

    device = _test_device()
    processor, subset, names = _make_qkv_processor(device, enable_shared_hessian_cache=True)
    processor.prepare_subset(subset, subset_index=0, subset_total=1)

    with pytest.raises(RuntimeError, match="does not match the active subset"):
        processor.cleanup_subset(dict(subset), subset_index=0, subset_total=1)

    assert processor._active_shared_hessian_subset is None
    assert processor._shared_hessian_states == {}
    for name in names:
        task = processor.tasks[name]
        assert task._shared_hessian_state is None
        assert task._shared_hessian_inverse_cache is None


def test_gptq_shared_hessian_toggle_preserves_exact_quantization_math():
    device = _test_device()
    torch.cuda.set_device(device)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)

    x = torch.randn(2, 8, 16, device=device, dtype=torch.float16)
    base_weights = [
        torch.randn(out_features, 16, device=device, dtype=torch.float16)
        for out_features in (12, 10, 14)
    ]

    def run_case(enabled: bool):
        processor, subset, names = _make_qkv_processor(
            device,
            enable_shared_hessian_cache=enabled,
            act_group_aware=False,
        )
        for name, weight in zip(names, base_weights):
            subset[name].module.weight.data.copy_(weight)

        processor.prepare_subset(subset, subset_index=0, subset_total=1)
        for name in names:
            output = subset[name].module(x)
            processor.pre_process_fwd_hook(name)(subset[name].module, (x,), output)

        hessians = [processor.tasks[name].finalize_hessian(target_device=device).detach().clone() for name in names]
        quantized = {}
        for name in names:
            task = processor.tasks[name]
            qweight, scales, zeros, g_idx, *_ = task.quantize(blocksize=8)
            quantized[name] = (
                qweight.detach().cpu(),
                scales.detach().cpu(),
                zeros.detach().cpu(),
                g_idx.detach().cpu(),
            )
        stats = processor.shared_hessian_stats()
        processor.cleanup_subset(subset, subset_index=0, subset_total=1)
        return hessians, quantized, stats

    enabled_hessians, enabled_quantized, enabled_stats = run_case(True)
    disabled_hessians, disabled_quantized, disabled_stats = run_case(False)

    assert enabled_stats["batch_hits"] == len(base_weights) - 1
    assert disabled_stats["batch_requests"] == 0

    for enabled_hessian, disabled_hessian in zip(enabled_hessians, disabled_hessians):
        torch.testing.assert_close(enabled_hessian, disabled_hessian, rtol=0.0, atol=0.0)

    for name in enabled_quantized:
        for enabled_tensor, disabled_tensor in zip(enabled_quantized[name], disabled_quantized[name]):
            assert torch.equal(enabled_tensor, disabled_tensor)


def test_gptq_shared_hessian_toggle_serializes_as_process_config():
    qcfg = QuantizeConfig(
        enable_shared_hessian_cache=False,
    )
    assert qcfg.enable_shared_hessian_cache is False
    assert qcfg.moe_parallel_input_capture is True

    restored = QuantizeConfig.from_quant_config(qcfg.to_dict())
    assert restored.enable_shared_hessian_cache is False
    assert restored.moe_parallel_input_capture is True

    opted_out = QuantizeConfig(moe_parallel_input_capture=False)
    restored_opt_out = QuantizeConfig.from_quant_config(opted_out.to_dict())
    assert restored_opt_out.moe_parallel_input_capture is False
