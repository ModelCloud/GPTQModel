# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig, VramStrategy


def _prepare_dataset_func(**kwargs):
    return kwargs["calibration_dataset"]


def _make_awq_processor(*, enable_activation_x_mean_cache: bool = True) -> AWQProcessor:
    qcfg = QuantizeConfig(
        bits=4,
        group_size=8,
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        device="cpu",
        dense_vram_strategy=VramStrategy.EXCLUSIVE,
        enable_activation_x_mean_cache=enable_activation_x_mean_cache,
    )
    model = torch.nn.Module()
    gptq_model = SimpleNamespace(
        model=model,
        qlinear_kernel=None,
        rotary_embedding=None,
        lm_head="lm_head",
    )
    return AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=_prepare_dataset_func,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        gptq_model=gptq_model,
        model=model,
    )


def _test_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for AWQ shared activation GPU capture test")

    requested = int(os.environ.get("GPTQMODEL_TEST_CUDA_INDEX", "6"))
    fallback = int(os.environ.get("GPTQMODEL_TEST_CUDA_FALLBACK_INDEX", "7"))
    count = torch.cuda.device_count()
    if requested < count:
        return torch.device("cuda", requested)
    if fallback < count:
        return torch.device("cuda", fallback)
    return torch.device("cuda", 0)


def _make_same_input_subset(processor: AWQProcessor, names) -> dict[str, NamedModule]:
    subset = {}
    for name in names:
        module = torch.nn.Linear(8, 4, bias=False).eval()
        named = NamedModule(
            module,
            name=name,
            full_name=f"model.layers.0.{name}",
            layer_index=0,
        )
        processor.preprocess(named)
        subset[name] = named
    return subset


def test_awq_same_input_modules_share_capture_and_activation_mean():
    processor = _make_awq_processor()
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    subset = _make_same_input_subset(processor, names)
    state = _AWQLayerState(modules=subset)

    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    processor._set_current_batch_index(0)
    x = torch.randn(2, 3, 8)

    for name in names:
        module = subset[name].module
        processor.pre_process_fwd_hook(name)(module, (x,), module(x))

    stats = processor.shared_activation_stats()
    assert stats["feature_requests"] == len(names)
    assert stats["feature_misses"] == 1
    assert stats["feature_hits"] == len(names) - 1

    captured_ids = {id(processor.tasks[name]["inputs"][0]) for name in names}
    assert len(captured_ids) == 1

    input_features = processor._layer_input_features(state)
    data_ptrs = {input_features[name].data_ptr() for name in names}
    assert len(data_ptrs) == len(names)
    for name in names:
        assert torch.equal(input_features[name], x)

    expected = x.abs().view(-1, x.shape[-1]).to(torch.float32).mean(dim=0).to(x.dtype)
    means = [processor._compute_activation_x_mean(input_features[name]) for name in names]
    for mean in means:
        assert torch.allclose(mean, expected)
    assert means[1].data_ptr() == means[0].data_ptr()
    assert means[2].data_ptr() == means[0].data_ptr()

    stats = processor.shared_activation_stats()
    assert stats["x_mean_requests"] == len(names)
    assert stats["x_mean_misses"] == 1
    assert stats["x_mean_hits"] == len(names) - 1

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)


def test_awq_activation_x_mean_toggle_disabled_uses_isolated_captures_and_means():
    processor = _make_awq_processor(enable_activation_x_mean_cache=False)
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    subset = _make_same_input_subset(processor, names)
    state = _AWQLayerState(modules=subset)

    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    processor._set_current_batch_index(0)
    x = torch.randn(2, 3, 8)

    for name in names:
        module = subset[name].module
        processor.pre_process_fwd_hook(name)(module, (x,), module(x))

    stats = processor.shared_activation_stats()
    assert stats["feature_requests"] == 0
    assert stats["feature_misses"] == 0
    assert stats["feature_hits"] == 0

    captured_ids = {id(processor.tasks[name]["inputs"][0]) for name in names}
    assert len(captured_ids) == len(names)

    input_features = processor._layer_input_features(state)
    expected = x.abs().view(-1, x.shape[-1]).to(torch.float32).mean(dim=0).to(x.dtype)
    means = [processor._compute_activation_x_mean(input_features[name]) for name in names]
    for mean in means:
        assert torch.equal(mean, expected)
    assert len({mean.data_ptr() for mean in means}) == len(names)

    stats = processor.shared_activation_stats()
    assert stats["x_mean_requests"] == 0
    assert stats["x_mean_misses"] == 0
    assert stats["x_mean_hits"] == 0

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)


def test_awq_activation_x_mean_toggle_preserves_exact_activation_math():
    names = ["mlp.gate_proj", "mlp.up_proj"]
    x = torch.randn(2, 5, 8)

    def run_case(enabled: bool):
        processor = _make_awq_processor(enable_activation_x_mean_cache=enabled)
        subset = _make_same_input_subset(processor, names)
        state = _AWQLayerState(modules=subset)
        processor.prepare_subset(subset, subset_index=0, subset_total=1)
        processor._set_current_batch_index(0)
        for name in names:
            module = subset[name].module
            processor.pre_process_fwd_hook(name)(module, (x,), module(x))
        input_features = processor._layer_input_features(state)
        means = {
            name: processor._compute_activation_x_mean(input_features[name]).detach().clone()
            for name in names
        }
        stats = processor.shared_activation_stats()
        processor.cleanup_subset(subset, subset_index=0, subset_total=1)
        return input_features, means, stats

    enabled_features, enabled_means, enabled_stats = run_case(True)
    disabled_features, disabled_means, disabled_stats = run_case(False)

    assert enabled_stats["feature_hits"] == len(names) - 1
    assert enabled_stats["x_mean_hits"] == len(names) - 1
    assert disabled_stats["feature_requests"] == 0
    assert disabled_stats["x_mean_requests"] == 0

    for name in names:
        assert torch.equal(enabled_features[name], disabled_features[name])
        assert torch.equal(enabled_means[name], disabled_means[name])


def test_awq_feature_cache_does_not_alias_distinct_tensor_objects(monkeypatch):
    processor = _make_awq_processor()
    processor._set_current_batch_index(0)

    # Force the storage/view fingerprint to collide. The cache must still keep
    # distinct Tensor objects isolated because CUDA may recycle a storage
    # pointer for unrelated activations within one subset forward.
    monkeypatch.setattr(
        AWQProcessor,
        "_tensor_cache_fingerprint",
        staticmethod(lambda _tensor: ("forced-collision",)),
    )

    first = torch.ones(1, 2, 8)
    second = torch.zeros(1, 2, 8)
    cached_first, first_key = processor._cache_input_feature(first)
    cached_second, second_key = processor._cache_input_feature(second)

    assert first_key != second_key
    assert torch.equal(cached_first, first)
    assert torch.equal(cached_second, second)

    stats = processor.shared_activation_stats()
    assert stats["feature_requests"] == 2
    assert stats["feature_misses"] == 2
    assert stats["feature_hits"] == 0


def test_awq_activation_x_mean_toggle_serializes_as_process_config():
    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        enable_activation_x_mean_cache=False,
    )
    assert qcfg.enable_activation_x_mean_cache is False

    restored = QuantizeConfig.from_quant_config(qcfg.to_dict())
    assert restored.enable_activation_x_mean_cache is False


@pytest.mark.cuda
def test_awq_cuda_same_input_capture_copies_to_cpu_once():
    device = _test_cuda_device()
    torch.cuda.set_device(device)
    processor = _make_awq_processor()
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    subset = _make_same_input_subset(processor, names)

    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    processor._set_current_batch_index(0)
    x = torch.randn(2, 3, 8, device=device)

    for name in names:
        module = subset[name].module.to(device)
        processor.pre_process_fwd_hook(name)(module, (x,), module(x))

    stats = processor.shared_activation_stats()
    assert stats["feature_misses"] == 1
    assert stats["feature_hits"] == len(names) - 1

    captured = [processor.tasks[name]["inputs"][0] for name in names]
    assert all(feature.device.type == "cpu" for feature in captured)
    assert len({id(feature) for feature in captured}) == 1

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)


def test_awq_variable_length_shared_capture_clones_latest_feature():
    processor = _make_awq_processor()
    names = ["mlp.gate_proj", "mlp.up_proj"]
    subset = _make_same_input_subset(processor, names)
    state = _AWQLayerState(modules=subset)

    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    for batch_index, seq_len in enumerate([5, 3]):
        processor._set_current_batch_index(batch_index)
        x = torch.randn(1, seq_len, 8)
        for name in names:
            module = subset[name].module
            processor.pre_process_fwd_hook(name)(module, (x,), module(x))

    input_features = processor._layer_input_features(state)
    assert input_features[names[0]].shape == (1, 3, 8)
    assert input_features[names[1]].shape == (1, 3, 8)
    assert input_features[names[0]].data_ptr() != input_features[names[1]].data_ptr()

    before = input_features[names[1]].clone()
    input_features[names[0]].add_(1)
    assert torch.equal(input_features[names[1]], before)

    processor.cleanup_subset(subset, subset_index=0, subset_total=1)
