# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import pytest
import torch

import gptqmodel.quantization.config as config_module
from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.quantization import QuantizeConfig, Quantizer, ScaleSearchConfig
from gptqmodel.quantization.quantizer import quantize


def _configured_quantizer(config: QuantizeConfig) -> Quantizer:
    quantizer = Quantizer(config)
    quantizer.configure(perchannel=True)
    return quantizer


@pytest.mark.parametrize(
    ("bits", "sym", "expected"),
    [(2, False, True), (2, True, False), (4, False, False)],
)
def test_adjacent_zero_point_policy_follows_resolved_config(bits, sym, expected):
    config = QuantizeConfig(
        bits=bits,
        group_size=128,
        sym=sym,
        offload_to_disk=False,
    )

    for resolved in (config, QuantizeConfig.from_quant_config(config.to_dict())):
        assert resolved.scale_search is ScaleSearchConfig.ACTIVATION
        quantizer = _configured_quantizer(resolved)
        assert quantizer._search_adjacent_zero_points((1 << bits) - 1) is expected


def test_dynamic_w2_asymmetric_override_inherits_adjacent_default():
    pattern = r"+:^model\.layers\.0\.self_attn\.q_proj$"
    config = QuantizeConfig(
        bits=4,
        sym=True,
        dynamic={pattern: {"bits": 2, "sym": False}},
        offload_to_disk=False,
    )

    resolved = clone_gptq_config_for_module(config, "model.layers.0.self_attn.q_proj")

    assert resolved is not None
    assert resolved.bits == 2
    assert resolved.sym is False
    assert resolved.scale_search is ScaleSearchConfig.ACTIVATION
    assert resolved.mse == 2.0
    assert _configured_quantizer(resolved)._search_adjacent_zero_points(3)


def test_dynamic_cache_is_identity_safe_and_bounded(monkeypatch):
    module_name = "model.layers.0.self_attn.q_proj"
    pattern = r"+:^model\.layers\.0\.self_attn\.q_proj$"
    monkeypatch.setattr(config_module, "_DYNAMIC_CACHE_MAX_CONFIGS", 8)

    for index in range(32):
        method = "activation" if index % 2 == 0 else "hessian"
        config = QuantizeConfig(
            scale_search=None,
            dynamic={pattern: {"bits": 2, "sym": False, "scale_search": method}},
            offload_to_disk=False,
        )
        resolved = clone_gptq_config_for_module(config, module_name)
        assert resolved is not None
        assert resolved.scale_search is ScaleSearchConfig(method)
        assert resolved.bits == 2
        assert resolved.sym is False

    with config_module._DYNAMIC_CACHE_LOCK:
        assert len(config_module._DYNAMIC_CACHE) <= 8
        assert all(
            cache_key == id(entry.dynamic)
            for cache_key, entry in config_module._DYNAMIC_CACHE.items()
        )


def test_dynamic_override_cache_has_a_global_memory_bound(monkeypatch):
    monkeypatch.setattr(config_module, "_DYNAMIC_CACHE_MAX_OVERRIDES", 8)
    config = QuantizeConfig(
        bits=2,
        sym=False,
        dynamic={
            r"+:^model\.layers\.\d+\.self_attn\.q_proj$": {"bits": 2, "sym": False}
        },
        offload_to_disk=False,
    )

    for layer_index in range(32):
        assert config.dynamic_get(f"model.layers.{layer_index}.self_attn.q_proj") == {
            "bits": 2,
            "sym": False,
        }

    with config_module._DYNAMIC_CACHE_LOCK:
        assert config_module._DYNAMIC_CACHE_OVERRIDE_COUNT <= 8
        assert config_module._DYNAMIC_CACHE_OVERRIDE_COUNT == sum(
            len(entry.override_cache) for entry in config_module._DYNAMIC_CACHE.values()
        )


def test_dynamic_cache_invalidation_releases_owner_and_override_entries():
    config = QuantizeConfig(
        bits=2,
        sym=False,
        dynamic={r"+:^model\.layers\.0\.self_attn\.q_proj$": {"bits": 2, "sym": False}},
        offload_to_disk=False,
    )
    cache_key = id(config.dynamic)
    assert config.dynamic_get("model.layers.0.self_attn.q_proj") == {
        "bits": 2,
        "sym": False,
    }

    with config_module._DYNAMIC_CACHE_LOCK:
        entry = config_module._DYNAMIC_CACHE[cache_key]
        override_count = len(entry.override_cache)
        total_before = config_module._DYNAMIC_CACHE_OVERRIDE_COUNT

    config._invalidate_dynamic_cache()

    with config_module._DYNAMIC_CACHE_LOCK:
        assert cache_key not in config_module._DYNAMIC_CACHE
        assert (
            config_module._DYNAMIC_CACHE_OVERRIDE_COUNT == total_before - override_count
        )


def test_dynamic_cache_is_thread_safe_across_distinct_configs():
    module_name = "model.layers.0.self_attn.q_proj"
    pattern = r"+:^model\.layers\.0\.self_attn\.q_proj$"
    configs = [
        QuantizeConfig(
            scale_search=None,
            dynamic={pattern: {"bits": 2, "sym": False, "scale_search": method}},
            offload_to_disk=False,
        )
        for method in ("activation", "hessian")
    ]

    def resolve(index):
        expected = (
            ScaleSearchConfig.ACTIVATION
            if index % 2 == 0
            else ScaleSearchConfig.HESSIAN
        )
        for _ in range(32):
            resolved = clone_gptq_config_for_module(configs[index % 2], module_name)
            assert resolved is not None
            assert resolved.scale_search is expected
            assert resolved.bits == 2
            assert resolved.sym is False

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(resolve, range(64)))


@pytest.mark.parametrize("cpu_extension", [False, True])
@torch.inference_mode()
def test_default_w2_asymmetric_search_selects_adjacent_objective_winner(
    monkeypatch,
    cpu_extension,
):
    monkeypatch.setenv("GPTQMODEL_SCALE_SEARCH_CPU", "1" if cpu_extension else "0")
    native_calls = 0
    if cpu_extension:
        quantizer_module = import_module("gptqmodel.quantization.quantizer")
        native_search = quantizer_module._find_params_batched_cpu
        assert native_search is not None

        def counted_native_search(*args, **kwargs):
            nonlocal native_calls
            native_calls += 1
            return native_search(*args, **kwargs)

        monkeypatch.setattr(
            quantizer_module, "_find_params_batched_cpu", counted_native_search
        )

    weights = torch.tensor([[[-1.0, -0.2, 0.2, 1.0]]], dtype=torch.float32)
    importance = torch.tensor([[1.0, 1.0, 1.0, 10.0]], dtype=torch.float32)
    config = QuantizeConfig(
        bits=2,
        group_size=4,
        sym=False,
        adaptive_clipping=None,
        offload_to_disk=False,
    )
    quantizer = _configured_quantizer(config)

    scale, zero = quantizer.find_params_batched(
        weights, weight=True, hessian=importance
    )

    assert config.scale_search is ScaleSearchConfig.ACTIVATION
    assert config.mse == 2.0
    assert zero.item() == 1.0
    selected = quantize(weights, scale, zero, 3, requires_groupwise_processing=False)
    midpoint = quantize(
        weights,
        scale,
        torch.full_like(zero, 2.0),
        3,
        requires_groupwise_processing=False,
    )
    selected_loss = ((selected - weights).square() * importance).sum()
    midpoint_loss = ((midpoint - weights).square() * importance).sum()
    assert selected_loss < midpoint_loss
    assert native_calls == int(cpu_extension)


@torch.inference_mode()
def test_default_w2_asymmetric_scalar_and_batched_search_are_identical(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_SCALE_SEARCH_CPU", "0")
    weights = torch.tensor([[-1.0, -0.2, 0.2, 1.0]], dtype=torch.float32)
    importance = torch.tensor([1.0, 1.0, 1.0, 10.0], dtype=torch.float32)
    config = QuantizeConfig(
        bits=2,
        group_size=4,
        sym=False,
        adaptive_clipping=None,
        offload_to_disk=False,
    )

    scalar = _configured_quantizer(config)
    scalar.find_params(weights, weight=True, hessian=importance)
    batched = _configured_quantizer(config)
    batched_scale, batched_zero = batched.find_params_batched(
        weights.reshape(1, 1, 4),
        weight=True,
        hessian=importance.reshape(1, 4),
    )

    assert torch.equal(scalar.scale.reshape_as(batched_scale), batched_scale)
    assert torch.equal(scalar.zero.reshape_as(batched_zero), batched_zero)
    assert scalar.zero.item() == 1.0
