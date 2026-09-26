# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import pickle
import weakref
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import pcre
import pytest
import torch.nn as nn

from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel
from gptqmodel.quantization.config import (
    _DYNAMIC_ALL_EXACT_CACHE,
    _DYNAMIC_EXACT_LOOKUP_CACHE,
    _DYNAMIC_IDENTITY_CACHE,
    _DYNAMIC_OVERRIDE_CACHE,
    _DYNAMIC_OVERRIDE_CACHE_MAXSIZE,
    _DYNAMIC_PATTERN_CACHE,
    _DYNAMIC_PATTERN_CACHE_MAXSIZE,
    _DYNAMIC_REGEX_PATTERN_CACHE,
    QuantizeConfig,
    _dynamic_value_fingerprint,
    _TrackedDict,
    _TrackedList,
    _dynamic_override_cache_for,
    configure_dynamic_override_cache,
    configure_dynamic_override_cache_for_model,
    dynamic_get,
    dynamic_override_cache_stats,
    log_dynamic_override_cache_stats,
)


def _clear_dynamic_caches():
    """Drop the global dynamic pattern/override caches so tests are independent."""
    _DYNAMIC_PATTERN_CACHE.clear()
    _DYNAMIC_EXACT_LOOKUP_CACHE.clear()
    _DYNAMIC_REGEX_PATTERN_CACHE.clear()
    _DYNAMIC_OVERRIDE_CACHE.clear()
    _DYNAMIC_ALL_EXACT_CACHE.clear()
    _DYNAMIC_IDENTITY_CACHE.clear()


@pytest.fixture(autouse=True)
def clear_caches():
    original_capacity = _DYNAMIC_OVERRIDE_CACHE.maxsize
    _clear_dynamic_caches()
    _DYNAMIC_OVERRIDE_CACHE.maxsize = _DYNAMIC_OVERRIDE_CACHE_MAXSIZE
    yield
    _clear_dynamic_caches()
    _DYNAMIC_OVERRIDE_CACHE.maxsize = original_capacity


def test_dynamic_cache_default_and_grow_only_sizing():
    assert _DYNAMIC_OVERRIDE_CACHE.maxsize == 8192
    assert configure_dynamic_override_cache(module_count=3_000) == 8192
    assert configure_dynamic_override_cache(module_count=8_000) == 16384
    assert configure_dynamic_override_cache(module_count=50_000, layer_count=64, expert_count=256) == 65536
    assert configure_dynamic_override_cache(module_count=1_000) == 65536


def test_dynamic_cache_keeps_negative_and_no_match_results():
    cfg = QuantizeConfig(dynamic={r"-:^model\.skip$": {}, r"+:^model\.hit$": {"bits": 2}})
    original_match = pcre.Pattern.match
    match_calls = 0

    def counted_match(pattern, name):
        nonlocal match_calls
        match_calls += 1
        return original_match(pattern, name)

    with patch.object(pcre.Pattern, "match", counted_match):
        assert cfg.dynamic_get("model.skip") is False
        assert cfg.dynamic_get("model.miss") is None
        first_pass = match_calls
        assert cfg.dynamic_get("model.skip") is False
        assert cfg.dynamic_get("model.miss") is None
        assert match_calls == first_pass
    assert dynamic_override_cache_stats(cfg.dynamic)["hits"] == 2
    assert dynamic_override_cache_stats(cfg.dynamic)["misses"] == 2


def test_dynamic_cache_second_pass_avoids_regex_for_120_layer_512_expert_model():
    class FakeMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({"moe": nn.ModuleDict({"experts": nn.ModuleList([
                    nn.Module() for _ in range(512)
                ])})})
                for _ in range(120)
            ])

    model = FakeMoEModel()
    cfg = QuantizeConfig(dynamic={r"+:^layers\.\d+\.moe\.experts\.\d+$": {"bits": 2}})
    capacity = configure_dynamic_override_cache_for_model(model, cfg)
    names = [name for name, _ in model.named_modules(remove_duplicate=False) if name]
    expert_names = [name for name in names if ".experts." in name]
    assert len(expert_names) == 120 * 512
    assert capacity == 131072
    assert capacity >= len(names)

    original_match = pcre.Pattern.match
    match_calls = 0

    def counted_match(pattern, name):
        nonlocal match_calls
        match_calls += 1
        return original_match(pattern, name)

    with patch.object(pcre.Pattern, "match", counted_match):
        for name in names:
            assert cfg.dynamic_get(name, "bits", cfg.bits) == (2 if ".experts." in name else cfg.bits)
        first_pass_calls = match_calls
        first_pass_stats = dynamic_override_cache_stats(cfg.dynamic)
        for name in names:
            assert cfg.dynamic_get(name, "bits", cfg.bits) == (2 if ".experts." in name else cfg.bits)

    stats = dynamic_override_cache_stats(cfg.dynamic)
    assert first_pass_calls == len(names)
    assert match_calls == first_pass_calls
    assert first_pass_stats["misses"] == len(names)
    assert stats["misses"] == len(names)
    assert stats["hits"] == len(names)
    assert stats["evictions"] == 0
    assert stats["peak_size"] == len(names)


def test_dynamic_cache_sizes_fused_moe_from_logical_plan():
    class FusedMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(nn.Module() for _ in range(120))
            self.config = SimpleNamespace(n_routed_experts=512)

    model = FusedMoEModel()
    plan = DeepSeekV3QModel.full_layer_modules(model.config)
    physical_count = sum(bool(name) for name, _ in model.named_modules())
    assert physical_count < 8192

    dynamic = {f"+:^never\\.{i}\\..*$": {"bits": 3} for i in range(15)}
    dynamic[r"+:^model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj$"] = {"bits": 2}
    cfg = QuantizeConfig(dynamic=dynamic)
    capacity = configure_dynamic_override_cache_for_model(
        model, cfg, layer_modules=plan, layer_prefixes=DeepSeekV3QModel.extract_layers_node(),
    )
    assert capacity == 262144

    names = (
        f"model.layers.{layer}.mlp.experts.{expert}.gate_proj"
        for layer in range(120) for expert in range(512)
    )
    original_match = pcre.Pattern.match
    match_calls = 0

    def counted_match(pattern, name):
        nonlocal match_calls
        match_calls += 1
        return original_match(pattern, name)

    with patch.object(pcre.Pattern, "match", counted_match):
        for name in names:
            assert cfg.dynamic_get(name, "bits", cfg.bits) == 2
        first_pass_calls = match_calls
        for layer in range(120):
            for expert in range(512):
                name = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj"
                assert cfg.dynamic_get(name, "bits", cfg.bits) == 2

    stats = dynamic_override_cache_stats(cfg.dynamic)
    assert first_pass_calls == 120 * 512 * len(dynamic)
    assert match_calls == first_pass_calls
    assert stats["misses"] == 120 * 512
    assert stats["hits"] == 120 * 512
    assert stats["evictions"] == 0


def test_dynamic_cache_isolated_between_configs_for_same_module():
    module_name = "model.layers.0.mlp.proj"
    cfg_a = QuantizeConfig(dynamic={r"+:^model\.layers\.\d+\.mlp\.proj$": {"bits": 2}})
    cfg_b = QuantizeConfig(dynamic={r"+:^model\.layers\.\d+\.mlp\.proj$": {"bits": 3}})
    assert cfg_a.dynamic_get(module_name, "bits") == 2
    assert cfg_b.dynamic_get(module_name, "bits") == 3
    assert cfg_a.dynamic_get(module_name, "bits") == 2
    assert dynamic_override_cache_stats(cfg_a.dynamic)["size"] == 1
    assert dynamic_override_cache_stats(cfg_b.dynamic)["size"] == 1


def test_active_configs_have_independent_cache_capacity_and_lifetime():
    cfg_a = QuantizeConfig(dynamic={r"+:^module\.\d+$": {"bits": 2}})
    cfg_b = QuantizeConfig(dynamic={r"+:^module\.\d+$": {"bits": 3}})
    for cfg in (cfg_a, cfg_b):
        assert configure_dynamic_override_cache(module_count=12_000, dynamic=cfg.dynamic) == 16384

    original_match = pcre.Pattern.match
    match_calls = 0

    def counted_match(pattern, name):
        nonlocal match_calls
        match_calls += 1
        return original_match(pattern, name)

    with patch.object(pcre.Pattern, "match", counted_match):
        for index in range(12_000):
            name = f"module.{index}"
            assert cfg_a.dynamic_get(name, "bits") == 2
            assert cfg_b.dynamic_get(name, "bits") == 3
        first_pass_calls = match_calls
        for index in range(12_000):
            name = f"module.{index}"
            assert cfg_a.dynamic_get(name, "bits") == 2
            assert cfg_b.dynamic_get(name, "bits") == 3

    assert first_pass_calls == 24_000
    assert match_calls == first_pass_calls
    for cfg in (cfg_a, cfg_b):
        stats = dynamic_override_cache_stats(cfg.dynamic)
        assert (stats["size"], stats["hits"], stats["misses"], stats["evictions"]) == (12_000, 12_000, 12_000, 0)
    assert dynamic_override_cache_stats()["size"] == 0
    assert _DYNAMIC_OVERRIDE_CACHE.maxsize == 8192

    mapping_ref = weakref.ref(cfg_a.dynamic)
    cache_ref = weakref.ref(_dynamic_override_cache_for(cfg_a.dynamic))
    del cfg_a
    gc.collect()
    assert mapping_ref() is None
    assert cache_ref() is None


def test_standalone_dynamic_lookup_keeps_bounded_global_fallback():
    dynamic = {r"+:^module\.\d+$": {"bits": 2}}
    for index in range(_DYNAMIC_OVERRIDE_CACHE_MAXSIZE + 32):
        assert dynamic_get(dynamic, f"module.{index}", "bits") == 2
    stats = dynamic_override_cache_stats()
    assert stats["size"] == _DYNAMIC_OVERRIDE_CACHE_MAXSIZE
    assert stats["evictions"] == 32


def test_dynamic_cache_log_reports_since_start_counters():
    cfg = QuantizeConfig(dynamic={r"+:^module\.\d+$": {"bits": 2}})
    cfg.dynamic_get("module.0")
    before = dynamic_override_cache_stats(cfg.dynamic)
    cfg.dynamic_get("module.0")
    cfg.dynamic_get("module.1")

    with patch("gptqmodel.quantization.config.log.debug") as mock_debug:
        log_dynamic_override_cache_stats(before, dynamic=cfg.dynamic)

    assert mock_debug.call_args.args[1:5] == (2, 1, 1, 0)


def _exact_pattern(module_name: str) -> str:
    """Build an anchored regex that matches only `module_name` using escaped dots."""
    escaped = module_name.replace(".", "\\.")
    return f"+:^{escaped}$"


def test_dynamic_exact_patterns_bypass_pcre_match():
    """Fully-exact dynamic configs must not call pcre.Pattern.match per module."""
    modules = [f"model.layers.{i}.mlp.down_proj" for i in range(100)]
    dynamic = {_exact_pattern(name): {"bits": 2} for name in modules}
    cfg = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)

    with patch.object(pcre.Pattern, "match") as mock_match:
        for name in modules:
            assert cfg.dynamic_get(name, "bits", cfg.bits) == 2
        assert cfg.dynamic_get("model.layers.unknown.mlp.down_proj", "bits", cfg.bits) == cfg.bits
        mock_match.assert_not_called()


def test_dynamic_negative_exact_pattern_bypasses_pcre_match():
    """A negative exact pattern should short-circuit without regex matching."""
    dynamic = {
        "-:^model\\.layers\\.0\\.mlp\\.down_proj$": {},
        "+:^model\\.layers\\.1\\.mlp\\.down_proj$": {"bits": 2},
    }
    cfg = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)

    with patch.object(pcre.Pattern, "match") as mock_match:
        assert cfg.dynamic_get("model.layers.0.mlp.down_proj", "bits", cfg.bits) is False
        assert cfg.dynamic_get("model.layers.1.mlp.down_proj", "bits", cfg.bits) == 2
        assert cfg.dynamic_get("model.layers.2.mlp.down_proj", "bits", cfg.bits) == cfg.bits
        mock_match.assert_not_called()


def test_dynamic_mixed_uses_pcre_only_for_regex_patterns():
    """Mixed exact + regex configs should only call match for regex patterns."""
    dynamic = {
        "+:^model\\.layers\\.0\\.mlp\\.down_proj$": {"bits": 2},
        "+:^model\\.layers\\.\\d+\\.mlp\\.gate_proj$": {"bits": 8},
    }
    cfg = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)

    original_match = pcre.Pattern.match
    match_calls = []

    def _counted_match(self, string):
        match_calls.append(string)
        return original_match(self, string)

    with patch.object(pcre.Pattern, "match", _counted_match):
        # Exact match should resolve without ever calling pcre.match.
        assert cfg.dynamic_get("model.layers.0.mlp.down_proj", "bits", cfg.bits) == 2
        assert not match_calls

        # This matches the regex pattern; one pcre.match call is expected.
        assert cfg.dynamic_get("model.layers.5.mlp.gate_proj", "bits", cfg.bits) == 8
        assert len(match_calls) == 1


def test_dynamic_prefix_pattern_is_not_exact():
    """A pattern without a trailing `$` is a prefix regex and must be compiled/used as one."""
    dynamic = {"+:^model\\.layers\\.1": {"bits": 2}}
    cfg = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)

    # Should match the prefix (model.layers.1) but not a different layer.
    assert cfg.dynamic_get("model.layers.1.mlp.down_proj", "bits", cfg.bits) == 2
    assert cfg.dynamic_get("model.layers.10.mlp.down_proj", "bits", cfg.bits) == 2
    assert cfg.dynamic_get("model.layers.2.mlp.down_proj", "bits", cfg.bits) == cfg.bits


def test_dynamic_mixed_ordering_respected():
    """When both a regex and an exact pattern match, the earlier one in the dict wins."""
    # Regex first, then exact override for layer 5.
    dynamic = {
        "+:^model\\.layers\\.\\d+\\.mlp\\.gate_proj$": {"bits": 8},
        "+:^model\\.layers\\.5\\.mlp\\.gate_proj$": {"bits": 2},
    }
    cfg = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)

    # Regex appears first, so it wins for layer 5.
    assert cfg.dynamic_get("model.layers.5.mlp.gate_proj", "bits", cfg.bits) == 8

    # Exact appears second, so it wins when the regex does not match.
    dynamic_reordered = {
        "+:^model\\.layers\\.5\\.mlp\\.gate_proj$": {"bits": 2},
        "+:^model\\.layers\\.\\d+\\.mlp\\.gate_proj$": {"bits": 8},
    }
    cfg2 = QuantizeConfig(dynamic=dynamic_reordered, bits=4, group_size=128, sym=False)
    assert cfg2.dynamic_get("model.layers.5.mlp.gate_proj", "bits", cfg2.bits) == 2


def test_dynamic_large_exact_config_no_pcre_regression():
    """Reproduce a large exact-only dynamic config and verify zero pcre.match calls."""
    pattern_count = 2270
    module_count = 36432

    # Build a set of exact patterns and a larger set of module names to resolve.
    patterns = {
        _exact_pattern(f"model.layers.{i}.mlp.down_proj"): {"bits": 2}
        for i in range(pattern_count)
    }
    cfg = QuantizeConfig(dynamic=patterns, bits=4, group_size=128, sym=False)

    modules = []
    for i in range(module_count):
        layer = i % 1000
        proj = i % 3
        modules.append(f"model.layers.{layer}.mlp.proj.{proj}")
    # Make sure a subset actually matches so the test is realistic.
    for i in range(min(pattern_count, module_count)):
        modules[i] = f"model.layers.{i}.mlp.down_proj"

    with patch.object(pcre.Pattern, "match") as mock_match:
        for name in modules:
            cfg.dynamic_get(name, "bits", cfg.bits)
        assert mock_match.call_count == 0, (
            f"pcre.Pattern.match called {mock_match.call_count} times for a "
            f"fully exact dynamic config; expected zero calls."
        )


def test_dynamic_same_content_shares_fingerprint_cache_and_order_does_not():
    first = {
        "+:^model\\.layers\\.0\\.mlp\\.proj$": {"bits": 2, "nested": {"kind": "a"}},
        "+:^model\\.layers\\.\\d+\\.mlp\\.proj$": {"bits": 8},
    }
    same = {
        "+:^model\\.layers\\.0\\.mlp\\.proj$": {"bits": 2, "nested": {"kind": "a"}},
        "+:^model\\.layers\\.\\d+\\.mlp\\.proj$": {"bits": 8},
    }
    reordered = dict(reversed(list(first.items())))

    cfg = QuantizeConfig(dynamic=first, bits=4, group_size=128, sym=False)
    same_cfg = QuantizeConfig(dynamic=same, bits=4, group_size=128, sym=False)
    reordered_cfg = QuantizeConfig(dynamic=reordered, bits=4, group_size=128, sym=False)

    assert cfg.dynamic_get("model.layers.0.mlp.proj", "bits", cfg.bits) == 2
    assert same_cfg.dynamic_get("model.layers.0.mlp.proj", "bits", cfg.bits) == 2
    assert reordered_cfg.dynamic_get("model.layers.0.mlp.proj", "bits", cfg.bits) == 8
    assert len(_DYNAMIC_PATTERN_CACHE) == 2


def test_dynamic_results_are_defensive_copies_across_equal_configs():
    dynamic = {
        "+:^model\\.layers\\.0\\.mlp\\.proj$": {
            "bits": 2,
            "nested": {"tag": "original"},
        }
    }
    cfg_a = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)
    cfg_b = QuantizeConfig(dynamic={**dynamic}, bits=4, group_size=128, sym=False)
    module_name = "model.layers.0.mlp.proj"

    result_a = cfg_a.dynamic_get(module_name)
    result_a["bits"] = 8
    result_a["nested"]["tag"] = "changed"

    assert cfg_b.dynamic_get(module_name) == {
        "bits": 2,
        "nested": {"tag": "original"},
    }

    nested_b = cfg_b.dynamic_get(module_name, "nested")
    nested_b["tag"] = "changed-again"
    assert cfg_a.dynamic_get(module_name, "nested") == {"tag": "original"}


def test_dynamic_in_place_nested_mutation_invalidates_regex_snapshot():
    dynamic = {"+:^model\\.layers\\.\\d+\\.mlp\\.proj$": {"bits": 2, "meta": {"tag": "old"}}}
    cfg = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)
    module_name = "model.layers.0.mlp.proj"
    assert cfg.dynamic_get(module_name, "bits", cfg.bits) == 2

    cfg.dynamic[next(iter(cfg.dynamic))]["bits"] = 8
    assert cfg.dynamic_get(module_name, "bits", cfg.bits) == 8


def test_dynamic_config_pickle_round_trip_preserves_nested_mutation_tracking():
    pattern = r"+:^model\.layers\.\d+\.mlp\.proj$"
    cfg = QuantizeConfig(dynamic={pattern: {"bits": 2, "meta": [{"tag": "old"}]}})
    restored = pickle.loads(pickle.dumps(cfg))
    module_name = "model.layers.0.mlp.proj"

    assert restored.dynamic_get(module_name, "bits", restored.bits) == 2
    restored.dynamic[pattern]["meta"][0]["tag"] = "new"
    restored.dynamic[pattern]["bits"] = 8
    assert restored.dynamic_get(module_name) == {"bits": 8, "meta": [{"tag": "new"}]}
    assert cfg.dynamic_get(module_name) == {"bits": 2, "meta": [{"tag": "old"}]}


def test_standalone_tracked_list_nested_edits_touch_its_root():
    tracked = _TrackedList([{"tag": "old"}])
    restored = pickle.loads(pickle.dumps(tracked))

    tracked[0]["tag"] = "new"
    restored[0]["tag"] = "restored"
    assert tracked._mutation_version == 1
    assert restored._mutation_version == 1


def test_dynamic_caches_are_bounded():
    for index in range(_DYNAMIC_PATTERN_CACHE_MAXSIZE + 32):
        cfg = QuantizeConfig(
            dynamic={f"+:^module\\.{index}$": {"bits": 2}},
            bits=4,
            group_size=128,
            sym=False,
        )
        cfg.dynamic_get(f"module.{index}", "bits", cfg.bits)

    lookup_cfg = QuantizeConfig(
        dynamic={r"+:^module\.": {"bits": 2}},
        bits=4,
        group_size=128,
        sym=False,
    )
    for index in range(_DYNAMIC_OVERRIDE_CACHE_MAXSIZE + 32):
        assert lookup_cfg.dynamic_get(f"module.{index}", "bits", lookup_cfg.bits) == 2

    assert len(_DYNAMIC_PATTERN_CACHE) <= _DYNAMIC_PATTERN_CACHE_MAXSIZE
    assert len(_DYNAMIC_OVERRIDE_CACHE) <= _DYNAMIC_OVERRIDE_CACHE_MAXSIZE


def test_tracked_containers_preserve_content_comparison_semantics():
    tracked_dict = _TrackedDict({"key": [1, 2]})
    assert tracked_dict == {"key": [1, 2]}
    with pytest.raises(TypeError):
        hash(tracked_dict)

    tracked_list = _TrackedList([1, 2])
    assert tracked_list == [1, 2]
    assert tracked_list != [1, 3]


def test_dynamic_value_fingerprint_has_fixed_shape_and_type_metadata():
    class FirstEnum(Enum):
        VALUE = 1

    class SecondEnum(Enum):
        VALUE = 1

    class FirstObject:
        def __repr__(self):
            return "same-repr"

    class SecondObject:
        def __repr__(self):
            return "same-repr"

    values = [
        {"key": [1]},
        [1],
        (1,),
        {1},
        frozenset({1}),
        FirstEnum.VALUE,
        None,
        FirstObject(),
    ]
    fingerprints = [_dynamic_value_fingerprint(value) for value in values]

    assert all(len(fingerprint) == 4 for fingerprint in fingerprints)
    assert fingerprints[0][0] == "mapping"
    assert fingerprints[0][1][0][1][0] == "list"
    assert _dynamic_value_fingerprint({"first": 1, "second": 2}) != _dynamic_value_fingerprint(
        {"second": 2, "first": 1}
    )
    assert fingerprints[1] != fingerprints[2]
    assert fingerprints[5][0] == "enum"
    assert fingerprints[5][1:3] == (FirstEnum.__module__, FirstEnum.__qualname__)
    assert _dynamic_value_fingerprint(FirstEnum.VALUE) != _dynamic_value_fingerprint(SecondEnum.VALUE)
    assert fingerprints[7][1:] == (FirstObject.__module__, FirstObject.__qualname__, "same-repr")
    assert _dynamic_value_fingerprint(FirstObject()) != _dynamic_value_fingerprint(SecondObject())
