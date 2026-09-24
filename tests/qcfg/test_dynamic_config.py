# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pickle
from enum import Enum
from unittest.mock import patch

import pcre
import pytest

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
    _clear_dynamic_caches()
    yield
    _clear_dynamic_caches()


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
