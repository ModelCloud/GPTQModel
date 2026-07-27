# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pcre
import pytest

from gptqmodel.quantization.config import (
    QuantizeConfig,
    _DYNAMIC_ALL_EXACT_CACHE,
    _DYNAMIC_EXACT_LOOKUP_CACHE,
    _DYNAMIC_OVERRIDE_CACHE,
    _DYNAMIC_PATTERN_CACHE,
    _DYNAMIC_REGEX_PATTERN_CACHE,
)


def _clear_dynamic_caches():
    """Drop the global dynamic pattern/override caches so tests are independent."""
    _DYNAMIC_PATTERN_CACHE.clear()
    _DYNAMIC_EXACT_LOOKUP_CACHE.clear()
    _DYNAMIC_REGEX_PATTERN_CACHE.clear()
    _DYNAMIC_OVERRIDE_CACHE.clear()
    _DYNAMIC_ALL_EXACT_CACHE.clear()


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
