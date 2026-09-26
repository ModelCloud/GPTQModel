# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed and adaptive dynamic-result caches over two model traversals.

Run: python scripts/benchmark_dynamic_override_cache.py --modules 50000 --rules 2000
"""

import argparse
import importlib
import time
from unittest.mock import patch

import pcre

from gptqmodel.quantization.config import QuantizeConfig


config = importlib.import_module("gptqmodel.quantization.config")


def run_case(label, capacity, cfg, names, calls_per_module):
    cache = config._BoundedLRUCache(capacity)
    config._DYNAMIC_PATTERN_CACHE.clear()
    config._DYNAMIC_EXACT_LOOKUP_CACHE.clear()
    config._DYNAMIC_REGEX_PATTERN_CACHE.clear()
    config._DYNAMIC_ALL_EXACT_CACHE.clear()
    original_match = pcre.Pattern.match
    match_calls = 0

    def counted_match(pattern, name):
        nonlocal match_calls
        match_calls += 1
        return original_match(pattern, name)

    with patch.object(config, "_DYNAMIC_OVERRIDE_CACHE", cache), patch.object(pcre.Pattern, "match", counted_match):
        rows = []
        for traversal in range(2):
            start = time.perf_counter()
            before = match_calls
            for name in names:
                for _ in range(calls_per_module):
                    cfg.dynamic_get(name, "bits", cfg.bits)
            rows.append((label, traversal + 1, time.perf_counter() - start,
                         match_calls - before, cache.hits, cache.misses,
                         cache.evictions, cache.peak_size))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=int, default=50_000)
    parser.add_argument("--rules", type=int, default=2_000)
    parser.add_argument("--calls-per-module", type=int, default=4)
    args = parser.parse_args()
    if args.modules < 1 or args.rules < 1 or args.calls_per_module < 1:
        parser.error("modules, rules and calls-per-module must be positive")

    names = [f"model.layers.{i // 512}.moe.experts.{i % 512}" for i in range(args.modules)]
    dynamic = {f"+:^unused\\.rule\\.{i}$": {"bits": 3} for i in range(args.rules - 1)}
    dynamic[r"+:^model\.layers\.\d+\.moe\.experts\.\d+$"] = {"bits": 2}
    cfg = QuantizeConfig(dynamic=dynamic)

    adaptive = 1 << (max(8192, (args.modules * 5 + 3) // 4) - 1).bit_length()
    print("case pass seconds pcre_calls hits misses evictions peak_size")
    for label, capacity in (("fixed", 8192), ("adaptive", adaptive), ("reference", 1 << 31)):
        for row in run_case(label, capacity, cfg, names, args.calls_per_module):
            print("%s %d %.3f %d %d %d %d %d" % row)


if __name__ == "__main__":
    main()
