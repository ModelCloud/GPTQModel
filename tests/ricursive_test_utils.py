# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Isolate package-owned Ripple classes when a test deliberately patches a class."""

import importlib.util
import itertools
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def package_source_dir():
    return Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/ricursive"


COUNTER = itertools.count()


def make_model(**overrides):
    # Only inspect/import installed package files. Unique namespaces prevent a
    # deliberate legacy-method mutation from leaking into the next test.
    root = package_source_dir()
    name = f"ripple_audit_{next(COUNTER)}"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".modeling_ricursive", root / "modeling_ricursive.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "sliding_window": 4,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention", "full_attention"],
        "use_valres": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": None,
    }
    config.update(overrides)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = module.RicursiveForCausalLM(module.RicursiveConfig(**config)).eval()
    return model
