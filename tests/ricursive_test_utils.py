# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Load isolated tiny Ripple models for native and compatibility regression tests."""

import importlib.util
import itertools
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download


# Pin to the native source reviewed alongside these tests; no model weights.
RIPPLE_REVISION = "2f007aabfcedd0f8763d86d08f9aa0afbc344cea"


@lru_cache(maxsize=1)
def reference_dir():
    configured = os.environ.get("GPTQMODEL_RIPPLE_REF")
    if configured:
        root = Path(configured)
        if not (root / "modeling_ricursive.py").is_file():
            raise FileNotFoundError(f"Missing Ripple custom code in {root}")
        return root
    try:
        return Path(
            snapshot_download(
                "ricursiveAI/Ripple-65B-Preview",
                revision=RIPPLE_REVISION,
                allow_patterns=["configuration_ricursive.py", "modeling_ricursive.py"],
            )
        )
    except Exception as exc:
        pytest.skip(f"Ripple custom code unavailable: {exc}")


COUNTER = itertools.count()


@lru_cache(maxsize=1)
def compatibility_module():
    """Load the standalone shim without importing unrelated Transformers-5 models.

    This lets the same committed regression tests exercise the shim on 4.57,
    even though the complete GPT-QModel model registry requires Transformers 5.
    """
    path = Path(__file__).resolve().parents[1] / "gptqmodel/utils/ricursive_patch.py"
    spec = importlib.util.spec_from_file_location("ricursive_compat_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(*, source_dir=None, **overrides):
    root = Path(source_dir) if source_dir is not None else reference_dir()
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
