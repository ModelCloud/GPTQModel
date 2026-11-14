# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The global adapter cache must survive partial consumption.

`AdapterCache` is process-global by adapter path. `Lora.post_init` used to pop
the A/B keys it consumed straight out of the cached mapping and only dropped the
entry once *every* key was gone. A model load that instantiates a subset of the
adapter's target modules therefore left a partial dictionary behind, and the
next load of the same adapter could not find the keys the first load had already
taken.

The cached tensors are now immutable; consumption is recorded in a separate set
and the entry is released once every cached weight has been consumed.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.adapter.adapter import AdapterCache, Lora


CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _clear_adapter_cache():
    AdapterCache.reset()
    yield
    AdapterCache.reset()


def _write_adapter(directory, module_keys, *, rank=2):
    """Write a PEFT adapter whose weights cover ``module_keys``."""
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": list(module_keys)}),
        encoding="utf-8",
    )
    weights = {}
    for module_key in module_keys:
        weights[f"base_model.model.{module_key}.lora_A.weight"] = torch.arange(
            rank * 4, dtype=torch.float16
        ).reshape(rank, 4)
        weights[f"base_model.model.{module_key}.lora_B.weight"] = torch.arange(
            3 * rank, dtype=torch.float16
        ).reshape(3, rank)
    save_file(weights, str(directory / "adapter_model.safetensors"))


def _cache_key(directory):
    return (str(directory), None, None)


def test_second_sequential_load_sees_the_modules_the_first_load_ignored(tmp_path):
    """The reported failure: an unused target module poisoned the next load."""
    _write_adapter(tmp_path, ("layer0", "layer1"))

    first = Lora(rank=None, path=str(tmp_path))
    first.post_init("layer0", CPU)

    # `layer1` is never instantiated by this model, so it stays cached.
    second = Lora(rank=None, path=str(tmp_path))
    second.post_init("layer0", CPU)

    assert second.lora_A.shape == (4, 2)
    assert second.lora_B.shape == (2, 3)


def test_cached_source_weights_are_not_mutated_by_a_load(tmp_path):
    _write_adapter(tmp_path, ("layer0", "layer1"))

    Lora(rank=None, path=str(tmp_path)).post_init("layer0", CPU)

    entry = AdapterCache.cache[_cache_key(tmp_path)]
    assert sorted(entry["weights"]) == [
        "base_model.model.layer0.lora_A.weight",
        "base_model.model.layer0.lora_B.weight",
        "base_model.model.layer1.lora_A.weight",
        "base_model.model.layer1.lora_B.weight",
    ]


def test_a_later_load_can_consume_the_module_an_earlier_load_ignored(tmp_path):
    _write_adapter(tmp_path, ("layer0", "layer1"))

    Lora(rank=None, path=str(tmp_path)).post_init("layer0", CPU)
    second = Lora(rank=None, path=str(tmp_path))
    second.post_init("layer1", CPU)

    assert second.lora_A.shape == (4, 2)


def test_interleaved_loads_keep_every_module_addressable(tmp_path):
    """Two adapters for the same path, alternating modules, must both work."""
    _write_adapter(tmp_path, ("layer0", "layer1"))

    first = Lora(rank=None, path=tmp_path)
    second = Lora(rank=None, path=tmp_path)

    first.post_init("layer0", CPU)
    second.post_init("layer1", CPU)

    first_again = Lora(rank=None, path=tmp_path)
    first_again.post_init("layer0", CPU)

    assert first.lora_A.shape == (4, 2)
    assert second.lora_A.shape == (4, 2)
    assert first_again.lora_A.shape == (4, 2)


def test_fully_consumed_adapter_releases_the_cache_entry(tmp_path):
    _write_adapter(tmp_path, ("layer0", "layer1"))
    key = _cache_key(tmp_path)

    Lora(rank=None, path=tmp_path).post_init("layer0", CPU)
    assert key in AdapterCache.cache

    Lora(rank=None, path=tmp_path).post_init("layer1", CPU)
    assert key not in AdapterCache.cache
