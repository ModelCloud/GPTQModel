# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""PEFT ``trainable_token_indices`` adapters must fail closed (#4310).

PEFT saves the trained embedding-row deltas as ``*.trainable_tokens_delta``
tensors. This runtime only applies ``lora_A`` / ``lora_B``, so accepting such a
checkpoint silently drops the trained tokens and runs a different adapter.
Loading must raise, matching the DoRA/aLoRA fail-closed behavior from #4211.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.adapter.adapter import AdapterCache, Lora
from gptqmodel.adapter.peft import LoraConfig


ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def _write_adapter(directory, config, weights):
    (directory / ADAPTER_CONFIG).write_text(json.dumps(config))
    save_file(weights, str(directory / ADAPTER_WEIGHTS))


def _lora_weights():
    return {
        "base_model.model.m.lora_A.weight": torch.ones(2, 4, dtype=torch.float16),
        "base_model.model.m.lora_B.weight": torch.ones(3, 2, dtype=torch.float16),
    }


@pytest.fixture(autouse=True)
def _clear_adapter_cache():
    AdapterCache.reset()
    yield
    AdapterCache.reset()


def test_trainable_token_indices_config_is_rejected(tmp_path):
    _write_adapter(
        tmp_path,
        {"r": 2, "lora_alpha": 2, "trainable_token_indices": {"embed_tokens": [5, 6]}},
        _lora_weights(),
    )

    with pytest.raises(ValueError, match="trainable_token_indices"):
        LoraConfig.from_pretrained(path=str(tmp_path), filename=ADAPTER_CONFIG)


def test_trainable_tokens_delta_tensor_is_rejected(tmp_path):
    weights = _lora_weights()
    weights["base_model.model.embed_tokens.token_adapter.trainable_tokens_delta"] = torch.ones(
        2, 4, dtype=torch.float16
    )
    _write_adapter(tmp_path, {"r": 2, "lora_alpha": 2, "target_modules": ["m"]}, weights)

    adapter = Lora(rank=2, path=str(tmp_path))
    with pytest.raises(ValueError, match="trainable_token_indices"):
        adapter.post_init("m", torch.device("cpu"))


def test_plain_lora_still_loads(tmp_path):
    _write_adapter(
        tmp_path,
        {"r": 2, "lora_alpha": 4, "target_modules": ["m"], "bias": "none"},
        _lora_weights(),
    )

    adapter = Lora(rank=2, path=str(tmp_path))
    adapter.post_init("m", torch.device("cpu"))

    assert adapter.lora_A.shape == (4, 2)
    assert adapter.lora_B.shape == (2, 3)
    assert adapter.lora_scale == pytest.approx(2.0)

