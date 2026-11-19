# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Unsupported PEFT adapter features must fail closed, not run as plain LoRA.

DoRA, trained biases, ``modules_to_save`` and the newer PEFT feature keys all
add tensors or change the adapter math. This runtime only applies ``lora_A`` /
``lora_B``, so loading them silently would produce a model that runs but is not
the checkpoint that was requested.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.adapter import peft
from gptqmodel.adapter.adapter import AdapterCache, Lora
from gptqmodel.adapter.peft import LoraConfig


ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def _write_adapter(directory, config, weights=None):
    (directory / ADAPTER_CONFIG).write_text(json.dumps(config))
    if weights is None:
        weights = {
            "base_model.model.m.lora_A.weight": torch.ones(2, 4, dtype=torch.float16),
            "base_model.model.m.lora_B.weight": torch.ones(3, 2, dtype=torch.float16),
        }
    save_file(weights, str(directory / ADAPTER_WEIGHTS))


@pytest.fixture(autouse=True)
def _clear_adapter_cache():
    AdapterCache.reset()
    yield
    AdapterCache.reset()


@pytest.mark.parametrize("megatron_config", [None, {"tensor_model_parallel_size": 1}])
def test_use_dora_is_rejected_with_and_without_megatron(megatron_config):
    with pytest.raises(ValueError, match="DoRA is not supported"):
        LoraConfig(use_dora=True, megatron_config=megatron_config)


def test_dora_adapter_config_is_rejected_before_weights_are_applied(tmp_path):
    _write_adapter(tmp_path, {"r": 2, "lora_alpha": 2, "use_dora": True})

    adapter = Lora(rank=2, path=str(tmp_path))
    with pytest.raises(ValueError, match="DoRA is not supported"):
        adapter.post_init("m", torch.device("cpu"))


def test_magnitude_vector_tensor_is_rejected_without_use_dora(tmp_path):
    """A stray magnitude vector must not be dropped even if the flag is absent."""

    weights = {
        "base_model.model.m.lora_A.weight": torch.ones(2, 4, dtype=torch.float16),
        "base_model.model.m.lora_B.weight": torch.ones(3, 2, dtype=torch.float16),
        "base_model.model.m.lora_magnitude_vector": torch.ones(3, dtype=torch.float16),
    }
    _write_adapter(tmp_path, {"r": 2, "lora_alpha": 2, "target_modules": ["m"]}, weights)

    adapter = Lora(rank=2, path=str(tmp_path))
    with pytest.raises(ValueError, match="DoRA is not supported"):
        adapter.post_init("m", torch.device("cpu"))


@pytest.mark.parametrize("bias", ["all", "lora_only"])
def test_non_none_bias_config_is_rejected(bias):
    with pytest.raises(ValueError, match="`bias`"):
        LoraConfig(bias=bias)


def test_bias_tensor_is_rejected_when_config_does_not_declare_it(tmp_path):
    weights = {
        "base_model.model.m.lora_A.weight": torch.ones(2, 4, dtype=torch.float16),
        "base_model.model.m.lora_B.weight": torch.ones(3, 2, dtype=torch.float16),
        "base_model.model.m.bias": torch.ones(3, dtype=torch.float16),
    }
    _write_adapter(tmp_path, {"r": 2, "lora_alpha": 2, "target_modules": ["m"]}, weights)

    adapter = Lora(rank=2, path=str(tmp_path))
    with pytest.raises(ValueError, match="bias"):
        adapter.post_init("m", torch.device("cpu"))


def test_modules_to_save_config_is_rejected():
    with pytest.raises(ValueError, match="modules_to_save"):
        LoraConfig(modules_to_save=["classifier"])


def test_modules_to_save_tensor_is_rejected(tmp_path):
    weights = {
        "base_model.model.m.lora_A.weight": torch.ones(2, 4, dtype=torch.float16),
        "base_model.model.m.lora_B.weight": torch.ones(3, 2, dtype=torch.float16),
        "base_model.model.classifier.modules_to_save.default.weight": torch.ones(
            2, 2, dtype=torch.float16
        ),
    }
    _write_adapter(tmp_path, {"r": 2, "lora_alpha": 2, "target_modules": ["m"]}, weights)

    adapter = Lora(rank=2, path=str(tmp_path))
    with pytest.raises(ValueError, match="modules_to_save"):
        adapter.post_init("m", torch.device("cpu"))


class _WarnRecorder:
    def __init__(self):
        self.messages = []

    def __call__(self, message, *args, **kwargs):
        self.messages.append(message)

    def once(self, message, *args, **kwargs):
        self.messages.append(message)


def test_unknown_config_key_warns_and_still_loads(tmp_path, monkeypatch):
    warn = _WarnRecorder()
    monkeypatch.setattr(
        peft, "log", SimpleNamespace(warn=warn, info=lambda *args, **kwargs: None)
    )

    _write_adapter(
        tmp_path,
        {
            "r": 2,
            "lora_alpha": 2,
            "peft_version": "0.17.0",
            "inference_mode": True,
        },
    )

    config = LoraConfig.from_pretrained(path=str(tmp_path), filename=ADAPTER_CONFIG)

    assert config.r == 2
    assert len(warn.messages) == 1
    assert "peft_version" in warn.messages[0]
    assert "inference_mode" in warn.messages[0]


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("use_qalora", True, "QALoRA"),
        ("alora_invocation_tokens", [1, 2], "aLoRA"),
        ("target_parameters", ["embed_tokens"], "target_parameters"),
        ("corda_config", {"x": 1}, "CORDA"),
    ],
)
def test_unsupported_peft_feature_keys_are_rejected(tmp_path, key, value, match):
    _write_adapter(tmp_path, {"r": 2, "lora_alpha": 2, key: value})

    with pytest.raises(ValueError, match=match):
        LoraConfig.from_pretrained(path=str(tmp_path), filename=ADAPTER_CONFIG)


def test_plain_lora_still_loads(tmp_path):
    _write_adapter(
        tmp_path,
        {
            "r": 2,
            "lora_alpha": 4,
            "target_modules": ["m"],
            "bias": "none",
            "use_dora": False,
        },
    )

    adapter = Lora(rank=2, path=str(tmp_path))
    adapter.post_init("m", torch.device("cpu"))

    assert adapter.lora_A.shape == (4, 2)
    assert adapter.lora_B.shape == (2, 3)
    assert adapter.lora_scale == pytest.approx(2.0)
    torch.testing.assert_close(
        adapter.lora_B, torch.full((2, 3), 2.0, dtype=torch.float16)
    )


@pytest.mark.parametrize("peft_type", ["IA3", "LOHA", "LOKR", "OFT", "ADALORA"])
def test_explicitly_unsupported_peft_type_is_rejected(tmp_path, peft_type):
    """A declared non-LoRA PEFT type must not be rewritten to LORA (#5425)."""

    _write_adapter(
        tmp_path,
        {"r": 2, "lora_alpha": 2, "target_modules": ["m"], "peft_type": peft_type},
    )

    with pytest.raises(ValueError, match="peft_type"):
        LoraConfig.from_pretrained(path=str(tmp_path), filename=ADAPTER_CONFIG)

    # The public load path reports the same error instead of attaching nothing.
    adapter = Lora(rank=2, path=str(tmp_path))
    with pytest.raises(ValueError, match="peft_type"):
        adapter.post_init("m", torch.device("cpu"))


@pytest.mark.parametrize(
    ("tensor", "feature"),
    [
        ("base_model.model.m.ia3_l", "IA3"),
        ("base_model.model.m.hada_w1_a", "LoHa"),
        ("base_model.model.m.lokr_w1", "LoKr"),
        ("base_model.model.m.oft_blocks", "OFT"),
        ("base_model.model.m.vera_lambda_d", "VeRA"),
        ("base_model.model.m.lora_E", "AdaLoRA"),
    ],
)
def test_non_lora_adapter_tensor_is_rejected_without_a_declared_type(tmp_path, tensor, feature):
    """A legacy config without `peft_type` cannot smuggle another layout in."""

    _write_adapter(
        tmp_path,
        {"r": 2, "lora_alpha": 2, "target_modules": ["m"]},
        {tensor: torch.ones(3, 4, dtype=torch.float16)},
    )

    adapter = Lora(rank=2, path=str(tmp_path))
    with pytest.raises(ValueError, match=feature):
        adapter.post_init("m", torch.device("cpu"))


def test_legacy_config_without_peft_type_still_loads(tmp_path):
    _write_adapter(tmp_path, {"r": 2, "lora_alpha": 2, "target_modules": ["m"]})

    adapter = Lora(rank=2, path=str(tmp_path))
    adapter.post_init("m", torch.device("cpu"))

    assert adapter.lora_A.shape == (4, 2)
    assert adapter.lora_B.shape == (2, 3)
