# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint gate has to name the architectures it actually supports."""

from types import SimpleNamespace

import pytest

from gptqmodel.looper.gptq_checkpoint import (
    CHECKPOINT_MODEL_TYPES,
    validate_checkpoint_support,
)


def _model(model_type: str):
    """Minimal stand-in: the gate only reads quantize_config and model_type."""

    quantize_config = SimpleNamespace(
        method="gptq",
        true_sequential=True,
        gptaq=None,
        foem=None,
        lm_head=False,
        dynamic=False,
        rotation=None,
        adapter=None,
        offload_to_disk=True,
    )
    return SimpleNamespace(
        quantize_config=quantize_config,
        model=SimpleNamespace(config=SimpleNamespace(model_type=model_type)),
    )


@pytest.mark.parametrize("model_type", sorted(CHECKPOINT_MODEL_TYPES))
def test_supported_architectures_pass_the_gate(model_type):
    validate_checkpoint_support(_model(model_type))


def test_qwen2_is_supported():
    """Gate only: real tied/untied recovery lives in test_checkpoint_quantization.py."""
    assert "qwen2" in CHECKPOINT_MODEL_TYPES
    validate_checkpoint_support(_model("qwen2"))


def test_unknown_architecture_names_the_supported_set():
    with pytest.raises(NotImplementedError) as excinfo:
        validate_checkpoint_support(_model("gpt2"))
    message = str(excinfo.value)
    assert "gpt2" in message
    for model_type in CHECKPOINT_MODEL_TYPES:
        assert model_type in message


def test_sequential_requirement_still_gates():
    model = _model("qwen2")
    model.quantize_config.true_sequential = False
    with pytest.raises(NotImplementedError, match="sequential"):
        validate_checkpoint_support(model)


def test_offload_requirement_still_gates():
    model = _model("qwen2")
    model.quantize_config.offload_to_disk = False
    with pytest.raises(ValueError, match="offload_to_disk"):
        validate_checkpoint_support(model)
