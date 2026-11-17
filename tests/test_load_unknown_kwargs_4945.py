# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""`GPTQModel.load` must not silently drop keywords it cannot honour (#4945).

The unquantized path forwards its keywords to the HF model constructor and
fails loudly, but the quantized path used to pop the keys it knew and hand the
rest to ``QuantizeConfig.from_pretrained(**kwargs)``, which ignored them. A
misspelled ``dtpye``/``attn_implmentation`` or an inapplicable ``bits`` loaded
with defaults and reported success, and a passed ``quantize_config`` was
dropped in favour of the checkpoint's serialized config.

These tests pin the loud failure, its suggestion, and that every keyword the
quantized loader really consumes still works.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization import AWQConfig, GPTQConfig


_VOCAB = 128
_HIDDEN = 64
_INTERMEDIATE = 128
_LAYERS = 2
_GROUP_SIZE = 32


def _config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


@pytest.fixture(scope="module")
def dense_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("dense_4945") / "dense"
    torch.manual_seed(0)
    Qwen2ForCausalLM(_config()).to(torch.float16).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def quantized_checkpoint(dense_model_dir, tmp_path_factory):
    checkpoint = tmp_path_factory.mktemp("quantized_4945") / "gptq"
    generator = torch.Generator().manual_seed(1)
    calibration = [
        {"input_ids": torch.randint(0, _VOCAB, (1, 32), generator=generator)}
        for _ in range(4)
    ]
    model = GPTQModel.load(
        str(dense_model_dir),
        GPTQConfig(bits=4, group_size=_GROUP_SIZE, offload_to_disk=False),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(str(checkpoint))
    return checkpoint


def _load(checkpoint, **kwargs):
    return GPTQModel.load(str(checkpoint), backend=BACKEND.GPTQ_TORCH, device="cpu", **kwargs)


@pytest.mark.parametrize(
    ("keyword", "value", "suggestion"),
    [
        ("dtpye", torch.float16, "dtype"),
        ("attn_implmentation", "eager", "attn_implementation"),
        ("devcie_map", {"": "cpu"}, "device_map"),
        # No close match: the message must still name the offending key.
        ("bits", 8, None),
    ],
)
def test_unknown_quantized_load_keyword_raises_with_a_hint(
    quantized_checkpoint, keyword, value, suggestion
):
    with pytest.raises(TypeError) as excinfo:
        _load(quantized_checkpoint, **{keyword: value})

    message = str(excinfo.value)
    assert f"`{keyword}`" in message
    assert "unexpected keyword argument" in message
    if suggestion is not None:
        assert f"did you mean `{suggestion}`" in message


def test_passing_a_quantize_config_to_a_quantized_checkpoint_raises(quantized_checkpoint):
    """The serialized config describes the stored tensors; a new one cannot apply."""

    with pytest.raises(TypeError) as excinfo:
        GPTQModel.load(
            str(quantized_checkpoint),
            AWQConfig(bits=8, group_size=32),
            backend=BACKEND.GPTQ_TORCH,
            device="cpu",
        )

    message = str(excinfo.value)
    assert "`quantize_config`" in message
    assert "already-quantized checkpoint" in message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": torch.float16},
        {"torch_dtype": torch.float16},
        {"attn_implementation": "eager"},
        {"revision": None},
        {"subfolder": ""},
        {"cache_dir": None},
        {"local_files_only": True},
        {"use_auth_token": None},
        {"trust_remote_code": False},
        {"offload_state_dict": False},
        {"offload_buffers": False},
        {"fuse": False},
    ],
)
def test_consumed_keywords_still_load(quantized_checkpoint, kwargs):
    """Every keyword the quantized path reads must keep working."""

    model = _load(quantized_checkpoint, **kwargs)
    assert model.quantize_config.bits == 4


def test_dtype_and_attention_override_are_honoured(quantized_checkpoint):
    """The two most damaging silent drops must actually change the load."""

    model = _load(quantized_checkpoint, dtype=torch.float16, attn_implementation="eager")

    assert model.model.dtype == torch.float16
    assert model.model.config._attn_implementation == "eager"


def test_device_map_is_placed_or_rejected(quantized_checkpoint):
    """A placement the load cannot honour raises instead of being dropped."""

    placed = _load(quantized_checkpoint, device_map={"": "cpu"})
    assert next(placed.model.parameters()).device.type == "cpu"

    with pytest.raises(ValueError, match="conflicts with"):
        GPTQModel.load(
            str(quantized_checkpoint),
            backend=BACKEND.GPTQ_TORCH,
            device="cpu",
            device_map={"": "cuda:0"},
        )


def test_unquantized_path_still_rejects_the_typo(dense_model_dir):
    """The dense path already failed loudly; keep the two paths consistent."""

    with pytest.raises(TypeError, match="dtpye"):
        GPTQModel.load(str(dense_model_dir), GPTQConfig(bits=4), dtpye=torch.float16)
