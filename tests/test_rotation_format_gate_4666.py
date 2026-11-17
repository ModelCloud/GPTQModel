# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Model-wide ``rotation`` is only valid on the GPTQ checkpoint layouts (#4666).

``rotation`` is a base-class field, so every config accepted it. Only the GPTQ
kernels apply the online Hadamard a rotated checkpoint needs, and the loader
only accepts a rotated checkpoint whose format is GPTQ/GPTQ_V2. A non-GPTQ
config therefore ran the whole quantization on a rotated model, scored worse
(FP8 nll 21.05 -> 30.20, AWQ 21.59 -> 30.68), and saved a checkpoint that could
not be reloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND
from gptqmodel.models import base as base_module
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import (
    AWQConfig,
    BitsAndBytesConfig,
    EXL3Config,
    FP8Config,
    GGUFConfig,
    GPTQConfig,
    MacaqConfig,
    NF12Config,
    ParoConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    VPTQConfig,
)


CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]

# Every config whose kernels ignore the online Hadamard.
_REJECTED = [
    ("awq", lambda rotation: AWQConfig(bits=4, group_size=128, rotation=rotation)),
    ("fp8", lambda rotation: FP8Config(rotation=rotation)),
    ("qqq", lambda rotation: QQQConfig(bits=4, group_size=128, rotation=rotation)),
    ("vptq", lambda rotation: VPTQConfig(rotation=rotation)),
    ("paro", lambda rotation: ParoConfig(rotation=rotation)),
    ("exl3", lambda rotation: EXL3Config(rotation=rotation)),
    ("bitsandbytes", lambda rotation: BitsAndBytesConfig(rotation=rotation)),
    ("gguf", lambda rotation: GGUFConfig(rotation=rotation)),
    ("nf12", lambda rotation: NF12Config(rotation=rotation)),
    ("rtn-gemm", lambda rotation: RTNConfig(bits=4, format="gemm", rotation=rotation)),
]


@pytest.mark.parametrize("name, factory", _REJECTED, ids=[case[0] for case in _REJECTED])
@pytest.mark.parametrize("rotation", ["hadamard", "random"])
def test_non_gptq_configs_reject_model_wide_rotation(name, factory, rotation):
    with pytest.raises(ValueError, match="only supported for the GPTQ") as error:
        factory(rotation)

    message = str(error.value)
    # The message names both layouts that do work, so the remedy is explicit.
    assert "FORMAT.GPTQ" in message and "FORMAT.GPTQ_V2" in message
    assert f"rotation={rotation!r}" in message


@pytest.mark.parametrize("name, factory", _REJECTED, ids=[case[0] for case in _REJECTED])
def test_rotation_none_is_accepted_everywhere(name, factory):
    assert factory(None).rotation is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda rotation: GPTQConfig(bits=4, group_size=128, rotation=rotation),
        lambda rotation: GPTQConfig(
            bits=4, group_size=128, format=FORMAT.GPTQ_V2, rotation=rotation
        ),
        # RTN saves the GPTQ layout and loads through the GPTQ kernels.
        lambda rotation: RTNConfig(bits=4, rotation=rotation),
        lambda rotation: QuantizeConfig(bits=4, group_size=128, rotation=rotation),
    ],
    ids=["gptq", "gptq-v2", "rtn-gptq-layout", "quantize-config"],
)
def test_gptq_layouts_still_accept_rotation(factory):
    assert factory("hadamard").rotation == "hadamard"


def test_macaq_keeps_its_per_layer_rotation():
    """MACaQ reuses ``rotation`` for its own per-layer transform."""

    config = MacaqConfig(rotation="hadamard", offload_to_disk=False)
    assert config.rotation == "hadamard"
    # `model_rotation` is None, so the model-wide gate does not apply.
    assert config.model_rotation is None


def _build_qmodel(tmp_path: Path, quantize_config) -> LlamaQModel:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(source)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    return LlamaQModel(
        model=model,
        quantized=False,
        tokenizer=tokenizer,
        quantize_config=quantize_config,
        model_local_path=str(source),
    )


def test_quantize_refuses_before_rotating_or_fusing(tmp_path, monkeypatch):
    """A refused run must leave the model exactly as it was loaded.

    The old order fused norms and rotated the weights first and only then
    rejected the backend, so the in-session model was unusable.
    """

    config = AWQConfig(bits=4, group_size=32, desc_act=False, device="cpu")
    # Simulate a config that reached the loader without construction-time
    # validation (a deserialized / legacy payload).
    config.rotation = "hadamard"

    qmodel = _build_qmodel(tmp_path, config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in qmodel.model.named_parameters()
    }

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("rotation ran before the format gate")

    monkeypatch.setattr(base_module, "fuse_layer_norms", _must_not_run)
    monkeypatch.setattr(base_module, "rotate_model", _must_not_run)

    with pytest.raises(ValueError, match="only supported for the GPTQ"):
        qmodel.quantize(
            CALIBRATION,
            batch_size=1,
            backend=BACKEND.AWQ_TORCH,
            calibration_data_min_length=1,
        )

    for name, parameter in qmodel.model.named_parameters():
        torch.testing.assert_close(parameter, before[name], msg=name)
