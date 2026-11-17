# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``GPTQModel.load()`` must reject unknown keywords on the quantized path.

``from_quantized`` consumed its known options and let every other keyword reach
``QuantizeConfig.from_pretrained``, which ignored it. A typo of a real option
(``dtpye=``, ``attn_implmentation=``) or an option that does not apply
(``bits=``) therefore "succeeded" and changed nothing, while the unquantized
path raises from the HF constructor. A passed ``quantize_config`` was dropped
the same way, so the checkpoint's own config won without a word (#4945).

The checkpoint is quantized once from a tiny cached model and reused.
"""

import json
import shutil

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models import loader as loader_module
from gptqmodel.quantization import FORMAT, AWQConfig, GPTQConfig


_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
_MODEL_FILES = ["*.json", "*.safetensors", "tokenizer*", "*.model"]

_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose for the tokenizer",
    "gravity attracts objects with mass and measurements help test predictions",
    "the baker mixed flour with water and yeast before leaving the dough to rise",
    "a function transforms inputs into results and tests compare expected values",
    "water evaporates from lakes, clouds form when vapor cools, and rain returns it",
    "a young child watched a small bird spread its wings and fly back to its nest",
]


@pytest.fixture(scope="module")
def source_model_path():
    """The cached checkpoint as a local path; the Hub is offline in CI."""

    return snapshot_download(
        _MODEL_ID,
        local_files_only=True,
        allow_patterns=_MODEL_FILES,
    )


@pytest.fixture(scope="module")
def quantized_checkpoint(tmp_path_factory, source_model_path):
    destination = tmp_path_factory.mktemp("load-strict-4945") / "gptq-4bit-g128"
    tokenizer = AutoTokenizer.from_pretrained(source_model_path)
    calibration = [
        dict(tokenizer(text, return_tensors="pt")) for text in _CALIBRATION_TEXTS
    ]

    model = GPTQModel.load(
        source_model_path,
        quantize_config=GPTQConfig(
            bits=4,
            group_size=128,
            device="cpu",
            offload_to_disk=False,
        ),
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(str(destination))
    del model
    return str(destination)


@pytest.fixture(scope="module")
def checkpoint_without_format(quantized_checkpoint, tmp_path_factory):
    """A checkpoint that does not pin a format, so ``format=`` may override it."""

    destination = tmp_path_factory.mktemp("load-strict-4945-format") / "checkpoint"
    shutil.copytree(quantized_checkpoint, destination)
    for name in ("quantize_config.json", "config.json"):
        path = destination / name
        payload = json.loads(path.read_text())
        target = (
            payload
            if name == "quantize_config.json"
            else payload.get("quantization_config", {})
        )
        target.pop("format", None)
        target.pop("checkpoint_format", None)
        path.write_text(json.dumps(payload))
    return str(destination)


def test_quantized_load_rejects_dtype_typo_with_a_suggestion(quantized_checkpoint):
    with pytest.raises(TypeError, match="unexpected keyword argument") as excinfo:
        GPTQModel.load(quantized_checkpoint, device="cpu", dtpye=torch.float16)

    message = str(excinfo.value)
    assert "`dtpye`" in message
    assert "did you mean `dtype`" in message


def test_quantized_load_rejects_attn_implementation_typo_with_a_suggestion(
    quantized_checkpoint,
):
    with pytest.raises(TypeError, match="unexpected keyword argument") as excinfo:
        GPTQModel.load(quantized_checkpoint, device="cpu", attn_implmentation="eager")

    message = str(excinfo.value)
    assert "`attn_implmentation`" in message
    assert "did you mean `attn_implementation`" in message


def test_quantized_load_rejects_an_option_that_does_not_apply(quantized_checkpoint):
    """``bits=`` has no close match, but must still raise instead of no-op."""

    with pytest.raises(TypeError, match="unexpected keyword argument") as excinfo:
        GPTQModel.load(quantized_checkpoint, device="cpu", bits=8)

    assert "`bits`" in str(excinfo.value)


def test_quantized_load_rejects_a_passed_quantize_config(quantized_checkpoint):
    """The checkpoint's own config is used; a different one must not be ignored."""

    with pytest.raises(ValueError, match="quantize_config.*quantized checkpoint") as excinfo:
        GPTQModel.load(
            quantized_checkpoint,
            AWQConfig(bits=8, group_size=32),
            device="cpu",
        )

    message = str(excinfo.value)
    assert "checkpoint" in message
    assert "unquantized model" in message


def test_quantized_load_accepts_its_documented_keywords(checkpoint_without_format):
    model = GPTQModel.load(
        checkpoint_without_format,
        device="cpu",
        dtype=torch.float16,
        attn_implementation="eager",
        backend=BACKEND.TORCH,
        format=FORMAT.GPTQ,
    )

    assert model.model.dtype == torch.float16
    assert model.quantize_config.bits == 4
    assert model.quantize_config.group_size == 128


def test_unquantized_load_still_raises_for_a_dtype_typo(source_model_path):
    with pytest.raises(TypeError, match="dtpye"):
        GPTQModel.load(
            source_model_path,
            GPTQConfig(bits=4, group_size=128),
            device="cpu",
            dtpye=torch.float16,
        )


def test_new_load_options_are_accepted_from_the_signature_and_pops():
    """The accepted set follows the code, so a new option is not rejected."""

    class _FutureLoader:
        @classmethod
        def from_quantized(cls, model_id_or_path, brand_new_option=None, **kwargs):
            return None

    accepted = loader_module._accepted_quantized_load_kwargs(
        _FutureLoader, {"newly_popped_option"}
    )

    assert "brand_new_option" in accepted
    assert "newly_popped_option" in accepted
