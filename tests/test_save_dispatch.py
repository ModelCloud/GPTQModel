# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``save()`` and ``save_pretrained()`` must dispatch on the same model state.

``save_pretrained()`` used to redirect unconditionally to ``save_quantized()``:
a dense wrapper raised ``ValueError: Save aborted as model is not quantized``,
and a loaded checkpoint with requantized module replacements wrote the dense
``model.embed_tokens.weight`` instead of the packed
``qweight``/``qzeros``/``scales`` payload. ``save()`` already selected the
writer by state; the two public names must agree.
"""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import CalibrationConfig, GPTQModel, QuantizeEmbed
from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.quantization import QuantizeConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


def _calibration_texts(count: int = 8):
    return [
        " ".join(["the quick brown fox"] * (1 + index % 3))
        for index in range(count)
    ]


def _saved_tensor_keys(directory: str | Path) -> set[str]:
    keys = set()
    for path in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def _saved_tensors(directory: str | Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


@pytest.fixture
def tiny_dense(tmp_path: Path) -> Path:
    destination = tmp_path / "dense"
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(destination)
    build_wordlevel_tokenizer().save_pretrained(destination)
    return destination


def _stub(*, quantized: bool, replacement_only: bool = False) -> BaseQModel:
    model = object.__new__(BaseQModel)
    torch.nn.Module.__init__(model)
    model.quantized = quantized
    model._model_free_weight_only_replacements_only = replacement_only
    return model


def test_save_pretrained_dense_delegates_to_transformers_saver(tmp_path):
    """A dense wrapper has no quantized writer; the HF model owns the save."""

    model = _stub(quantized=False)
    calls = {}

    class _NativeModel:
        def save_pretrained(self, save_dir, **kwargs):
            calls["save_dir"] = save_dir
            calls["kwargs"] = kwargs
            return "native-result"

    model.model = _NativeModel()
    model.save_quantized = lambda **_kwargs: pytest.fail(
        "a dense model must not be routed to save_quantized"
    )

    result = model.save_pretrained(str(tmp_path / "dense"), max_shard_size="1MB")

    assert result == "native-result"
    assert calls == {
        "save_dir": str(tmp_path / "dense"),
        "kwargs": {"max_shard_size": "1MB"},
    }


def test_save_pretrained_replacement_uses_module_replacement_writer(tmp_path):
    """A weight-only replacement must keep its packed endpoint payload."""

    model = _stub(quantized=True, replacement_only=True)
    captured = {}
    model.save_quantized_embeddings = lambda **kwargs: captured.update(kwargs) or "replacement-result"
    model.save_quantized = lambda **_kwargs: pytest.fail(
        "a module-replacement checkpoint must not be routed to save_quantized"
    )

    result = model.save_pretrained(
        str(tmp_path / "replacement"), meta_quantizer="test-quantizer:1.0"
    )

    assert result == "replacement-result"
    assert captured == {
        "save_dir": str(tmp_path / "replacement"),
        "meta_quantizer": "test-quantizer:1.0",
    }


def test_save_pretrained_ordinary_quantized_redirects_to_save_quantized(tmp_path):
    """The existing quantized redirect is unchanged."""

    model = _stub(quantized=True)
    captured = {}
    model.save_quantized = lambda **kwargs: captured.update(kwargs) or "quantized-result"
    model.save_quantized_embeddings = lambda **_kwargs: pytest.fail(
        "an ordinary quantized model must not use the module-replacement writer"
    )

    model.save_pretrained(str(tmp_path / "quantized"), max_shard_size="2MB")

    assert captured == {
        "save_dir": str(tmp_path / "quantized"),
        "max_shard_size": "2MB",
    }


def test_dense_save_and_save_pretrained_match_the_native_transformers_saver(
    tiny_dense: Path, tmp_path: Path
):
    """Both public names write the same dense layout as the wrapped model."""

    wrapper = GPTQModel.load(str(tiny_dense), device="cpu", device_map={"": "cpu"})
    assert wrapper.quantized is False

    native_dir = tmp_path / "native"
    wrapper.model.save_pretrained(str(native_dir))
    native_keys = _saved_tensor_keys(native_dir)
    native_state = LlamaForCausalLM.from_pretrained(str(native_dir)).state_dict()

    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        expected = wrapper.model(input_ids=ids).logits

    for name, save in (("save", wrapper.save), ("save_pretrained", wrapper.save_pretrained)):
        output = tmp_path / name
        save(str(output))

        assert _saved_tensor_keys(output) == native_keys
        reloaded = LlamaForCausalLM.from_pretrained(str(output))
        for key, tensor in native_state.items():
            torch.testing.assert_close(reloaded.state_dict()[key], tensor, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(input_ids=ids).logits, expected, rtol=0, atol=0)


def test_requantized_input_embedding_keeps_packed_layout_through_both_save_names(
    tiny_dense: Path, tmp_path: Path
):
    """A requantized embedding must not fall back to the dense tensor."""

    quantized_dir = tmp_path / "quantized"
    quantizer = GPTQModel.load(
        str(tiny_dense), QuantizeConfig(bits=4, group_size=32), device="cpu"
    )
    quantizer.quantize(CalibrationConfig(dataset=_calibration_texts(), batch_size=2))
    quantizer.save(str(quantized_dir))
    del quantizer

    model = GPTQModel.load(str(quantized_dir), device="cpu")
    dynamic = dict(model.quantize_config.dynamic or {})
    dynamic[model.get_input_embeddings_name()] = {
        "bits": 4,
        "group_size": 32,
        "sym": True,
        "desc_act": False,
    }
    model.quantize_config.dynamic = dynamic
    model.quantize_config.pack_impl = "original"
    model.requantize(
        CalibrationConfig(dataset=_calibration_texts(), batch_size=2),
        embed_quant_mode=QuantizeEmbed.INPUT,
    )
    assert isinstance(model.get_input_embeddings(), TorchQuantEmbeddings)
    assert model._model_free_weight_only_replacements_only is True

    source_tensors = _saved_tensors(quantized_dir)
    non_embedding_tensors = {
        key: value
        for key, value in source_tensors.items()
        if not key.startswith("model.embed_tokens.")
    }
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        expected = model.model(input_ids=ids).logits

    for name, save in (("save", model.save), ("save_pretrained", model.save_pretrained)):
        output = tmp_path / f"replacement-{name}"
        save(str(output))

        keys = _saved_tensor_keys(output)
        for suffix in ("g_idx", "qweight", "qzeros", "scales"):
            assert f"model.embed_tokens.{suffix}" in keys
        assert "model.embed_tokens.weight" not in keys

        saved_tensors = _saved_tensors(output)
        for key, tensor in non_embedding_tensors.items():
            assert key in saved_tensors
            torch.testing.assert_close(saved_tensors[key], tensor, rtol=0, atol=0)

        reloaded = GPTQModel.load(str(output), device="cpu")
        assert isinstance(reloaded.get_input_embeddings(), TorchQuantEmbeddings)
        with torch.inference_mode():
            torch.testing.assert_close(
                reloaded.model(input_ids=ids).logits,
                expected,
                rtol=0,
                atol=2e-3,
            )
