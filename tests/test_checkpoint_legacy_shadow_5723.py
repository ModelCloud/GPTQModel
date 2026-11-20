# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A stale legacy ``gptq_model-*bit-*g`` export must not shadow a fresh save.

``save_quantized`` always writes the canonical ``model`` base name, but
``get_checkpoints`` probed the legacy AutoGPTQ name first. A directory that
already held a legacy export therefore reloaded the stale weights while
``config.json``/``quantize_config.json`` described the new one, silently.
"""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.quantization import QuantizeConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


LEGACY_BASE = "gptq_model-4bit-32g"


def _calibration_texts(count: int = 8):
    return [
        " ".join(["the quick brown fox"] * (1 + index % 3))
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy-shadow-5723")
    dense = root / "dense"
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(dense)
    build_wordlevel_tokenizer().save_pretrained(dense)

    model = GPTQModel.load(str(dense), QuantizeConfig(bits=4, group_size=32), device="cpu")
    model.quantize(CalibrationConfig(dataset=_calibration_texts(), batch_size=2))
    saved = root / "saved"
    model.save(str(saved))
    return model, saved


def _tensors(directory: Path):
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def test_loader_basename_order_prefers_the_canonical_name():
    from gptqmodel.models.loader import _quantized_checkpoint_basenames

    qcfg = QuantizeConfig(bits=4, group_size=32)
    assert _quantized_checkpoint_basenames(qcfg) == ["model", LEGACY_BASE]


def test_loader_prefers_the_fresh_canonical_checkpoint(quantized):
    """A leftover legacy file must not win over the just-written ``model``."""

    model, saved = quantized
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        expected = model.model(input_ids=ids).logits

    # Recreate the reported trigger: an older legacy-named export in the same
    # directory. Poison its qweight so a wrong resolution is visible.
    poisoned = {key: value.clone() for key, value in _tensors(saved).items()}
    for key in list(poisoned):
        if key.endswith(".qweight"):
            poisoned[key].zero_()
    save_file(poisoned, str(saved / f"{LEGACY_BASE}.safetensors"))

    reloaded = GPTQModel.load(str(saved), device="cpu")
    canonical = _tensors(saved)
    state = reloaded.model.state_dict()
    for key, tensor in canonical.items():
        if key in state:
            torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        actual = reloaded.model(input_ids=ids).logits
    # Reload re-runs the CPU pack/decode path, so accumulation order can shift
    # the last FP16 bit; a legacy load (zeroed qweight) is orders larger.
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-2)


def test_save_removes_a_stale_legacy_export(quantized, tmp_path):
    """The writer owns the output directory and drops the shadowing name."""

    model, _ = quantized
    output = tmp_path / "resaved"
    output.mkdir()
    for name in (
        f"{LEGACY_BASE}.safetensors",
        f"{LEGACY_BASE}.safetensors.index.json",
        f"{LEGACY_BASE}-00001-of-00002.safetensors",
    ):
        (output / name).write_bytes(b"stale")
    # A different export's legacy file is not this config's shadow.
    other = "gptq_model-8bit-128g.safetensors"
    (output / other).write_bytes(b"other")

    model.save(str(output))

    assert (output / "model.safetensors").is_file()
    for name in (
        f"{LEGACY_BASE}.safetensors",
        f"{LEGACY_BASE}.safetensors.index.json",
        f"{LEGACY_BASE}-00001-of-00002.safetensors",
    ):
        assert not (output / name).exists()
    assert (output / other).is_file()
