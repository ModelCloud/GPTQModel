# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5380: re-saving a checkpoint loaded through ``subfolder=``.

A nested GPTQ checkpoint loads correctly, but the export shell used to be built
from the repository root instead of the directory the load selected, so an
immediate ``save()`` failed with ``ValueError: Unrecognized model``. The wrapper
now keeps the resolved checkpoint directory, which this round trip pins.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import RTNConfig
from tests.test_adapter_targets import _write_tiny_llama_tokenizer


pytestmark = [pytest.mark.cpu]


def _nested_quantized_checkpoint(root):
    """Quantize a tiny Llama into ``root/gptq`` and return that directory."""

    nested = root / "gptq"
    nested.mkdir(parents=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(nested)
    _write_tiny_llama_tokenizer(nested)

    model = GPTQModel.load(
        str(nested), RTNConfig(bits=4, group_size=32, offload_to_disk=True)
    )
    model.quantize()
    model.save(str(nested))

    # The root deliberately carries no config: the load must resolve the nested
    # directory, and the save must not fall back to the root either.
    assert not (root / "config.json").exists()
    assert "quantization_config" in json.loads(
        (nested / "config.json").read_text(encoding="utf-8")
    )
    return nested


def _packed_tensors(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.model.state_dict().items()
        if name.endswith((".qweight", ".qzeros", ".scales"))
    }


def test_nested_checkpoint_saves_and_reloads_the_same_packed_tensors(tmp_path):
    _nested_quantized_checkpoint(tmp_path)

    loaded = GPTQModel.load(
        str(tmp_path), subfolder="gptq", device="cpu", backend=BACKEND.GPTQ_TORCH
    )
    assert loaded.quantized is True
    # The wrapper keeps the directory the load selected (#5380).
    assert os.path.samefile(loaded.model_local_path, tmp_path / "gptq")

    resaved = tmp_path / "resaved"
    loaded.save(str(resaved))

    reloaded = GPTQModel.load(
        str(resaved), device="cpu", backend=BACKEND.GPTQ_TORCH
    )
    assert reloaded.quantized is True

    before = _packed_tensors(loaded)
    after = _packed_tensors(reloaded)
    assert before, "the fixture produced no packed tensors"
    assert set(after) == set(before)
    for name, tensor in before.items():
        torch.testing.assert_close(after[name], tensor, rtol=0.0, atol=0.0)
