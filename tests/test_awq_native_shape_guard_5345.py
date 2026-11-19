# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""GPTQModel.load validates packed AWQ shapes against the built module."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.ports.native.qwen2.configuration_qwen2 import Qwen2Config
from gptqmodel.models.ports.native.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from gptqmodel.quantization.config import AWQConfig


_HIDDEN = 128
_INTERMEDIATE = 256
_VOCAB = 128
_LAYERS = 3


def _build_native_awq_checkpoint(directory: Path, group_size: int) -> Path:
    source = directory / "source"
    checkpoint = directory / "awq-native"

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).to(torch.float16).save_pretrained(source)

    generator = torch.Generator().manual_seed(1)
    calibration = [
        {"input_ids": torch.randint(0, _VOCAB, (1, 32), generator=generator)}
        for _ in range(8)
    ]
    model = GPTQModel.load(
        str(source),
        AWQConfig(bits=4, group_size=group_size, offload_to_disk=False),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(str(checkpoint))

    manifest = json.loads((checkpoint / "config.json").read_text())["quantization_config"]
    assert manifest["quant_method"] == "awq"
    assert manifest["group_size"] == group_size
    assert not manifest.get("dynamic")
    assert any(
        str(entry).startswith("gptqmodel")
        for entry in manifest.get("meta", {}).get("quantizer", ())
    )
    return checkpoint


@pytest.fixture(scope="module", params=[128, -1], ids=["grouped", "per_channel"])
def awq_native_checkpoint(tmp_path_factory, request):
    return _build_native_awq_checkpoint(tmp_path_factory.mktemp("awq_native_5345"), request.param)


def test_uncorrupted_native_checkpoint_loads(awq_native_checkpoint):
    model = GPTQModel.load(
        str(awq_native_checkpoint), device="cpu", dtype=torch.float16, backend=BACKEND.TORCH
    )
    module = model.model.model.layers[1].self_attn.q_proj

    assert module.group_size == module.in_features == _HIDDEN
    assert tuple(module.qzeros.shape) == (1, 16)
    assert tuple(module.scales.shape) == (1, 128)


@pytest.mark.parametrize("buffer", ["qzeros", "scales"])
def test_wrong_shape_native_packed_buffer_is_rejected(awq_native_checkpoint, tmp_path, buffer):
    corrupted = tmp_path / "awq-native-corrupt"
    shutil.copytree(awq_native_checkpoint, corrupted)
    shard = corrupted / "model.safetensors"
    tensors = load_file(str(shard))
    key = f"model.layers.1.self_attn.q_proj.{buffer}"
    tensors[key] = tensors[key].repeat(4, 1)
    save_file(tensors, str(shard), metadata={"format": "pt"})

    with pytest.raises(ValueError, match=rf'"{buffer}" .*has shape'):
        GPTQModel.load(
            str(corrupted), device="cpu", dtype=torch.float16, backend=BACKEND.TORCH
        )
