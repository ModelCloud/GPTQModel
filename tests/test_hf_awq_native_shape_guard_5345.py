# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""A native AWQ checkpoint with a wrong-shape packed buffer must not load silently.

Transformers 5.23 records a checkpoint tensor whose shape differs from the meta
module as a mismatched key unless ``hf_quantizer.param_needs_quantization`` is
true. The GPT-QModel bridge may claim that hook only when the built module's
geometry disagrees with the checkpoint's serialized per-module config, because
only then does ``hf_gptqmodel_repair_awq_load`` rebuild it. A native checkpoint
without ``dynamic`` has no such override, so a corrupted ``qzeros``/``scales``
must keep Transformers' rejection instead of loading with the wrong geometry
(#5345).
"""

import json
import shutil
from pathlib import Path

import pytest
import torch
from packaging.version import Version
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers import __version__ as transformers_version

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import AWQConfig


_HIDDEN = 128
_INTERMEDIATE = 256
_VOCAB = 128
_LAYERS = 3
_GROUP_SIZE = 128


pytestmark = pytest.mark.skipif(
    Version(transformers_version) < Version("5.23"),
    reason="the packed-shape guard under test was added in Transformers 5.23",
)


def _build_native_awq_checkpoint(directory: Path) -> Path:
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
        AWQConfig(bits=4, group_size=_GROUP_SIZE, offload_to_disk=False),
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
    assert manifest["group_size"] == _GROUP_SIZE
    assert not manifest.get("dynamic")
    assert any(
        str(entry).startswith("gptqmodel")
        for entry in manifest.get("meta", {}).get("quantizer", ())
    )
    return checkpoint


def _corrupt_packed_buffers(checkpoint: Path):
    changed = []
    for shard in sorted(checkpoint.glob("*.safetensors")):
        tensors = load_file(str(shard))
        for key in list(tensors):
            if "layers.1.self_attn.q_proj" in key and key.endswith((".qzeros", ".scales")):
                tensors[key] = tensors[key].repeat(4, 1)
                changed.append(key)
        save_file(tensors, str(shard))
    return changed


@pytest.fixture(scope="module")
def awq_native_checkpoint(tmp_path_factory):
    return _build_native_awq_checkpoint(tmp_path_factory.mktemp("awq_native_5345"))


def test_uncorrupted_native_checkpoint_loads(awq_native_checkpoint):
    model = AutoModelForCausalLM.from_pretrained(
        str(awq_native_checkpoint), device_map="cpu", dtype=torch.float16
    )
    module = model.model.layers[1].self_attn.q_proj

    assert int(module.group_size) == _GROUP_SIZE
    assert tuple(module.qzeros.shape) == (1, 16)
    assert tuple(module.scales.shape) == (1, 128)


def test_wrong_shape_native_packed_buffers_are_rejected(awq_native_checkpoint, tmp_path):
    corrupted = tmp_path / "awq-native-corrupt"
    shutil.copytree(awq_native_checkpoint, corrupted)
    changed = _corrupt_packed_buffers(corrupted)
    assert sorted(key.rsplit(".", 1)[-1] for key in changed) == ["qzeros", "scales"]

    with pytest.raises(RuntimeError, match="ignore_mismatched_sizes"):
        AutoModelForCausalLM.from_pretrained(
            str(corrupted), device_map="cpu", dtype=torch.float16
        )
