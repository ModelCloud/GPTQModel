# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A module-replacement save must keep live generation settings (#5387).

Requantizing an embedding rewrites only the affected shards, and the writer
copies the source checkpoint's auxiliary files before saving config, tokenizer
and processor. It never serialized ``generation_config.json``, so edits such as
``loaded.generation_config.max_new_tokens = 23`` were silently dropped and the
reload used the checkpoint's original stopping behavior.
"""

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, PreTrainedModel

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QuantizeConfig


class _TinyModel(PreTrainedModel):
    def __init__(self) -> None:
        super().__init__(LlamaConfig(num_hidden_layers=1, hidden_size=32))
        self.norm = torch.nn.LayerNorm(32)
        self.proj = TorchLinear(
            bits=4,
            group_size=32,
            sym=True,
            desc_act=False,
            in_features=32,
            out_features=32,
            bias=False,
        )

    def get_input_embeddings(self):
        return self.proj

    def get_output_embeddings(self):
        return None

    def can_generate(self) -> bool:
        return True


class _NonGenerativeTinyModel(_TinyModel):
    """A model class without a generation path, like an encoder-only head."""

    def can_generate(self) -> bool:
        return False


def _writer(tmp_path, *, can_generate: bool = True) -> LlamaQModel:
    source = tmp_path / "source"
    source.mkdir()
    model = _TinyModel() if can_generate else _NonGenerativeTinyModel()
    model.config.save_pretrained(source)
    save_file({"norm.weight": model.norm.weight.detach()}, source / "model.safetensors")

    writer = LlamaQModel(
        model=model,
        quantized=True,
        quantize_config=QuantizeConfig(group_size=32, offload_to_disk=False),
        qlinear_kernel=TorchLinear,
        model_local_path=str(source),
    )
    # A loaded checkpoint holds the dense embedding; the replacement writer
    # swaps it for the packed module under the same prefix.
    save_file({"proj.weight": torch.ones(32, 32)}, source / "model.safetensors")
    writer._module_replacement_prefixes = {"proj"}
    writer._module_replacement_source = SimpleNamespace(
        model_local_path=str(source),
        _weight_map={"proj.weight": "model.safetensors"},
    )
    return writer


def test_module_replacement_save_keeps_live_generation_settings(tmp_path):
    writer = _writer(tmp_path)
    writer.model.generation_config.max_new_tokens = 23

    output = tmp_path / "saved"
    writer.save_quantized_embeddings(str(output))

    saved = json.loads((output / "generation_config.json").read_text())
    assert saved["max_new_tokens"] == 23
    # The replacement tensors are still published by the same save.
    assert (output / "model.safetensors").is_file()


def test_non_generative_model_writes_no_generation_config(tmp_path):
    """The standard writer's ``can_generate`` guard applies here too."""
    writer = _writer(tmp_path, can_generate=False)

    output = tmp_path / "saved"
    writer.save_quantized_embeddings(str(output))

    assert not (output / "generation_config.json").exists()
