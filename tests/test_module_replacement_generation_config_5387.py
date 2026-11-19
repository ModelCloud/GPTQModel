# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Module-replacement saves must keep the live generation settings (#5387).

``save_quantized_embeddings`` copies the source checkpoint's auxiliary files and
then rewrites the replaced tensors. It never serialized ``generation_config``,
so a ``max_new_tokens`` edit made on the loaded model disappeared from the
exported checkpoint.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import GenerationConfig, GenerationMixin, LlamaConfig, PreTrainedModel

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QuantizeConfig


class _TinyModel(PreTrainedModel, GenerationMixin):
    def __init__(self) -> None:
        super().__init__(LlamaConfig(num_hidden_layers=1, hidden_size=32, is_decoder=True))
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

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.proj

    def get_output_embeddings(self) -> None:
        return None


class _DenseModel(_TinyModel):
    """A checkpoint shell that cannot generate."""

    @classmethod
    def can_generate(cls) -> bool:
        return False


def _writer(tmp_path: Path, model_cls=_TinyModel) -> LlamaQModel:
    source = tmp_path / "source"
    source.mkdir()
    model = model_cls()
    model.config.save_pretrained(source)
    save_file({"proj.weight": torch.ones(32, 32)}, source / "model.safetensors")
    writer = LlamaQModel(
        model=model,
        quantized=True,
        quantize_config=QuantizeConfig(group_size=32, offload_to_disk=False),
        qlinear_kernel=TorchLinear,
        model_local_path=str(source),
    )
    writer._module_replacement_prefixes = {"proj"}
    writer._module_replacement_source = SimpleNamespace(
        model_local_path=str(source),
        _weight_map={"proj.weight": "model.safetensors"},
    )
    return writer


def test_module_replacement_save_writes_the_live_generation_config(tmp_path):
    writer = _writer(tmp_path)
    writer.model.generation_config = GenerationConfig(max_new_tokens=23)
    output = tmp_path / "saved"

    writer.save_quantized_embeddings(str(output))

    saved = json.loads((output / "generation_config.json").read_text(encoding="utf-8"))
    assert saved["max_new_tokens"] == 23


def test_module_replacement_save_does_not_invent_a_generation_config(tmp_path):
    writer = _writer(tmp_path, model_cls=_DenseModel)
    output = tmp_path / "saved"

    writer.save_quantized_embeddings(str(output))

    assert not (output / "generation_config.json").exists()
