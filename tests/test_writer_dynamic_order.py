# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, PreTrainedModel

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QuantizeConfig, dynamic_get


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

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.proj

    def get_output_embeddings(self) -> None:
        return None


@pytest.fixture
def writer(tmp_path: Path) -> LlamaQModel:
    source = tmp_path / "source"
    source.mkdir()
    model = _TinyModel()
    model.config.save_pretrained(source)
    save_file({"norm.weight": model.norm.weight.detach()}, source / "model.safetensors")
    return LlamaQModel(
        model=model,
        quantized=True,
        quantize_config=QuantizeConfig(group_size=32, offload_to_disk=False),
        qlinear_kernel=TorchLinear,
        model_local_path=str(source),
    )


@pytest.mark.parametrize("embedding_only", [False, True])
@pytest.mark.parametrize(
    "dynamic",
    [
        {r"^proj$": {"bits": 4}, r".*": {"bits": 8}},
        {r"-:^skip$": {}, r"+:^proj$": {"bits": 4}, r".*": {"bits": 8}},
        {},
        None,
    ],
)
def test_dynamic_rule_order_survives_save(
    writer: LlamaQModel,
    tmp_path: Path,
    dynamic: dict | None,
    embedding_only: bool,
) -> None:
    writer.quantize_config = QuantizeConfig(
        group_size=32,
        offload_to_disk=False,
        dynamic=dynamic,
    )
    output = tmp_path / "saved"
    if embedding_only:
        source = Path(writer.model_local_path)
        save_file({"proj.weight": torch.ones(32, 32)}, source / "model.safetensors")
        writer._embedding_replacement_prefixes = {"proj"}
        writer._embedding_replacement_source = SimpleNamespace(
            model_local_path=str(source),
            _weight_map={"proj.weight": "model.safetensors"},
        )
        writer.save_quantized_embeddings(str(output))
    else:
        writer.save_quantized(str(output))
    saved = json.loads((output / "config.json").read_text())
    quant = json.loads((output / "quantize_config.json").read_text())
    expected = writer.quantize_config.dynamic
    for config in (saved["quantization_config"], quant):
        assert list(config.get("dynamic") or {}) == list(expected or {})
        for name in ("proj", "skip", "other"):
            assert dynamic_get(config.get("dynamic"), name, "bits") == dynamic_get(expected, name, "bits")
    assert saved["hidden_size"] == 32
    assert saved["model_type"] == "llama"
    assert saved["quantization_config"]["group_size"] == quant["group_size"] == 32
