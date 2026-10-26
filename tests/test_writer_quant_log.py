# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, PreTrainedModel

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.writer import (
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_TIME,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
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


@pytest.mark.parametrize("samples", [128, 0, None])
def test_quant_log_columns(writer: LlamaQModel, tmp_path: Path, samples: int | None) -> None:
    entry = {
        PROCESS_LOG_LAYER: 0,
        PROCESS_LOG_MODULE: "proj",
        QUANT_LOG_LOSS: 0.25,
        QUANT_LOG_DAMP: 0.05,
        PROCESS_LOG_TIME: 1.5,
    }
    if samples is not None:
        entry[QUANT_LOG_NSAMPLES] = samples
    writer.quant_log = [entry]
    output = tmp_path / "saved"
    writer.save_quantized(str(output))
    with (output / "quant_log.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            PROCESS_LOG_LAYER: "0",
            PROCESS_LOG_MODULE: "proj",
            QUANT_LOG_LOSS: "0.25",
            QUANT_LOG_NSAMPLES: "" if samples is None else str(samples),
            QUANT_LOG_DAMP: "0.05",
            PROCESS_LOG_TIME: "1.5",
        }
    ]
