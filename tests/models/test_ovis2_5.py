# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from model_test import ModelTest
from ovis.image_to_test_dataset import get_calib_dataset

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.ovis2_5 import Ovis2_5QModel
from gptqmodel.quantization.config import QuantizeConfig


def test_ovis2_5_config_shape():
    config_path = Path("/monster/data/model/Ovis2.5-9B/config.json")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    assert config["model_type"].lower() == "ovis2_5"
    assert config["visual_vocab_size"] == 65536
    assert config["llm_config"]["num_hidden_layers"] == 36
    assert config["vit_config"]["model_type"] == "siglip2_navit"


def test_ovis2_5_class_metadata():
    assert Ovis2_5QModel.require_trust_remote_code is True
    assert Ovis2_5QModel.pre_lm_head_norm_module == "llm.model.norm"
    assert Ovis2_5QModel.module_tree[0] == "llm"


def test_ovis2_5_prepare_dataset_quantization_ready():
    instance = object.__new__(Ovis2_5QModel)
    instance.IGNORE_ID = -100
    instance.quantize_config = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True, mock_quantization=True)
    fake_visual_tokenizer = SimpleNamespace(vit=SimpleNamespace(dtype=torch.float32))

    def _preprocess(_messages):
        input_ids = torch.tensor([[0, 1, 2]])
        pixel_values = torch.ones((1, 3), dtype=torch.float32)
        grid_thws = torch.tensor([[1, 1, 1]], dtype=torch.long)
        return input_ids, pixel_values, grid_thws

    instance.model = SimpleNamespace(
        preprocess_inputs=_preprocess,
        text_tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2),
        visual_tokenizer=fake_visual_tokenizer,
        vte=SimpleNamespace(),
    )

    sample_dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe the scene"},
                    ],
                }
            ]
        }
    ]

    prepared = instance.prepare_dataset(sample_dataset, batch_size=1)
    assert len(prepared) == 1
    batch = prepared[0]
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels", "pixel_values", "grid_thws"}
    assert batch["input_ids"].shape[0] == 1
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert isinstance(batch["pixel_values"], torch.Tensor) and batch["pixel_values"].shape[-1] == 3
    assert isinstance(batch["grid_thws"], torch.Tensor)


class TestOvis2_5Quant(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ovis2.5-9B"
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = False
    QUANT_BATCH_SIZE = 1
    DATASET_SIZE = 2
    POST_QUANT_VALIDATION_BACKENDS = []
    MOCK_QUANTIZATION = False
    OFFLOAD_TO_DISK = False
    LOAD_BACKEND = BACKEND.TORCH
    QUANT_BACKEND = BACKEND.TORCH
    MAX_QUANT_LAYERS = 1

    def test_quantize_single_layer(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for Ovis2.5 quantization test")

        quant_cfg = QuantizeConfig(
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            desc_act=False,
            act_group_aware=self.ACT_GROUP_AWARE,
            sym=self.SYM,
            mock_quantization=False,
            fail_safe=self.FAIL_SAFE,
            v2=self.V2,
        )
        quant_cfg.device = "cuda:0"

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quant_cfg,
            trust_remote_code=True,
            dtype=self.TORCH_DTYPE,
        )

        calibration_dataset = get_calib_dataset(model)[: self.DATASET_SIZE]

        prev_max_layers = os.environ.get("GPTQMODEL_MAX_QUANT_LAYERS")
        os.environ["GPTQMODEL_MAX_QUANT_LAYERS"] = str(self.MAX_QUANT_LAYERS)
        try:
            model.quantize(calibration_dataset, batch_size=self.QUANT_BATCH_SIZE)
        finally:
            if prev_max_layers is None:
                os.environ.pop("GPTQMODEL_MAX_QUANT_LAYERS", None)
            else:
                os.environ["GPTQMODEL_MAX_QUANT_LAYERS"] = prev_max_layers

        assert model.quantized is True
        torch.cuda.empty_cache()
