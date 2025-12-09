# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# -- do not touch
import os
import tempfile

from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel.utils.torch import torch_empty_cache


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

# isort: off
# isort: on
from parameterized import parameterized  # noqa: E402
from safetensors import safe_open

from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402


class TestModelSave(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_id, use_fast=True)

        traindata = load_dataset(path="/monster/data/model/dataset/nm-calibration", name="LLM", split="train")
        self.calibration_dataset = traindata.select(range(1))

    @parameterized.expand([
        True,
        False,
    ])
    def test_model_save_with_non_persistent_buffer(self, offload_to_disk):
        quantize_config = QuantizeConfig(
            bits=4,
            offload_to_disk=offload_to_disk,
        )

        model = GPTQModel.load(
            self.pretrained_model_id,
            quantize_config=quantize_config,
        )
        model.quantize(self.calibration_dataset, batch_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            model.save(tmp_dir_name)

            del model
            torch_empty_cache()

            with safe_open(tmp_dir_name+"/model.safetensors", framework="pt") as f:
                print("weight_map", f.keys())
                self.assertNotIn('model.rotary_emb.inv_freq', f.keys())
