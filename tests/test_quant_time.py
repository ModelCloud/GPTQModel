# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time  # noqa: E402

from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402


class TestQuantTime(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    INPUTS_MAX_LENGTH = 2048
    DATASETS_MAX_COUNT = 128
    QUANT_TIME = 136
    MAX_DELTA_PERCENT = 5 # %

    def test_quant_time(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
        )

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )
        tokenizer = model.tokenizer

        datasets = self.load_dataset(tokenizer)

        start_time = time.time()
        model.quantize(datasets, batch_size=4)
        end_time = time.time()

        quant_time = end_time - start_time
        diff_pct = (quant_time / self.QUANT_TIME)

        print("**************** Quant Time Result Info****************")
        print(f"Quant Time: {quant_time}s vs Expected: {self.QUANT_TIME}s, diff: {diff_pct}")
        print("**************** Quant Time Result Info End****************")

        self.assertTrue(abs(diff_pct) <= self.MAX_DELTA_PERCENT,
                        f"Quant Time(s): Actual `{quant_time}` vs Expected `{self.QUANT_TIME}`, diff {diff_pct:.2f}% is out of range of {self.MAX_DELTA_PERCENT}%]")
