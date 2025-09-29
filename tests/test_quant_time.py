# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time  # noqa: E402

from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402


class TestQuantTime(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    DATASETS_MAX_COUNT = 128
    QUANT_TIME = 116
    MAX_DELTA_PERCENT = 5 # %

    def test_quant_time(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=True,
        )

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )
        tokenizer = model.tokenizer

        datasets = self.load_dataset(tokenizer, self.DATASETS_MAX_COUNT)

        start = time.time()
        model.quantize(
            calibration=datasets,
            # calibration_dataset_concat_size=2048,
            batch_size=4,
        )
        end_time = time.time()

        quant_time = end_time - start
        diff_pct = (quant_time / self.QUANT_TIME)

        print("**************** Quant Time Result Info****************")
        print(f"Quant Time: {quant_time}s vs Expected: {self.QUANT_TIME}s, diff: {diff_pct}")
        print("**************** Quant Time Result Info End****************")

        self.assertTrue(abs(diff_pct) <= self.MAX_DELTA_PERCENT,
                        f"Quant Time(s): Actual `{quant_time}` vs Expected `{self.QUANT_TIME}`, diff {diff_pct:.2f}% is out of range of {self.MAX_DELTA_PERCENT}%]")
