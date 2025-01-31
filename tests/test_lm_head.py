# Copyright 2025 ModelCloud
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

# -- do not touch
import os
import tempfile

from datasets import load_dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
# -- end do not touch
from models.model_test import ModelTest  # noqa: E402


class TestLmHeadLoad(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"  # "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
    DEVICE = "cuda:0"
    NATIVE_ARC_CHALLENGE_ACC = 0.2799
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3046
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2

    def test_load(self):
        model = GPTQModel.load(self.NATIVE_MODEL_ID, device=self.DEVICE)

        # validate lm_head is loaded as quantized layer
        assert isinstance(model.model.lm_head, BaseQuantLinear)

    def test_eval(self):
        self.quant_lm_eval()


class TestLmHeadQuant(ModelTest):
    APPLY_CHAT_TEMPLATE = True

    sample_length = 1024
    samples = 128
    model_id = "Qwen/Qwen1.5-1.8B-Chat"

    @classmethod
    def setUpClass(cls):
        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).filter(lambda x: len(x["text"]) >= cls.sample_length).select(range(cls.samples))["text"]

        # Truncating sample text to reduce memory usage
        cls.calibration_dataset = [c[:cls.sample_length] for c in calibration_dataset]

    def test_quant_lm_head(self):
        self.NATIVE_ARC_CHALLENGE_ACC = 0.3148464163822526
        self.NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3310580204778157

        quant_config = QuantizeConfig(bits=4, group_size=32, lm_head=True)

        model = GPTQModel.load(self.model_id, quant_config)

        model.quantize(self.calibration_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.tokenizer.save_pretrained(tmp_dir)
            model.save(tmp_dir)

            del model.tokenizer
            del model

            model = GPTQModel.load(
                tmp_dir,
                device_map="auto",
            )

            task_results = self.lm_eval(model=model,
                                        apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                                        trust_remote_code=self.TRUST_REMOTE_CODE,
                                        delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
            self.check_results(task_results)

    def test_quant_lm_head_low_gpu(self):
        self.NATIVE_ARC_CHALLENGE_ACC = 0.3199658703071672
        self.NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3225255972696246
        quant_config = QuantizeConfig(bits=4, group_size=32, lm_head=True, lm_head_low_gpu_mem_usage=True)

        model = GPTQModel.load(self.model_id, quant_config)

        model.quantize(self.calibration_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.tokenizer.save_pretrained(tmp_dir)
            model.save(tmp_dir)

            del model.tokenizer
            del model

            model = GPTQModel.load(
                tmp_dir,
                device_map="auto",
            )

            task_results = self.lm_eval(model=model,
                                        apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                                        trust_remote_code=self.TRUST_REMOTE_CODE,
                                        delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
            self.check_results(task_results)
