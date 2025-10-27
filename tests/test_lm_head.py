# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os
import tempfile

from datasets import load_dataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402


class TestLmHeadLoad(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"  # "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
    DEVICE = "cuda:0"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.2799, "floor_pct": 0.2},
            "acc_norm": {"value": 0.3046, "floor_pct": 0.2},
        },
    }

    def test_load(self):
        model = GPTQModel.load(self.NATIVE_MODEL_ID, device=self.DEVICE)

        # validate lm_head is loaded as quantized layer
        assert isinstance(model.model.lm_head, BaseQuantLinear)

    def test_eval(self):
        self.quant_lm_eval()


class TestLmHeadQuant(ModelTest):
    EXPECT_LM_HEAD_LOSS = 0.0094

    sample_length = 1024
    samples = 128
    model_id = "/monster/data/model/Qwen1.5-1.8B-Chat"

    @classmethod
    def setUpClass(cls):
        calibration_dataset = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz", split="train").filter(lambda x: len(x["text"]) >= cls.sample_length).select(range(cls.samples))["text"]

        # Truncating sample text to reduce memory usage
        cls.calibration_dataset = [c[:cls.sample_length] for c in calibration_dataset]

    def test_quant_lm_head(self):
        self.EVAL_TASKS = {
            EVAL.LM_EVAL.ARC_CHALLENGE: {
                "chat_template": True,
                "acc": {"value": 0.3148464163822526, "floor_pct": 0.2},
                "acc_norm": {"value": 0.3310580204778157, "floor_pct": 0.2},
            },
        }

        quant_config = QuantizeConfig(bits=4, group_size=32, lm_head=True)

        model = GPTQModel.load(self.model_id, quant_config)

        model.quantize(self.calibration_dataset, batch_size=8)

        self.check_lm_head_loss(model.quant_log)

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
                                        trust_remote_code=self.TRUST_REMOTE_CODE,
                                        delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
            self.check_results(task_results)
