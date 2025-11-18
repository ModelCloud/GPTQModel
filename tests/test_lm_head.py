# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os
import tempfile

from datasets import load_dataset

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import EmbedQuantMode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402


# class TestLmHeadLoad(ModelTest):
#     NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"  # "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
#     DEVICE = "cuda:0"
#     EVAL_TASKS = {
#         EVAL.LM_EVAL.ARC_CHALLENGE: {
#             "acc": {"value": 0.2799, "floor_pct": 0.2},
#             "acc_norm": {"value": 0.3046, "floor_pct": 0.2},
#         },
#     }
#
#     def test_load(self):
#         model = GPTQModel.load(self.NATIVE_MODEL_ID, device=self.DEVICE)
#
#         # validate lm_head is loaded as quantized layer
#         assert isinstance(model.model.lm_head, BaseQuantLinear)
#
#     def test_eval(self):
#         self.quant_lm_eval()


class TestLmHeadQuant(ModelTest):
    EXPECT_LM_HEAD_LOSS = 0.0001088594
    # DATASET_CONCAT_SIZE = 2048
    EVAL_BATCH_SIZE = 64

    def test_requantize_lm_head(self):
        self.EVAL_TASKS = {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
                "chat_template": True,
                "exact_match,flexible-extract": {
                    "value": 0.3374,
                    "floor_pct": 0.04,
                },
            },
            EVAL.LM_EVAL.MMLU_STEM: {
                "chat_template": False,
                "acc": {
                    "value": 0.3828,
                    "floor_pct": 0.04,
                },
            },
            EVAL.LM_EVAL.ARC_CHALLENGE: {
                "chat_template": True,
                "acc": {
                    "value": 0.3208,
                    "floor_pct": 0.04,
                },
                "acc_norm": {
                    "value": 0.3242,
                    "floor_pct": 0.04,
                },
            },
        }

        model = GPTQModel.load("Qwen1.5-1.8B-Chat-GPTQ-4bits-gp32", device_map="auto")
        calibration = self.load_dataset(model.tokenizer, self.DATASET_SIZE)
        model.requantize(calibration=calibration, embed_quant_mode=EmbedQuantMode.BOTH)

        # self.check_lm_head_loss(model.quant_log)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.tokenizer.save_pretrained(tmp_dir)
            model.save(tmp_dir)

            del model.tokenizer
            del model

            model = GPTQModel.load(
                tmp_dir,
                device_map="auto",
            )

            assert isinstance(model.get_input_embeddings(), BaseQuantLinear)
            assert isinstance(model.get_output_embeddings(), BaseQuantLinear)

            task_results = self.lm_eval(model=model,
                                        trust_remote_code=self.TRUST_REMOTE_CODE,
                                        delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
            self.check_results(task_results)
