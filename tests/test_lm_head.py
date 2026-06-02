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

from gptqmodel import GPTQModel, QuantizeEmbed  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402


class TestLmHeadLoad(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"  # "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
    DEVICE = "cuda:0"
    EVAL_TASKS = {
        "arc_challenge": {
            "acc": {"value": 0.2799, "floor_pct": 0.2},
            "acc_norm": {"value": 0.3046, "floor_pct": 0.2},
        },
    }

    def test_load(self):
        model = GPTQModel.load(self.NATIVE_MODEL_ID, device=self.DEVICE)

        # validate lm_head is loaded as quantized layer
        assert isinstance(model.model.lm_head, BaseQuantLinear)

    def test_eval(self):
        self.quantize_and_evaluate()


class TestLmHeadReQuant(ModelTest):
    EXPECT_EMBED_LOSS = 0.0000000004
    EXPECT_LM_HEAD_LOSS = 0.0001088594

    def test_requantize_lm_head(self):
        self.EVAL_TASKS = {
            "arc_challenge": {
                "chat_template": True,
                "acc": {"value": 0.3148464163822526, "floor_pct": 0.2},
                "acc_norm": {"value": 0.3310580204778157, "floor_pct": 0.2},
            },
        }

        model = GPTQModel.load("/monster/data/model/Qwen1.5-1.8B-Chat-GPTQ-4bits-gp32", device_map="auto")
        calibration = self.load_dataset(model.tokenizer, self.DATASET_SIZE)
        model.requantize(calibration=calibration, embed_quant_mode=QuantizeEmbed.OUTPUT)

        # self.check_loss(model.get_input_embeddings_name(), self.EXPECT_EMBED_LOSS, model.quant_log)
        self.check_loss(model.get_output_embeddings_name(), self.EXPECT_LM_HEAD_LOSS, model.quant_log)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = "./temp/Qwen1.5-1.8B-Chat-GPTQ-4bits-gp32-embed"
            model.tokenizer.save_pretrained(tmp_dir)
            model.save(tmp_dir)

            del model.tokenizer
            del model

            model = GPTQModel.load(
                tmp_dir,
                device_map="auto",
            )

            # assert isinstance(model.get_input_embeddings(), BaseQuantLinear)
            assert isinstance(model.get_output_embeddings(), BaseQuantLinear)

            task_results = self.evaluate_model(model=model,
                                               trust_remote_code=self.TRUST_REMOTE_CODE,
                                               delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
            self.check_results(task_results)