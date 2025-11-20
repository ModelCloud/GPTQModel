# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os
import tempfile

from parameterized import parameterized

from gptqmodel.nn_modules.qlinear import BaseQuantLinear


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig, QuantizeEmbed
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

    # "/monster/data/model/Qwen1.5-1.8B-Chat-GPTQ-4bits-gp32"
    # lm_eval result:
    # GSM8K_PLATINUM_COT: 0.3259
    # MMLU_STEM: 0.3869
    # ARC_CHALLENGE: 0.3294, 0.3217 # (acc, acc_norm)
    tied_false_lm_eval_dict = {
        QuantizeEmbed.INPUT: {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: 0.3358,
            EVAL.LM_EVAL.MMLU_STEM: 0.3815,
            EVAL.LM_EVAL.ARC_CHALLENGE: (0.3233, 0.3242),  # (acc, acc_norm)
        },
        QuantizeEmbed.OUTPUT: {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: 0.3639,
            EVAL.LM_EVAL.MMLU_STEM: 0.3935,
            EVAL.LM_EVAL.ARC_CHALLENGE: (0.3250, 0.3208),  # (acc, acc_norm)
        },
        QuantizeEmbed.BOTH: {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: 0.3374,
            EVAL.LM_EVAL.MMLU_STEM: 0.3828,
            EVAL.LM_EVAL.ARC_CHALLENGE: (0.3208, 0.3242),  # (acc, acc_norm)
        },
    }

    # "/monster/data/model/Llama-3.2-1B-Instruct-GPTQ-4bits-gp32"
    # lm_eval result:
    # GSM8K_PLATINUM_COT: 0.0802
    # MMLU_STEM: 0.3587
    # ARC_CHALLENGE: 0.314, 0.3643 # (acc, acc_norm)
    tied_true_lm_eval_dict = {
        QuantizeEmbed.INPUT: {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: 0.0363,
            EVAL.LM_EVAL.MMLU_STEM: 0.3403,
            EVAL.LM_EVAL.ARC_CHALLENGE: (0.3054, 0.3430),  # (acc, acc_norm)
        },
        QuantizeEmbed.OUTPUT: {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: 0.0843,
            EVAL.LM_EVAL.MMLU_STEM: 0.3660,
            EVAL.LM_EVAL.ARC_CHALLENGE: (0.3139, 0.3634),  # (acc, acc_norm)
        },
        QuantizeEmbed.BOTH: {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: 0.03473,
            EVAL.LM_EVAL.MMLU_STEM: 0.3235,
            EVAL.LM_EVAL.ARC_CHALLENGE: (0.3046, 0.3225),  # (acc, acc_norm)
        },
    }

    requantize_tied_false_cases = [
        (QuantizeEmbed.INPUT, 0.0000),
    ]

    # "/monster/data/model/Qwen1.5-1.8B-Chat-GPTQ-4bits-gp32"
    # @parameterized.expand(requantize_tied_false_cases)
    def _test_requantize(self, model_id_or_path: str, embed_quant_mode: QuantizeEmbed,
                         expect_tied_word_embeddings: bool):
        assert embed_quant_mode is not None

        model = GPTQModel.load(model_id_or_path, device_map="auto", quantize_config=QuantizeConfig(offload_to_disk=False, bits=4, group_size=32))

        assert model.config.tie_word_embeddings == expect_tied_word_embeddings

        lm_eval_dict = self.tied_true_lm_eval_dict if model.config.tie_word_embeddings else self.tied_false_lm_eval_dict
        self.EVAL_TASKS = {
            EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
                "chat_template": True,
                "exact_match,flexible-extract": {
                    "value": lm_eval_dict[embed_quant_mode][EVAL.LM_EVAL.GSM8K_PLATINUM_COT],
                    "floor_pct": 0.04,
                },
            },
            EVAL.LM_EVAL.MMLU_STEM: {
                "chat_template": False,
                "acc": {
                    "value": lm_eval_dict[embed_quant_mode][EVAL.LM_EVAL.MMLU_STEM],
                    "floor_pct": 0.04,
                },
            },
            EVAL.LM_EVAL.ARC_CHALLENGE: {
                "chat_template": True,
                "acc": {
                    "value": lm_eval_dict[embed_quant_mode][EVAL.LM_EVAL.ARC_CHALLENGE][0],
                    "floor_pct": 0.04,
                },
                "acc_norm": {
                    "value": lm_eval_dict[embed_quant_mode][EVAL.LM_EVAL.ARC_CHALLENGE][1],
                    "floor_pct": 0.04,
                },
            },
        }

        calibration = self.load_dataset(model.tokenizer, self.DATASET_SIZE)
        model.requantize(calibration=calibration, embed_quant_mode=embed_quant_mode)

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

            assert not model.config.tie_word_embeddings

            print("model.get_input_embeddings()", model.get_input_embeddings())
            print("model.get_output_embeddings()", model.get_output_embeddings())
            if embed_quant_mode == QuantizeEmbed.INPUT:
                assert isinstance(model.get_input_embeddings(), BaseQuantLinear)
            elif embed_quant_mode == QuantizeEmbed.OUTPUT:
                assert isinstance(model.get_output_embeddings(), BaseQuantLinear)
            elif embed_quant_mode == QuantizeEmbed.BOTH:
                assert isinstance(model.get_input_embeddings(), BaseQuantLinear)
                assert isinstance(model.get_output_embeddings(), BaseQuantLinear)

            task_results = self.lm_eval(model=model,
                                        trust_remote_code=self.TRUST_REMOTE_CODE,
                                        delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
            self.check_results(task_results)

    requantize_cases = [
        (QuantizeEmbed.INPUT),
        (QuantizeEmbed.OUTPUT),
        (QuantizeEmbed.BOTH),
    ]

    @parameterized.expand(requantize_cases)
    def test_requantize_with_tied_false(self, embed_quant_mode: QuantizeEmbed):
        self._test_requantize(model_id_or_path="/monster/data/model/Qwen1.5-1.8B-Chat-GPTQ-4bits-gp32",
                              embed_quant_mode=embed_quant_mode, expect_tied_word_embeddings=False)

    @parameterized.expand(requantize_cases)
    def test_requantize_with_tied_true(self, embed_quant_mode: QuantizeEmbed):
        self._test_requantize(model_id_or_path="/monster/data/model/Llama-3.2-1B-Instruct-GPTQ-4bits-gp32",
                              embed_quant_mode=embed_quant_mode, expect_tied_word_embeddings=True)
