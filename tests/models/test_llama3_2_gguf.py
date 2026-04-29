# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.quantization import FORMAT, METHOD


class TestLlama3_2_GGUF(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"

    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.3871,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.3955,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3106,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3532,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
    }
    METHOD = METHOD.GGUF
    FORMAT = FORMAT.GGUF
    BITS = "q4_k_m"
    LOAD_BACKEND = BACKEND.GGUF_TORCH
    KERNEL_INFERENCE = {GGUFTorchLinear}

    def test_llama3_2_gguf_full_model(self):
        self.quantize_and_evaluate()

        module = self.model.model.model.layers[0].self_attn.q_proj
        assert isinstance(module, GGUFTorchLinear)
        assert module.gguf_tensor_qtype == "Q4_K"
        assert hasattr(module, "qweight")
        assert tuple(module.qweight.shape) == (2048, module._bytes_per_row())
        assert module.qweight.dtype == torch.uint8
