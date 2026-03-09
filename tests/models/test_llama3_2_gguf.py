# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchQuantLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import WeightOnlyConfig


class TestLlama3_2_GGUF(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"

    FORMAT = FORMAT.GGUF
    BITS = "q4_k_m"
    LOAD_BACKEND = BACKEND.TORCH
    WEIGHT_ONLY = WeightOnlyConfig(method="gguf")

    def test_llama3_2_gguf_full_model(self):
        model, tokenizer, _ = self.quantModel(
            self.NATIVE_MODEL_ID,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
            call_perform_post_quant_validation=False,
        )

        try:
            self.check_kernel(model, {GGUFTorchQuantLinear})
            self.assertInference(model, tokenizer)
        finally:
            self._cleanup_quantized_model(model, enabled=self.DELETE_QUANTIZED_MODEL)
