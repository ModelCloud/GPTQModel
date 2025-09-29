# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import BACKEND  # noqa: E402


class TestsTorchFused(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"  # "bigscience/bloom-560m"
    NATIVE_ARC_CHALLENGE_ACC = 0.28
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.31
    TORCH_DTYPE = torch.float16
    LOAD_BACKEND = BACKEND.TORCH_FUSED
    DELETE_QUANTIZED_MODEL = False
    USE_VLLM = False

    def test_torch_fused(self):
        self.quant_lm_eval()
