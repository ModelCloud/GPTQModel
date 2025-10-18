# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from models.model_test import ModelTest  # noqa: E402

from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.3567, "floor_pct": 0.36},
            "acc_norm": {"value": 0.3805, "floor_pct": 0.36},
        },
    }
    FORMAT = FORMAT.GPTQ
    SYM = False

    def test(self):
        self.quant_lm_eval()
