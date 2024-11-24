import os
import sys

import torch  # noqa: E402

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ..model_test import ModelTest  # noqa: E402


class TestGptBigCode(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt_bigcode-santacoder" # "bigcode/gpt_bigcode-santacoder"
    NATIVE_ARC_CHALLENGE_ACC = 0.1689
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2056
    TORCH_DTYPE = torch.float16
    TRUST_REMOTE_CODE = True

    def test_gptbigcode(self):
        self.quant_lm_eval()

