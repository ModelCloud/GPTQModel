import os
import sys

import torch  # noqa: E402

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ..model_test import ModelTest  # noqa: E402


class TestStarCode2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/starcoder2-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2901
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3080
    TORCH_DTYPE = torch.float16
    def test_starcode2(self):
        self.quant_lm_eval()


