import os
import sys

import torch  # noqa: E402

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from model_test import ModelTest


class TestBloom(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/bloom-560m" # "bigscience/bloom-560m"
    NATIVE_ARC_CHALLENGE_ACC = 0.2201
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2440
    TORCH_DTYPE = torch.float16

    def test_bloom(self):
        self.quant_lm_eval()

