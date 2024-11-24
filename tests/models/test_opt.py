import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ..model_test import ModelTest  # noqa: E402


class TestOpt(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/opt-125m" # "facebook/opt-125m"
    NATIVE_ARC_CHALLENGE_ACC = 0.1894
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2278

    def test_opt(self):
        self.quant_lm_eval()
