import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from model_test import ModelTest


class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm-7b" # "internlm/internlm-7b"
    NATIVE_ARC_CHALLENGE_ACC = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4309
    TRUST_REMOTE_CODE = True
    USE_VLLM = False

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()
