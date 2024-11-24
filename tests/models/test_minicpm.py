import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from model_test import ModelTest


class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/MiniCPM-2B-128k" # "openbmb/MiniCPM-2B-128k"
    NATIVE_ARC_CHALLENGE_ACC = 0.3848
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 4

    def test_minicpm(self):
        self.quant_lm_eval()
