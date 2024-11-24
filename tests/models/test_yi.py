import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ..model_test import ModelTest  # noqa: E402


class TestYi(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Yi-Coder-1.5B-Chat" # "01-ai/Yi-Coder-1.5B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.2679
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2986
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = True
    BATCH_SIZE = 4

    def test_yi(self):
        self.quant_lm_eval()
