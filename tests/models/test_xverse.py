from model_test import ModelTest


class TestXVerse(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/XVERSE-7B-Chat" # "xverse/XVERSE-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.4198
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4044
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = True
    BATCH_SIZE = 6
    USE_VLLM = False

    def test_xverse(self):
        self.quant_lm_eval()
