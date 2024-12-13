from model_test import ModelTest


class TestBaiChuan(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Baichuan2-7B-Chat" # "baichuan-inc/Baichuan2-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.4104
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4317
    MODEL_MAX_LEN = 4096
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 6

    def test_baichuan(self):
        self.quant_lm_eval()
