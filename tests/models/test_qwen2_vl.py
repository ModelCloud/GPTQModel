from model_test import ModelTest

class TestQwen2_VL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2-VL-2B-Instruct"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.2329
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2765
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    BATCH_SIZE = 6

    def test_qwen2_vl(self):
        self.quant_lm_eval()