from model_test import ModelTest

class TestOvis1_6_Llama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ovis1.6-Llama3.2-3B"
    NATIVE_ARC_CHALLENGE_ACC = 0.2739
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3055

    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = False
    BATCH_SIZE = 1

    def test_ovis_1_6(self):
        self.quant_lm_eval()
