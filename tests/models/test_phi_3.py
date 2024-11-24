from model_test import ModelTest


class TestPhi_3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Phi-3-mini-4k-instruct" # "microsoft/Phi-3-mini-4k-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.5401
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5674
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_phi_3(self):
        self.quant_lm_eval()
