from model_test import ModelTest # noqa: E402


class TestPhi_1(ModelTest):
    NATIVE_MODEL_ID = "microsoft/phi-1"
    NATIVE_ARC_CHALLENGE_ACC = 0.2005
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2338
    TRUST_REMOTE_CODE = True

    def test_phi_1(self):
        self.quant_lm_eval()

