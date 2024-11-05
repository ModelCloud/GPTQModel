from model_test import ModelTest  # noqa: E402


class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "internlm/internlm-7b"
    NATIVE_ARC_CHALLENGE_ACC = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4309
    TRUST_REMOTE_CODE = True

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()
