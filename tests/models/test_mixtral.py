from model_test import ModelTest # noqa: E402


class TestMixtral(ModelTest):
    NATIVE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    NATIVE_ARC_CHALLENGE_ACC = 0.5213
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5247
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_mixtral(self):
        self.quant_lm_eval()
