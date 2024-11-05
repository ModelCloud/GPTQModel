from model_test import ModelTest # noqa: E402


class TestQwen2_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4343
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4676
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = True

    def test_qwen2_5(self):
        self.quant_lm_eval()


