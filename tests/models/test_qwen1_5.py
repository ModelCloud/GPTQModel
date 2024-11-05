from model_test import ModelTest # noqa: E402


class TestQwen1_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
    NATIVE_ARC_CHALLENGE_ACC = 0.2568
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2918
    TRUST_REMOTE_CODE = True

    def test_qwen1_5(self):
        self.quant_lm_eval()
