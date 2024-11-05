from model_test import ModelTest # noqa: E402



class TestLlama3_1(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.5154
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5520
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.15

    def test_llama3_1(self):
        self.quant_lm_eval()


