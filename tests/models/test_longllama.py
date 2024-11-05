from model_test import ModelTest # noqa: E402


class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "syzymon/long_llama_3b_instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3515
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3652
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_longllama(self):
        self.quant_lm_eval()
