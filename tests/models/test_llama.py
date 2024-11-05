from model_test import ModelTest  # noqa: E402


class TestLlama(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/CodeLlama-7b-hf"
    NATIVE_ARC_CHALLENGE_ACC = 0.3234
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3507
    TRUST_REMOTE_CODE = True

    def test_llama(self):
        self.quant_lm_eval()

