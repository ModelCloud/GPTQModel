from model_test import ModelTest  # noqa: E402


class TestLlama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/CodeLlama-7b-hf" # "meta-llama/CodeLlama-7b-hf"
    NATIVE_ARC_CHALLENGE_ACC = 0.3234
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3507
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 4

    def test_llama(self):
        self.quant_lm_eval()

