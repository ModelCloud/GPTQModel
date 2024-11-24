from model_test import ModelTest  # noqa: E402


class TestDeci(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/DeciLM-7B-instruct" # "Deci/DeciLM-7B-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.55
    TRUST_REMOTE_CODE = True
    USE_VLLM = False
    BATCH_SIZE = 6

    def test_deci(self):
        self.quant_lm_eval()
