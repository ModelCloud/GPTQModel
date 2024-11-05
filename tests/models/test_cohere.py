from model_test import ModelTest # noqa: E402


class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "CohereForAI/aya-expanse-8b"
    NATIVE_ARC_CHALLENGE_ACC = 0.5401
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5640
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.12

    def test_cohere(self):
        self.quant_lm_eval()
