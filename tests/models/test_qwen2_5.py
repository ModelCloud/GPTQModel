from tests.model_test import ModelTest  # noqa: E402


class TestQwen2_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.2739
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3055
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    BATCH_SIZE = 6

    def test_qwen2_5(self):
        self.quant_lm_eval()
