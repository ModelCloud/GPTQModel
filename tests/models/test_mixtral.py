from tests.model_test import ModelTest  # noqa: E402


class TestMixtral(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Mixtral-8x7B-Instruct-v0.1" # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    NATIVE_ARC_CHALLENGE_ACC = 0.5213
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5247
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 6

    def test_mixtral(self):
        self.quant_lm_eval()
