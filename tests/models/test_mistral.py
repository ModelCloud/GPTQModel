from model_test import ModelTest  # noqa: E402


class TestMistral(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Mistral-7B-Instruct-v0.2" # "mistralai/Mistral-7B-Instruct-v0.2"
    NATIVE_ARC_CHALLENGE_ACC = 0.5427
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5597
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_mistral(self):
        self.quant_lm_eval()
