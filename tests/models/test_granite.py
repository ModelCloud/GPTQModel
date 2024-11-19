from model_test import ModelTest  # noqa: E402


class TestGranite(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/granite-3.0-2b-instruct" # "ibm-granite/granite-3.0-2b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4505
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4770
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_granite(self):
        self.quant_lm_eval()
