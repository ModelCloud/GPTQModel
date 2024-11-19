from model_test import ModelTest  # noqa: E402


class TestInternlm2_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm2_5-1_8b-chat" # "internlm/internlm2_5-1_8b-chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.3217
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3575
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 6

    def test_internlm2_5(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()


