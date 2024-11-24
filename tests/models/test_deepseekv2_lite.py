from tests.model_test import ModelTest


class TestDeepseekV2Lite(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/DeepSeek-Coder-V2-Lite-Instruct" # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4753
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4855
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_deepseekv2lite(self):
        self.quant_lm_eval()


