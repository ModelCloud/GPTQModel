from model_test import ModelTest


class TestGlm(ModelTest):
    # real: THUDM/glm-4-9b-chat-hf
    NATIVE_MODEL_ID = "/monster/data/model/glm-4-9b-chat-hf"
    NATIVE_ARC_CHALLENGE_ACC = 0.5154
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5316
    USE_VLLM = False

    def test_glm(self):
        self.quant_lm_eval()

