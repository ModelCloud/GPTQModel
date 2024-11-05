from model_test import ModelTest # noqa: E402


class TestGlm(ModelTest):
    NATIVE_MODEL_ID = "THUDM/chatglm3-6b"
    TRUST_REMOTE_CODE = True

    def test_glm(self):
        self.quant_lm_eval()

