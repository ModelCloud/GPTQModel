from model_test import ModelTest  # noqa: E402


class TestGlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/chatglm3-6b" # "THUDM/chatglm3-6b"
    TRUST_REMOTE_CODE = True
    USE_VLLM = False

    def test_glm(self):
        self.quant_lm_eval()

