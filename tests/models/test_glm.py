from model_test import ModelTest  # noqa: E402


class TestGlm(ModelTest):
    # real: THUDM/glm-4-9b-chat-hf
    NATIVE_MODEL_ID = "/monster/data/model/glm-4-9b-chat-hf"
    USE_VLLM = False

    def test_glm(self):
        self.quant_lm_eval()

