from model_test import ModelTest

class TestGlm(ModelTest):
    NATIVE_MODEL_ID = "THUDM/chatglm3-6b"

    def test_glm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        result = self.generateChat(model, tokenizer)
        self.assertTrue(len(result) > 0)

