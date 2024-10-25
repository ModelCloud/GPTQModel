from model_test import ModelTest

class TestChatGlm(ModelTest):
    NATIVE_MODEL_ID = "THUDM/chatglm3-6b"

    def test_chatglm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])