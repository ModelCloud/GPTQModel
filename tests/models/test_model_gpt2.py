from model_test import ModelTest

class TestGpt2(ModelTest):
    NATIVE_MODEL_ID = "openai-community/gpt2-medium"

    def test_gpt2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])