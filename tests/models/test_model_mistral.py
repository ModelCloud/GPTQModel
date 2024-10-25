from model_test import ModelTest

class TestMistral(ModelTest):
    NATIVE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

    def test_mistral(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])