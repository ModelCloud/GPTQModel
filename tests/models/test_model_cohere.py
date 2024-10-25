from model_test import ModelTest

class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "CohereForAI/aya-expanse-8b"

    def test_cohere(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])