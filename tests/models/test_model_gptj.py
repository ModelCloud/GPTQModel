from model_test import ModelTest

class TestGptJ(ModelTest):
    NATIVE_MODEL_ID = "EleutherAI/gpt-j-6b"

    def test_gptj(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])