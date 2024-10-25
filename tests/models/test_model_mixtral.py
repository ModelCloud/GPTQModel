from model_test import ModelTest

class TestMixtral(ModelTest):
    NATIVE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    def test_mixtral(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])