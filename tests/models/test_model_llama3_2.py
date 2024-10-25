from model_test import ModelTest

class TestLlama3_2(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    def test_llama3_2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])