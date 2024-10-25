from model_test import ModelTest

class TestGptBigCode(ModelTest):
    NATIVE_MODEL_ID = "bigcode/gpt_bigcode-santacoder"

    def test_gptbigcode(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])