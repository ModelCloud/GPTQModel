from model_test import ModelTest

class TestQwen1_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen1.5-0.5B"

    def test_qwen1_5(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])