from model_test import ModelTest

class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "stabilityai/stablelm-2-1_6b"

    def test_stablelm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])