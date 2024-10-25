from model_test import ModelTest

class TestMoss(ModelTest):
    NATIVE_MODEL_ID = "fnlp/moss2-2_5b-chat"

    def test_moss(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])