from model_test import ModelTest

class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "tiiuae/falcon-mamba-tiny-dev"

    def test_falcon(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])