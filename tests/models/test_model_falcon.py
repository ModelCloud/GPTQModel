from model_test import ModelTest

class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "tiiuae/falcon-7b-instruct"

    def test_falcon(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and,.....\n,,,,,,,, ,,, and and,, ,, and and and,, ,, and and, and, and, and, and, and, and, and and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and the, and"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])