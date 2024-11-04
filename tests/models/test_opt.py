from model_test import ModelTest

class TestOpt(ModelTest):
    NATIVE_MODEL_ID = "facebook/opt-125m"

    def test_opt(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "</s>I am in Paris and I have a friend who is in the same city. I am in the same city as her and I have a friend who is in the same city. I am in the same city as her and I have a friend who is in the same city. I am in the same city as her and I have a friend who is in the same city. I am in the same city as her and I have a friend who is in the same city. I am in the same city as her and"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])