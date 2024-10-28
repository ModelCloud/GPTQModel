from model_test import ModelTest

class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "internlm/internlm-7b"

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        reference_output = " <s>I am in Paris and I am in love with the city. I am in love with the people. I am in love with the food. I am in love with the art. I am in love with the architecture. I am in love with the fashion. I am in love with the language. I am in love with the history. I am in love with the culture. I am in love with the romance. I am in love with the city.\nI am in love with the city. I am in love"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])