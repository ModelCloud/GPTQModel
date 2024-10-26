from model_test import ModelTest

class TestInternlm2_5(ModelTest):
    NATIVE_MODEL_ID = "internlm/internlm2_5-1_8b-chat"

    def test_internlm2_5(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<|begin_of_text|>I am in Paris and I am so excited to be here! The city of love, art, fashion, and food. I have been dreaming of visiting Paris for years, and now that I am finally here, I am determined to make the most of my time. I have a list of all the must-see sights, but I also want to explore the city like a local. I want to discover the hidden gems, the secret spots, and the authentic experiences that only a local would know about."
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
