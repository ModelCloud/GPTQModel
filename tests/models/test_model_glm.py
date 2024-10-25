from model_test import ModelTest

class TestGlm(ModelTest):
    NATIVE_MODEL_ID = "THUDM/glm-4-9b-chat"
    QUANT_MODEL_ID = "ModelCloud/glm-4-9b-chat-gptq-4bit"

    def test_glm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, True)
        # model, tokenizer = self.loadQuantModel(self.QUANT_MODEL_ID, True)
        reference_output = "<|begin_of_text|>I am in Paris and I am so excited to be here! The city of love, art, fashion, and food. I have been dreaming of visiting Paris for years, and now that I am finally here, I am determined to make the most of my time. I have a list of all the must-see sights, but I also want to explore the city like a local. I want to discover the hidden gems, the secret spots, and the authentic experiences that only a local would know about."
        result = self.generateChat(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
