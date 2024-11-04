from model_test import ModelTest

class TestMistral(ModelTest):
    NATIVE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

    def test_mistral(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "<s> I am in Paris and I am looking for a good restaurant for a special occasion. I have heard that Le Jules Verne is a great option, but I am wondering if there are any other restaurants in Paris that would be worth considering for a special occasion?\n\nThere are indeed several other restaurants in Paris that are known for their exceptional dining experiences and would be worth considering for a special occasion. Here are a few suggestions:\n\n1. L'Arpege: This three-Michelin-star"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])