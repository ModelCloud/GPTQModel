from model_test import ModelTest

class TestTinyllama(ModelTest):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_tinyllama(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "Sure, here are some ways to visit the Natural History Museum in Shanghai:\n\n1. Public Transportation: Shanghai has a well-connected public transportation system, including subway lines, buses, and taxis. You can take the subway line 10 to the museum, or take a bus from the subway station to the museum.\n\n2. Taxi: If you prefer to take a taxi, you can call a taxi service in"
        result = self.generateChat(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
