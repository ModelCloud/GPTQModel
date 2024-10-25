from model_test import ModelTest

class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "CohereForAI/aya-expanse-8b"

    def test_cohere(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "<BOS_TOKEN>I am in Paris and I am in love. I am in love with the city, the people, the food, the art, the history, the architecture, the fashion, the music, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art,"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])