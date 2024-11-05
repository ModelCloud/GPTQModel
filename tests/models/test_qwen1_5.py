from model_test import ModelTest


class TestQwen1_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen1.5-0.5B"

    def test_qwen1_5(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and I am looking for a place to stay. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center."
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
