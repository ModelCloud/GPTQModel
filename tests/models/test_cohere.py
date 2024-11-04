from model_test import ModelTest

class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "CohereForAI/aya-expanse-8b"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.7657
    NATIVE_GSM8k_STRICT_MATCH = 0.7657
    def test_cohere(self):
        # model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        model, tokenizer = self.loadQuantModel(f"/monster/data/pzs/quantization/{self.NATIVE_MODEL_ID}",
                                               trust_remote_code=True)
        reference_output = "<BOS_TOKEN>I am in Paris and I am in love. I am in love with the city, the people, the food, the art, the history, the architecture, the fashion, the music, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art,"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])

        task_results = self.lm_eval(model)
        for filter, value in task_results.items():
            if "flexible" in filter:
                per = (value / self.NATIVE_GSM8k_FLEXIBLE_EXTRACT) * 100
                print(f"{filter}: {value} improve {per:.2f}")
                self.assertGreater(value, self.NATIVE_GSM8k_FLEXIBLE_EXTRACT)
            else:
                per = (value / self.NATIVE_GSM8k_STRICT_MATCH) * 100
                print(f"{filter}: {value} improve {per:.2f}")
                self.assertGreater(value, self.NATIVE_GSM8k_STRICT_MATCH)