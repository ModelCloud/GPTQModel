from model_test import ModelTest

class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "CohereForAI/aya-expanse-8b"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.7657
    NATIVE_GSM8k_STRICT_MATCH = 0.7657
    def test_cohere(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "<BOS_TOKEN>I am in Paris and I am in love. I am in love with the city, the people, the food, the art, the history, the architecture, the fashion, the music, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art, the art,"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])

        task_results = self.lm_eval(model)
        for filter, value in task_results.items():
            if "flexible" in filter:
                per = (value / self.NATIVE_GSM8k_FLEXIBLE_EXTRACT) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
                #flexible-extract 0.1304
                self.assertLess(value, self.NATIVE_GSM8k_FLEXIBLE_EXTRACT)
            else:
                per = (value / self.NATIVE_GSM8k_STRICT_MATCH) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
                #strict-match 0.1251
                self.assertLess(value, self.NATIVE_GSM8k_STRICT_MATCH)