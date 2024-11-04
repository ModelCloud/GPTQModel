from model_test import ModelTest

class TestQwen1_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.2055
    NATIVE_GSM8k_STRICT_MATCH = 0.1418

    def test_qwen1_5(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and I am looking for a place to stay. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center."
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])

        task_results = self.lm_eval(model, trust_remote_code=True)
        print(f"task_results---{task_results}")
        for filter, value in task_results.items():
            if "flexible" in filter:
                per = (value / self.NATIVE_GSM8k_FLEXIBLE_EXTRACT) * 100
                print(f"{filter}: {value} improve {per:.2f}")
                self.assertGreater(value, self.NATIVE_GSM8k_FLEXIBLE_EXTRACT)
            else:
                per = (value / self.NATIVE_GSM8k_STRICT_MATCH) * 100
                print(f"{filter}: {value} improve {per:.2f}")
                self.assertGreater(value, self.NATIVE_GSM8k_STRICT_MATCH)
