from model_test import ModelTest

class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "tiiuae/falcon-7b-instruct"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.0538
    NATIVE_GSM8k_STRICT_MATCH = 0.047

    def test_falcon(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and,.....\n,,,,,,,, ,,, and and,, ,, and and and,, ,, and and, and, and, and, and, and, and, and and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and the, and"
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
