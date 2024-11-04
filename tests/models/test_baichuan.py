from model_test import ModelTest

class TestBaiChuan(ModelTest):
    NATIVE_MODEL_ID = "baichuan-inc/Baichuan2-7B-Chat"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.3298
    NATIVE_GSM8k_STRICT_MATCH = 0.2532

    def test_baichuan(self):
        # model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        model, tokenizer = self.loadQuantModel(f"/monster/data/pzs/quantization/{self.NATIVE_MODEL_ID}", trust_remote_code=True)

        reference_output = "I am in Paris and I need to go to the airport. How can I get to the airport from here?\nThere are several ways to get to the airport from Paris. The most common way is to take the RER (Regional Express Train). You can take the RER A line from Gare de l'Est or Gare du Nord stations. The other option is to take the Métro (subway). You can take the Métro Line 1 or Line 14 to"
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
