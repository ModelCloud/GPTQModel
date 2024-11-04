from model_test import ModelTest

class TestTinyllama(ModelTest):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.0083
    NATIVE_GSM8k_STRICT_MATCH = 0

    def test_tinyllama(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "Sure, here are some ways to visit the Natural History Museum in Shanghai:\n\n1. Public Transportation: Shanghai has a well-connected public transportation system, including subway lines, buses, and taxis. You can take the subway line 10 to the museum, or take a bus from the subway station to the museum.\n\n2. Taxi: If you prefer to take a taxi, you can call a taxi service in"
        result = self.generateChat(model, tokenizer)



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
