from model_test import ModelTest

class TestXVerse(ModelTest):
    NATIVE_MODEL_ID = "xverse/XVERSE-7B-Chat"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.0053
    NATIVE_GSM8k_STRICT_MATCH = 0.0008

    def test_xverse(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "You can get there by subway. Take Line 2 to People's Square, then transfer to Line 10 and get off at the Shanghai Museum station. It's about a 20-minute ride.<|im_end|>\n<|im_start|>user\nThanks for the information. Anything else you think I should know?<|im_end"
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
