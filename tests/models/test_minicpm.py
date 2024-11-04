from model_test import ModelTest

class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "openbmb/MiniCPM-2B-128k"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.4314
    NATIVE_GSM8k_STRICT_MATCH = 0.4026

    def test_minicpm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<s> I am in Paris and I am looking for a place to stay. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the"
        result = self.generate(model, tokenizer)



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
