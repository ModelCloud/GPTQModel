from model_test import ModelTest

class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "stabilityai/stablelm-base-alpha-3b"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.0235
    NATIVE_GSM8k_STRICT_MATCH = 0.014

    def test_stablelm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        reference_output = "I am in Paris and I am looking for a place to stay. I have a dog, I am up to date on vaccinations and I am up to date on flea and worming. I am also going to be up to date on my shots, I am very confident, I do not have a problem when it comes to someones residence. I am looking for a woman that is confident, that does not feel like a last resort type of person. And if you feel like you would not love living with"
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
