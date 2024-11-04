from model_test import ModelTest

class TestMistral(ModelTest):
    NATIVE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    NATIVE_ARC_CHALLENGE_ACC = 0.5427
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5597

    def test_mistral(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)

        task_results = self.lm_eval(model, trust_remote_code=True)
        for filter, value in task_results.items():
            if "norm" in filter:
                per = (value / self.NATIVE_ARC_CHALLENGE_ACC_NORM) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
            else:
                per = (value / self.NATIVE_ARC_CHALLENGE_ACC) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")
