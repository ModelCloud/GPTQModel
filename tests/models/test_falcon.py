from model_test import ModelTest

class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "tiiuae/falcon-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3993
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4292

    def test_falcon(self):
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
