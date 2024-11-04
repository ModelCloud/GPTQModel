from model_test import ModelTest

class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "stabilityai/stablelm-base-alpha-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2363
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2577

    def test_stablelm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

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
