from model_test import ModelTest

class TestGranite(ModelTest):
    NATIVE_MODEL_ID = "ibm-granite/granite-3.0-2b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4505
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4770

    def test_granite(self):
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

