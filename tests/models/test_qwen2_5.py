from model_test import ModelTest

class TestQwen2_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4343
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4676


    def test_qwen2_5(self):
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


