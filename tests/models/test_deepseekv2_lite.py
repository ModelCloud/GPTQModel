from model_test import ModelTest

class TestDeepseekV2Lite(ModelTest):
    NATIVE_MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4753
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4855
    def test_deepseekv2lite(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        task_results = self.lm_eval(model)
        for filter, value in task_results.items():
            if "norm" in filter:
                per = (value / self.NATIVE_ARC_CHALLENGE_ACC_NORM) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
            else:
                per = (value / self.NATIVE_ARC_CHALLENGE_ACC) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")


