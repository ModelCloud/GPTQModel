from model_test import ModelTest

class TestGpt2(ModelTest):
    NATIVE_MODEL_ID = "openai-community/gpt2"
    NATIVE_ARC_CHALLENGE_ACC = 0.2270
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2270

    def test_gpt2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)

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

