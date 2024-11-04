from model_test import ModelTest

class TestLlama2(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
    NATIVE_ARC_CHALLENGE_ACC = 0.3490
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3635
    def test_llama2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
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

