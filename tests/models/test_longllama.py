from model_test import ModelTest

class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "syzymon/long_llama_3b_instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3515
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3652

    def test_longllama(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        task_results = self.lm_eval(model, trust_remote_code=True)
        for filter, value in task_results.items():
            if "norm" in filter:
                # 0.2287
                per = (value / self.NATIVE_ARC_CHALLENGE_ACC_NORM) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
            else:
                # 0.1911
                per = (value / self.NATIVE_ARC_CHALLENGE_ACC) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")
