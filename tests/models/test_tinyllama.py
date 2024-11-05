from model_test import ModelTest

class TestTinyllama(ModelTest):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    NATIVE_ARC_CHALLENGE_ACC = 0.2995
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3268

    def test_tinyllama(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)

        task_results = self.lm_eval(model, trust_remote_code=True)
        for filter, value in task_results.items():
            per = self.calculatorPer(filter=filter, value=value)
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")
