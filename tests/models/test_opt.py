from model_test import ModelTest

class TestOpt(ModelTest):
    NATIVE_MODEL_ID = "facebook/opt-125m"
    NATIVE_ARC_CHALLENGE_ACC = 0.1894
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2278

    def test_opt(self):
        self.model, self.tokenizer = self.quantModel(self.NATIVE_MODEL_ID)

        task_results = self.lm_eval(self.model)
        for filter, value in task_results.items():
            per = self.calculatorPer(filter=filter, value=value)
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")

