from model_test import ModelTest

class TestMixtral(ModelTest):
    NATIVE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    NATIVE_ARC_CHALLENGE_ACC = 0.5213
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5247
    def test_mixtral(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        task_results = self.lm_eval(model)
        for filter, value in task_results.items():
            per = self.calculatorPer(filter=filter, value=value)
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")
