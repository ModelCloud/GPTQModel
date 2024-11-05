from model_test import ModelTest

class TestDeci(ModelTest):
    NATIVE_MODEL_ID = "Deci/DeciLM-7B-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.5239
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5222
    def test_deci(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        task_results = self.lm_eval(model)
        for filter, value in task_results.items():
            per = self.calculatorPer(filter=filter, value=value)
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")