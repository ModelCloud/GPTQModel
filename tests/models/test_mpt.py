from model_test import ModelTest

class TestMpt(ModelTest):
    NATIVE_MODEL_ID = "mosaicml/mpt-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4275
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4454

    def test_mpt(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        task_results = self.lm_eval(model, trust_remote_code=True)
        for filter, value in task_results.items():
            per = self.calculatorPer(filter=filter, value=value)
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")

