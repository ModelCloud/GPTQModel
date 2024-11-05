from model_test import ModelTest

class TestGptNeoX(ModelTest):
    NATIVE_MODEL_ID = "EleutherAI/gpt-neox-20b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3805
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4078
    def test_gptneox(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        task_results = self.lm_eval(model)
        for filter, value in task_results.items():
            per = self.calculatorPer(filter=filter, value=value)
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")
