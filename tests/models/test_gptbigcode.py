from model_test import ModelTest
import torch
class TestGptBigCode(ModelTest):
    NATIVE_MODEL_ID = "bigcode/gpt_bigcode-santacoder"
    NATIVE_ARC_CHALLENGE_ACC = 0.1689
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2056

    def test_gptbigcode(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True)
        task_results = self.lm_eval(model, trust_remote_code=True)

        for filter, value in task_results.items():
            per = self.calculatorPer(filter=filter, value=value)
            self.assertTrue(90 <= per <= 110,
                            f"{filter}: {value} diff {per:.2f}% is out of the expected range (90%-110%)")

