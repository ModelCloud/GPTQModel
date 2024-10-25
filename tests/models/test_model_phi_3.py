from model_test import ModelTest

class TestPhi_3(ModelTest):
    NATIVE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

    def test_phi_3(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])