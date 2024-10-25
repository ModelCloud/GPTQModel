from model_test import ModelTest

class TestPhi_1(ModelTest):
    NATIVE_MODEL_ID = "microsoft/phi-1"

    def test_phi_1(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])