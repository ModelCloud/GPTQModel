from model_test import ModelTest

class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "internlm/internlm-7b"

    def test_internlm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])