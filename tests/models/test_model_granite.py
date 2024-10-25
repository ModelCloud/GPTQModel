from model_test import ModelTest

class TestGranite(ModelTest):
    NATIVE_MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"

    def test_granite(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])