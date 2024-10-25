from model_test import ModelTest

class TestMpt(ModelTest):
    NATIVE_MODEL_ID = "mosaicml/mpt-1b-redpajama-200b"

    def test_mpt(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])