from model_test import ModelTest


class TestMoss(ModelTest):
    NATIVE_MODEL_ID = "fnlp/moss2-2_5b-chat"

    def test_moss(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
