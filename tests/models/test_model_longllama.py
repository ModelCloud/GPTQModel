from model_test import ModelTest

class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "syzymon/long_llama_3b_instruct"

    def test_longllama(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])