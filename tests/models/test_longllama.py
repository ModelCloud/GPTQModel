from model_test import ModelTest


class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "syzymon/long_llama_3b_instruct"

    def test_longllama(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<s> I am in Paris andP\n\nP\n\nP\n\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\n"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
