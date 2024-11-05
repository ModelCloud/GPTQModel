from model_test import ModelTest # noqa: E402



class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "syzymon/long_llama_3b_instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3515
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3652
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_longllama(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<s> I am in Paris andP\n\nP\n\nP\n\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\nP\n"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
