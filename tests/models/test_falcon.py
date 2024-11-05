from model_test import ModelTest # noqa: E402



class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "tiiuae/falcon-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3993
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4292
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_falcon(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and,.....\n,,,,,,,, ,,, and and,, ,, and and and,, ,, and and, and, and, and, and, and, and, and and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and the, and"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
