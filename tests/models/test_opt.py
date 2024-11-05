from model_test import ModelTest # noqa: E402



class TestOpt(ModelTest):
    NATIVE_MODEL_ID = "facebook/opt-125m"
    NATIVE_ARC_CHALLENGE_ACC = 0.1894
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2278

    def test_opt(self):
        self.quant_lm_eval()

<<<<<<< HEAD
=======
        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
