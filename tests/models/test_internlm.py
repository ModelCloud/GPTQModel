from model_test import ModelTest # noqa: E402



class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "internlm/internlm-7b"
    NATIVE_ARC_CHALLENGE_ACC = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4309
    TRUST_REMOTE_CODE = True

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()

<<<<<<< HEAD
=======
        reference_output = " <s>I am in Paris and I am in love with the city. I am in love with the people. I am in love with the food. I am in love with the art. I am in love with the architecture. I am in love with the fashion. I am in love with the language. I am in love with the history. I am in love with the culture. I am in love with the romance. I am in love with the city.\nI am in love with the city. I am in love"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
