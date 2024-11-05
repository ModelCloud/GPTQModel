from model_test import ModelTest # noqa: E402



class TestQwen1_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
    NATIVE_ARC_CHALLENGE_ACC = 0.2568
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2918
    TRUST_REMOTE_CODE = True

    def test_qwen1_5(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and I am looking for a place to stay. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center. I am looking for a place to stay in the city center."
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
