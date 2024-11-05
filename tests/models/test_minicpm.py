from model_test import ModelTest # noqa: E402



class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "openbmb/MiniCPM-2B-128k"
    NATIVE_ARC_CHALLENGE_ACC = 0.3848
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = True

    def test_minicpm(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<s> I am in Paris and I am looking for a place to stay. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
