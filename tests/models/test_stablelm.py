from model_test import ModelTest # noqa: E402



class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "stabilityai/stablelm-base-alpha-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2363
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2577
    TRUST_REMOTE_CODE = True

    def test_stablelm(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        reference_output = "I am in Paris and I am looking for a place to stay. I have a dog, I am up to date on vaccinations and I am up to date on flea and worming. I am also going to be up to date on my shots, I am very confident, I do not have a problem when it comes to someones residence. I am looking for a woman that is confident, that does not feel like a last resort type of person. And if you feel like you would not love living with"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
