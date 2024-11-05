from model_test import ModelTest


class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "stabilityai/stablelm-base-alpha-3b"

    def test_stablelm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        reference_output = "I am in Paris and I am looking for a place to stay. I have a dog, I am up to date on vaccinations and I am up to date on flea and worming. I am also going to be up to date on my shots, I am very confident, I do not have a problem when it comes to someones residence. I am looking for a woman that is confident, that does not feel like a last resort type of person. And if you feel like you would not love living with"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
