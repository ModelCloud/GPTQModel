from model_test import ModelTest


class TestMpt(ModelTest):
    NATIVE_MODEL_ID = "mosaicml/mpt-7b-instruct"

    def test_mpt(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "I am in Paris and I am going to the Louvre.\nI am in Paris and I am going to the Louvre. I am going to see the Mona Lisa.\nI am in Paris and I am going to the Louvre. I am going to see the Mona Lisa. I am going to see the Mona Lisa.\nI am in Paris and I am going to the Louvre. I am going to see the Mona Lisa. I am going to see the Mona Lisa."
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
