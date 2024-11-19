from model_test import ModelTest

class TestGptNeoX(ModelTest):
    # TODO: this model requires 24G vram at least.
    NATIVE_MODEL_ID = "/monster/data/model/gpt-neox-20b"

    def test_gptneox(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])