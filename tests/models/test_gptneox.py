from model_test import ModelTest

class TestGptNeoX(ModelTest):
    NATIVE_MODEL_ID = "EleutherAI/gpt-neox-20b"

    def test_gptneox(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to the Louvre. I am going to"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])