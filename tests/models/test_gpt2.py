import torch
from model_test import ModelTest


class TestGpt2(ModelTest):
    NATIVE_MODEL_ID = "openai-community/gpt2"

    def test_gpt2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        reference_output = "I am in Paris and I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
