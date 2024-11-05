import torch
from model_test import ModelTest


class TestBloom(ModelTest):
    NATIVE_MODEL_ID = "bigscience/bloom-560m"

    def test_bloom(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        reference_output = "I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
