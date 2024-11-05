import torch
from model_test import ModelTest


class TestStarCode2(ModelTest):
    NATIVE_MODEL_ID = "bigcode/starcoder2-3b"

    def test_starcode2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        reference_output = "I am in Paris and I am going to visit you.\n\nI am in Paris and I am going to visit you.\n\nI am in Paris and I am going to visit you.\n\nI am in Paris and I am going to visit you.\n\nI am in Paris and I am going to visit you.\n\nI am in Paris and I am going to visit you.\n\nI am in Paris and I am going to visit you.\n\nI"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
