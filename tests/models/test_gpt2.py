from model_test import ModelTest
import torch
class TestGpt2(ModelTest):
    NATIVE_MODEL_ID = "openai-community/gpt2"

    def test_gpt2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        reference_output = "I am in Paris and I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to be in Paris. I am going to"
        result = self.generate(model, tokenizer)

