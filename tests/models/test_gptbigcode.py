import torch
from model_test import ModelTest


class TestGptBigCode(ModelTest):
    NATIVE_MODEL_ID = "bigcode/gpt_bigcode-santacoder"

    def test_gptbigcode(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        reference_output = "I am in Paris and I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Berlin. I am in Ber"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
