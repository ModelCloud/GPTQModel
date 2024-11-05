import torch
<<<<<<< HEAD
from model_test import ModelTest # noqa: E402
=======
from model_test import ModelTest
>>>>>>> main


class TestBloom(ModelTest):
    NATIVE_MODEL_ID = "bigscience/bloom-560m"
    NATIVE_ARC_CHALLENGE_ACC = 0.2201
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2440
    TORCH_DTYPE = torch.float16

    def test_bloom(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        reference_output = "I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
