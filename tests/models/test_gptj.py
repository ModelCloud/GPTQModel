import torch
<<<<<<< HEAD
from model_test import ModelTest # noqa: E402
=======
from model_test import ModelTest
>>>>>>> main


class TestGptJ(ModelTest):
    NATIVE_MODEL_ID = "EleutherAI/gpt-j-6b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3396
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3660
    TORCH_DTYPE = torch.float16

    def test_gptj(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        reference_output = ""
        result = self.generate(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
