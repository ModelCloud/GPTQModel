import torch
<<<<<<< HEAD
from model_test import ModelTest # noqa: E402
=======
from model_test import ModelTest
>>>>>>> main


class TestStarCode2(ModelTest):
    NATIVE_MODEL_ID = "bigcode/starcoder2-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2901
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3080
    TORCH_DTYPE = torch.float16
    def test_starcode2(self):
        self.quant_lm_eval()

<<<<<<< HEAD
=======
        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
