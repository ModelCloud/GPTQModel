import torch
from model_test import ModelTest  # noqa: E402


class TestBloom(ModelTest):
    NATIVE_MODEL_ID = "bigscience/bloom-560m"
    NATIVE_ARC_CHALLENGE_ACC = 0.2201
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2440
    TORCH_DTYPE = torch.float16

    def test_bloom(self):
        self.quant_lm_eval()

