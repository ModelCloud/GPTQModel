import torch  # noqa: E402

from tests.model_test import ModelTest  # noqa: E402


class TestGptJ(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt-j-6b" # "EleutherAI/gpt-j-6b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3396
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3660
    TORCH_DTYPE = torch.float16
    INPUTS_MAX_LENGTH = 1024

    def test_gptj(self):
        self.quant_lm_eval()

