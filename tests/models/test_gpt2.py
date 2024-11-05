import torch  # noqa: E402
from model_test import ModelTest  # noqa: E402


class TestGpt2(ModelTest):
    NATIVE_MODEL_ID = "openai-community/gpt2"
    NATIVE_ARC_CHALLENGE_ACC = 0.1903
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2270
    TORCH_DTYPE = torch.float16
    TRUST_REMOTE_CODE = True

    def test_gpt2(self):
        self.quant_lm_eval()

