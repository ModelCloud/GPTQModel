import torch  # noqa: E402from tests.model_test import ModelTest


class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/falcon-7b-instruct" # "tiiuae/falcon-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3993
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4292
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    TORCH_DTYPE = torch.float16
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.52
    BATCH_SIZE = 6
    USE_VLLM = False

    def test_falcon(self):
        self.quant_lm_eval()
