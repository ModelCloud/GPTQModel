import torch  # noqa: E402

from tests.model_test import ModelTest


class TestStarCode2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/starcoder2-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2901
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3080
    TORCH_DTYPE = torch.float16
    def test_starcode2(self):
        self.quant_lm_eval()


