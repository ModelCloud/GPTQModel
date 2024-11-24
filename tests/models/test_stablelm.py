from tests.model_test import ModelTest  # noqa: E402


class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/stablelm-base-alpha-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2363
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2577
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 6

    def test_stablelm(self):
        self.quant_lm_eval()
