from model_test import ModelTest  # noqa: E402


class TestExaone(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/EXAONE-3.0-7.8B-Instruct" # "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.46
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.46
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 6

    def test_exaone(self):
        self.quant_lm_eval()


