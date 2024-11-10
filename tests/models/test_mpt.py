from model_test import ModelTest  # noqa: E402


class TestMpt(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/mpt-7b-instruct" # "mosaicml/mpt-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4275
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4454
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = False
    BATCH_SIZE = 6

    def test_mpt(self):
        self.quant_lm_eval()
