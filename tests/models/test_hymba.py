from model_test import ModelTest


class TestHymba(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Hymba-1.5B-Instruct/"  # "baichuan-inc/Baichuan2-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.4104
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4317
    MODEL_MAX_LEN = 8192
    TRUST_REMOTE_CODE = True
    # Hymba currently only supports a batch size of 1.
    # See https://huggingface.co/nvidia/Hymba-1.5B-Instruct
    BATCH_SIZE = 1

    def test_hymba(self):
        self.quant_lm_eval()
