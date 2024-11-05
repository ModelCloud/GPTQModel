from model_test import ModelTest # noqa: E402


class TestMoss(ModelTest):
    NATIVE_MODEL_ID = "fnlp/moss2-2_5b-chat"
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_moss(self):
        self.quant_lm_eval()
