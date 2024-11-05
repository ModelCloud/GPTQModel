from model_test import ModelTest # noqa: E402


class TestCodeGen(ModelTest):
    NATIVE_MODEL_ID = "Salesforce/codegen2-1B_P"
    NATIVE_ARC_CHALLENGE_ACC = 0.1749
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2005
    TRUST_REMOTE_CODE = True

    def test_codegen(self):
        self.quant_lm_eval()

