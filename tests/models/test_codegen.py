from ..model_test import ModelTest  # noqa: E402


class TestCodeGen(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/codegen2-1B_P" # "Salesforce/codegen2-1B_P"
    NATIVE_ARC_CHALLENGE_ACC = 0.1749
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2005
    TRUST_REMOTE_CODE = True
    USE_VLLM = False

    def test_codegen(self):
        self.quant_lm_eval()

