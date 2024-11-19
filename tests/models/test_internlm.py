import subprocess
import sys

from model_test import ModelTest  # noqa: E402


class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm-7b" # "internlm/internlm-7b"
    NATIVE_ARC_CHALLENGE_ACC = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4309
    TRUST_REMOTE_CODE = True
    USE_VLLM = False

    @classmethod
    def setUpClass(cls):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.44.2"])

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()

    @classmethod
    def tearDownClass(cls):
        subprocess.check_call([sys.executable, "-m", "pip", "-U", "install", "transformers"])