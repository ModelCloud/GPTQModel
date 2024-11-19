import subprocess
import sys

from gptqmodel.models import XverseGPTQ
from model_test import ModelTest  # noqa: E402


class TestXVerse(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/XVERSE-7B-Chat" # "xverse/XVERSE-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.4198
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4044
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = True
    BATCH_SIZE = 6
    USE_VLLM = False

    @classmethod
    def setUpClass(cls):
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"transformers{XverseGPTQ.require_transformers_version}"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers==0.15.2"])

    def test_xverse(self):
        self.quant_lm_eval()

    @classmethod
    def tearDownClass(cls):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "transformers"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "tokenizers"])
