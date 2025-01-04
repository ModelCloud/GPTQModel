import transformers
from model_test import ModelTest
from packaging.version import Version


class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/MiniCPM-2B-128k"  # "openbmb/MiniCPM-2B-128k"
    NATIVE_ARC_CHALLENGE_ACC = 0.3848
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 4

    def test_minicpm(self):
        args = {}
        # if flash_attn was installed and _attn_implementation_autoset was None, flash attention would be loaded
        # but device map is cpu, it will trow non-supported device error
        if Version(transformers.__version__) >= Version("4.46.0"):
            args["_attn_implementation_autoset"] = True

        self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE, **args)
