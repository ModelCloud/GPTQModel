from model_test import ModelTest  # noqa: E402

# TODO, this model requires 24G vram at least
class TestGptNeoX(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt-neox-20b" # "EleutherAI/gpt-neox-20b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3805
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4078
    def test_gptneox(self):
        self.quant_lm_eval()

