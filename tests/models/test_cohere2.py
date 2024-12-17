from model_test import ModelTest


class TestCohere2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/c4ai-command-r7b-12-2024"
    NATIVE_ARC_CHALLENGE_ACC = 0.4680
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4693
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.15
    BATCH_SIZE = 4

    def test_cohere2(self):
        self.quant_lm_eval()
