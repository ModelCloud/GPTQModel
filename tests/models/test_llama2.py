from model_test import ModelTest

class TestLlama2(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
    NATIVE_ARC_CHALLENGE_ACC = 0.3490
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3635
    APPLY_CHAT_TEMPLATE = True
    QUANT_ARC_MAX_POSITIVE_DELTA = 23

    def test_llama2(self):
        self.quant_lm_eval()

