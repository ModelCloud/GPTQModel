from model_test import ModelTest


class TestInternlm2_VL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/InternVL2-8B-MPO"
    NATIVE_ARC_CHALLENGE_ACC = 0.3217
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3575
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 6
    USE_VLLM = False


    def test_internlm2_5(self):
        # transformers<=4.44.2 run normal
        model, tokenizer, processor = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                                      torch_dtype=self.TORCH_DTYPE, use_flash_attn=False)



