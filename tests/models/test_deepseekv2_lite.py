from model_test import ModelTest

class TestDeepseekV2Lite(ModelTest):
    NATIVE_MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

    def test_deepseekv2lite(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<｜begin▁of▁sentence｜>I am in Paris and I am looking for a good place to eat. I am a vegetarian and I am looking for a place that has a good vegetarian menu. I am not looking for a fancy restaurant, just a good place to eat.\nI am looking for a place that has a good vegetarian menu and is not too expensive. I am not looking for a fancy restaurant, just a good place to eat.\nI am in Paris and I am looking for a good place to eat. I am a vegetarian and"
        result = self.generate(model, tokenizer)


