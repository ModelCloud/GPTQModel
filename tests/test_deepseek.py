# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from base_test import BaseTest

class TestDeepseek(BaseTest):
    NATIVE_MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    QUANT_MODEL_ID = "ModelCloud/DeepSeek-V2-Lite-gptq-4bit"

    def test_deepseek(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, True)
        reference_output = "<｜begin▁of▁sentence｜>I am in Paris and I am looking for a good place to eat. I am a vegetarian and I am looking for a place that has a good vegetarian menu. I am not looking for a fancy restaurant, just a good place to eat.\nI am looking for a place that has a good vegetarian menu and is not too expensive. I am not looking for a fancy restaurant, just a good place to eat.\nI am in Paris and I am looking for a good place to eat. I am a vegetarian and"
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
