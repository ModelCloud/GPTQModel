# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from base_test import BaseTest

class TestLlama3(BaseTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    # QUANT_MODEL_ID = "ModelCloud/Meta-Llama-3.1-8B-gptq-4bit"

    def test_llama3(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, True)
        # model, tokenizer = self.loadQuantModel(self.QUANT_MODEL_ID)
        reference_output = "<|begin_of_text|>I am in Paris and I am looking for a job in the field of international relations. I am a French citizen, 24 years old, and I have a master degree in international relations. I am fluent in English and I am looking for a job in the field of international relations. I am available for a job in"
        result = self.generate(model, tokenizer)
        print(result)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
