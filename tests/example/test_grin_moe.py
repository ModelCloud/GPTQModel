# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from base_test import BaseTest

class TestGrinMoE(BaseTest):

    QUANT_MODEL_ID = "ModelCloud/GRIN-MoE-gptq-4bit"
    prompt = [
        {"role": "system",
         "content": "You are GRIN-MoE model from microsoft, a helpful assistant."},
        {"role": "user",
         "content": "I am in Shanghai, preparing to visit the natural history museum. Can you tell me the best way to"}
    ]

    def test_grinMoE_quant(self):
        model, tokenizer = self.loadQuantModel(self.QUANT_MODEL_ID)

        reference_output = "To get to the Shanghai Natural History Museum from your current location in Shanghai, you can consider the following options:\n\n1. Public Transportation:\nThe Shanghai Natural History Museum is located at 188 Lujiabang Rd, Huangpu District, Shanghai. The most convenient way to reach the museum is by using Shanghai's extensive public transportation system.\n\na. Metro: Take Line 1 (Red Line) and get off at"
        input_tensor = tokenizer.apply_chat_template(self.prompt, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_ids=input_tensor.to(model.device), max_new_tokens=100)
        result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

        print(result)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
