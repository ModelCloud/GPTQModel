# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from base_test import BaseTest

class TestExaone(BaseTest):
    NATIVE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    # QUANT_MODEL_ID = "ModelCloud/EXAONE-3.0-7.8B-Instruct-gptq-4bit"

    def test_exaone(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, True)
        reference_output = "Certainly! Here's how you can get to the Shanghai Natural History Museum:\n\n1. **By Metro**: The museum is located near Line 10 of the Shanghai Metro. You can take the Line 10 train to the People's Park station. From there, it's a short walk to the museum.\n\n2. **By Bus**: Several bus lines pass near the museum. For example, bus routes 10, 11,"
        prompt = [
            {"role": "system",
             "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
            {"role": "user",
             "content": "I am in Shanghai, preparing to visit the natural history museum. Can you tell me the best way to"}
        ]
        input_tensor = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_ids=input_tensor.to(model.device), max_new_tokens=self.GENERATE_EVAL_SIZE)
        result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        print(f"Result is: \n{result}")

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
