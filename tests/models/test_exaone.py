from model_test import ModelTest


class TestExaone(ModelTest):
    NATIVE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    prompt = [
        {"role": "system",
         "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
        {"role": "user",
         "content": "I am in Shanghai, preparing to visit the natural history museum. Can you tell me the best way to"}
    ]

    def test_exaone(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "Certainly! Here's how you can get to the Shanghai Natural History Museum:\n\n1. **By Metro**: The museum is located near Line 10 of the Shanghai Metro. You can take the Line 10 train to the People's Park station. From there, it's a short walk to the museum.\n\n2. **By Bus**: Several bus lines pass near the museum. For example, bus routes 10, 11,"

        result = self.generateChat(model, tokenizer, prompt=self.prompt)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
