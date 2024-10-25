from model_test import ModelTest

class TestYi(ModelTest):
    NATIVE_MODEL_ID = "01-ai/Yi-Coder-1.5B-Chat"

    def test_yi(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, True)
        reference_output = "Certainly! Here's how you can get to the Shanghai Natural History Museum:\n\n1. **By Metro**: The museum is located near Line 10 of the Shanghai Metro. You can take the Line 10 train to the People's Park station. From there, it's a short walk to the museum.\n\n2. **By Bus**: Several bus lines pass near the museum. For example, bus routes 10, 11,"

        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])