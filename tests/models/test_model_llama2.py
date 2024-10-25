from model_test import ModelTest


class TestLlama2(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-2-7b-chat"

    def test_llama2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generateChat(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])