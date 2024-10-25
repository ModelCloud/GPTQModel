from model_test import ModelTest

class TestXVerse(ModelTest):
    NATIVE_MODEL_ID = "xverse/XVERSE-MoE-A4.2B-Chat"

    def test_xverse(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generateChat(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])