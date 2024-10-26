from model_test import ModelTest

class TestYi(ModelTest):
    NATIVE_MODEL_ID = "01-ai/Yi-Coder-1.5B-Chat"

    def test_yi(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "Sure, I can guide you on how to get there, get there's a lot of public transport in Shanghai.\n\n1. From the Out of Towns area, you can take the Express Train 101. It's a 10-minute ride from the Out of Towns area to the National Museum.\n\n2. From the Out of Towns area, you can also take the Express Train 102. It's a 10"

        result = self.generateChat(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])