from model_test import ModelTest


class TestXVerse(ModelTest):
    NATIVE_MODEL_ID = "xverse/XVERSE-7B-Chat"

    def test_xverse(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "You can get there by subway. Take Line 2 to People's Square, then transfer to Line 10 and get off at the Shanghai Museum station. It's about a 20-minute ride.<|im_end|>\n<|im_start|>user\nThanks for the information. Anything else you think I should know?<|im_end"
        result = self.generateChat(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
