from model_test import ModelTest

class TestBaiChuan(ModelTest):
    NATIVE_MODEL_ID = "baichuan-inc/Baichuan2-7B-Chat"

    def test_baichuan(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = ""
        result = self.generateChat(model, tokenizer)

        self.assertTrue(len(result) > 0)
        # self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])