from model_test import ModelTest


class TestBaiChuan(ModelTest):
    NATIVE_MODEL_ID = "baichuan-inc/Baichuan2-7B-Chat"

    def test_baichuan(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)

        reference_output = "I am in Paris and I need to go to the airport. How can I get to the airport from here?\nThere are several ways to get to the airport from Paris. The most common way is to take the RER (Regional Express Train). You can take the RER A line from Gare de l'Est or Gare du Nord stations. The other option is to take the Métro (subway). You can take the Métro Line 1 or Line 14 to"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
