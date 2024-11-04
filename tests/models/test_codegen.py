from model_test import ModelTest

class TestCodeGen(ModelTest):
    NATIVE_MODEL_ID = "Salesforce/codegen2-1B_P"

    def test_codegen(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])