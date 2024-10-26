from model_test import ModelTest

class TestDeci(ModelTest):
    NATIVE_MODEL_ID = "Deci/DeciLM-7B-instruct"

    def test_deci(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<s> I am in Paris and I am going to the Eiffel Tower.\n\nQuestion: Where is the narrator going?\n\nAnswer: The Eiffel Tower\n\nTitle: The Eiffel Tower\n\nBackground: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Construction began on 28 January 1887"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])