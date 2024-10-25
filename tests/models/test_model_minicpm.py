from model_test import ModelTest

class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "openbmb/MiniCPM-2B-128k"

    def test_minicpm(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, True)
        reference_output = "<s> I am in Paris and I am looking for a place to stay. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the Eiffel Tower. I am looking for a place that is close to the"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])