from model_test import ModelTest

class TestDeci(ModelTest):
    NATIVE_MODEL_ID = "Deci/DeciLM-7B-instruct"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.5125
    NATIVE_GSM8k_STRICT_MATCH = 0.5011
    def test_deci(self):
        # model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        model, tokenizer = self.loadQuantModel(f"/monster/data/pzs/quantization/{self.NATIVE_MODEL_ID}",
                                               trust_remote_code=True)
        reference_output = "<s> I am in Paris and I am going to the Eiffel Tower.\n\nQuestion: Where is the narrator going?\n\nAnswer: The Eiffel Tower\n\nTitle: The Eiffel Tower\n\nBackground: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Construction began on 28 January 1887"
        result = self.generate(model, tokenizer)



        task_results = self.lm_eval(model)
        for filter, value in task_results.items():
            if "flexible" in filter:
                per = (value / self.NATIVE_GSM8k_FLEXIBLE_EXTRACT) * 100
                print(f"{filter}: {value} improve {per:.2f}")
                self.assertGreater(value, self.NATIVE_GSM8k_FLEXIBLE_EXTRACT)
            else:
                per = (value / self.NATIVE_GSM8k_STRICT_MATCH) * 100
                print(f"{filter}: {value} improve {per:.2f}")
                self.assertGreater(value, self.NATIVE_GSM8k_STRICT_MATCH)