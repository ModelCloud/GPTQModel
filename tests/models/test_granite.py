from model_test import ModelTest

class TestGranite(ModelTest):
    NATIVE_MODEL_ID = "ibm-granite/granite-3.0-2b-instruct"

    def test_granite(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and<fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix><fim_prefix>"
        result = self.generate(model, tokenizer)

