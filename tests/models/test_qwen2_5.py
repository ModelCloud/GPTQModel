from model_test import ModelTest


class TestQwen2_5(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


    def test_qwen2_5(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)

        reference_output = "I am in Paris and I have a problem with my phone. I need to charge it but I don't have a charger. I have a laptop charger that has the same voltage and amperage as my phone charger. Can I use the laptop charger to charge my phone? Using a laptop charger to charge your phone is generally not recommended, even if the voltage and amperage seem to match. Here are a few reasons why:\n\n1. **Connector Type**: The charger's connector (the part that plugs into your"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
