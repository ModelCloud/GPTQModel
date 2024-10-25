from model_test import ModelTest

class TestQwen(ModelTest):
    QUANT_MODEL_ID = "LnL-AI/qwen2.5-14b-instruct-trained-G4-gptq-4bit-9-19-2024"


    def test_qwen(self):
        model, tokenizer = self.loadQuantModel(self.QUANT_MODEL_ID)

        reference_output = "I am in Paris and I have a problem with my phone. I need to charge it but I don't have a charger. I have a laptop charger that has the same voltage and amperage as my phone charger. Can I use the laptop charger to charge my phone? Using a laptop charger to charge your phone is generally not recommended, even if the voltage and amperage seem to match. Here are a few reasons why:\n\n1. **Connector Type**: The charger's connector (the part that plugs into your"
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
