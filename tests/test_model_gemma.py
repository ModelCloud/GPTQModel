from base_test import BaseTest

class TestGemma(BaseTest):
    NATIVE_MODEL_ID = "google/gemma-2-9b"

    def test_gemma(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "<bos>I am in Paris and I am going to the Louvre. I am going to see the Mona Lisa. I am going to see the Venus de Milo. I am going to see the Winged Victory of Samothrace. I am going to see the Coronation of Napoleon. I am going to see the Raft of the Medusa. I am going to see the Code of Hammurabi. I am going to see the Rosetta Stone. I am going to see the Venus de Milo. I am going to see the Winged"
        result = self.generate(model, tokenizer)
        print(f"Result is: \n{result}")

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
