from model_test import ModelTest

class TestPhi_3(ModelTest):
    NATIVE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

    def test_phi_3(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and I want to visit the Eiffel Tower. Can you provide me with directions?\n\nAssistant: Of course! To reach the Eiffel Tower from Paris, you can follow these directions:\n\n1. Start by taking the metro from your current location to the nearest metro station.\n2. Take the Metro Line 6 from your starting point to the Charles de Gaulle - Étoile station.\n3. Once you arrive at Charles de Gaulle - É"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])