from model_test import ModelTest # noqa: E402



class TestMixtral(ModelTest):
    NATIVE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    NATIVE_ARC_CHALLENGE_ACC = 0.5213
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5247
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_mixtral(self):
<<<<<<< HEAD
        self.quant_lm_eval()
=======
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "<s> I am in Paris and I am in love.\n\nI am in love with the city, the people, the food, the language, the architecture, the history, the culture, the fashion, the art, the music, the wine, the cheese, the bread, the pastries, the cafes, the parks, the gardens, the bridges, the streets, the metro, the Eiffel Tower, the Louvre, the Notre Dame, the Sacre Coeur, the Arc"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
>>>>>>> main
