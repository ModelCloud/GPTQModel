from model_test import ModelTest

class TestInternlm2_5(ModelTest):
    NATIVE_MODEL_ID = "internlm/internlm2_5-1_8b-chat"

    def test_internlm2_5(self):
        # transformers<=4.44.2 run normal
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=True)
        reference_output = "<s>I am in Paris and I want to visit the Eiffel Tower. I have a 3 day trip to Paris. I want to visit the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. I have a budget of $100 per day. I can stay in hotels or hostels. I can also use public transportation.I have a good idea of how to get around Paris. I will stay in a hotel or hostel for the first two nights. I will stay in a hotel"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])
