from model_test import ModelTest

class TestLlama3_2(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    def test_llama3_2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        # model, tokenizer = self.loadQuantModel(f"/monster/data/pzs/quantization/c4data/{self.NATIVE_MODEL_ID}", True,
        #                                        tokenizer_path=self.NATIVE_MODEL_ID)

        reference_output = "<|begin_of_text|>I am in Paris and I am planning to visit the Eiffel Tower. I am not a fan of heights, but I am willing to take the stairs or elevator to get to the top. I am looking for a fun and unique experience to take with me as a souvenir.\n\nHere are a few ideas for souvenirs that might fit the bill:\n\n1. A beautiful piece of French art or sculpture\n2. A delicious French pastry or dessert\n3. A charming Parisian postcard or print"
        result = self.generate(model, tokenizer)
        self.lm_eval(model, trust_remote_code=True)