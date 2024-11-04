from model_test import ModelTest
import torch
class TestBloom(ModelTest):
    NATIVE_MODEL_ID = "bigscience/bloom-560m"
    NATIVE_GSM8k_FLEXIBLE_EXTRACT = 0.0167
    NATIVE_GSM8k_STRICT_MATCH = 0.0144

    def test_bloom(self):
        # model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, torch_dtype=torch.float16)
        model, tokenizer = self.loadQuantModel(f"/monster/data/pzs/quantization/{self.NATIVE_MODEL_ID}",
                                               trust_remote_code=True)
        reference_output = "I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and I am in Paris. I am in Paris and"
        result = self.generate(model, tokenizer)

        self.assertEqual(result[:self.GENERATE_EVAL_SIZE], reference_output[:self.GENERATE_EVAL_SIZE])

        task_results = self.lm_eval(model, apply_chat_template=False)
        for filter, value in task_results.items():
            if "flexible" in filter:
                per = (value / self.NATIVE_GSM8k_FLEXIBLE_EXTRACT) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
                self.assertGreater(value, self.NATIVE_GSM8k_FLEXIBLE_EXTRACT)
            else:
                per = (value / self.NATIVE_GSM8k_STRICT_MATCH) * 100
                print(f"{filter}: {value} diff {per:.2f}%")
                self.assertGreater(value, self.NATIVE_GSM8k_STRICT_MATCH)