import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import unittest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestsTorchBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_id = "/monster/data/model/gptq_4bit_07-29_06-46-12_maxlen2048_ns1024_descFalse_damp0.005/"
        self.device = 'cuda'
        self.min_new_tokens = 100
        self.prompts = [
            "I am in Paris and I",
            "The capital of the United Kingdom is",
            "The largest ocean on Earth is",
            "The world’s longest river is",
            "The tallest mountain in the world is",
            "The currency used in Japan is",
            "How to consult a dictionary?",
            "What is the boiling point of water in degrees Celsius?",
            "Which is the most widely used Internet search engine in the world?",
            "What is the official language of France?",
        ]
        self.num_runs = 10
        self.tokens_per_second = 49.20

    def test_torch_benchmark(self):
        model = GPTQModel.from_quantized(
            self.model_id,
            device=self.device,
            backend=BACKEND.TORCH,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inp = tokenizer(self.prompts, padding=True, truncation=True, return_tensors="pt", padding_side='left').to(self.device)

        times = []
        for _ in range(self.num_runs):
            start_time = time.time()
            res = model.generate(**inp, num_beams=1, min_new_tokens=self.min_new_tokens,
                                 max_new_tokens=self.min_new_tokens)
            end_time = time.time()

            elapsed_time = end_time - start_time
            times.append(elapsed_time)

        sum_time = sum(times)
        sum_tokens = len(self.prompts) * self.min_new_tokens * self.num_runs
        avg_tokens_per_second = sum_tokens / sum_time

        print("**************** Benchmark Result Info****************")
        print(f"Times: {times}")
        print(f"Sum Times: {sum_time}")
        print(f"Sum New Tokens: {sum_tokens}")
        print(f"Benchmark Result: {avg_tokens_per_second} token/s")
        print("**************** Benchmark Result Info End****************")

        self.assertTrue(avg_tokens_per_second > self.tokens_per_second, f"Average tokens per second {avg_tokens_per_second} is not greater than {self.tokens_per_second}.")










