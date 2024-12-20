import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import unittest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.utils.progress import ProgressBar


class BenchmarkTest(unittest.TestCase):
    MODEL_id = "/monster/data/model/gptq_4bits_11-21_15-47-09_maxlen2048_ns2048_descFalse_damp0.1"
    MIN_NEW_TOEKNS = 100
    NUM_RUNS = 10
    PROMPTS = [
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
    MAX_DELTA_FLOOR_PERCENT = 0.15
    MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0

    def benchmark(self, backend, device, tokens_per_second):
        model = GPTQModel.from_quantized(
            self.MODEL_id,
            device=device,
            backend=backend,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_id)
        tokenizer.pad_token = tokenizer.eos_token
        inp = tokenizer(self.PROMPTS, padding=True, truncation=True, return_tensors="pt", padding_side='left').to(device)

        times = []
        pb = ProgressBar(range(self.NUM_RUNS))
        for i in pb:
            pb.set_description(f"run index {i} of {self.NUM_RUNS -1}")
            start_time = time.time()
            res = model.generate(**inp, num_beams=1, min_new_tokens=self.MIN_NEW_TOEKNS,
                                 max_new_tokens=self.MIN_NEW_TOEKNS)
            end_time = time.time()

            elapsed_time = end_time - start_time
            times.append(elapsed_time)

        sum_time = sum(times)
        sum_tokens = len(self.PROMPTS) * self.MIN_NEW_TOEKNS * self.NUM_RUNS
        avg_tokens_per_second = sum_tokens / sum_time

        print("**************** Benchmark Result Info****************")
        print(f"Times: {times}")
        print(f"Sum Times: {sum_time}")
        print(f"Sum New Tokens: {sum_tokens}")
        print(f"Benchmark Result: {avg_tokens_per_second} token/s")
        print("**************** Benchmark Result Info End****************")

        diff_pct = (avg_tokens_per_second / tokens_per_second) * 100
        negative_pct = 100 * (1 - self.MAX_DELTA_FLOOR_PERCENT)
        positive_pct = 100 * (1 + self.MAX_POSITIVE_DELTA_CEIL_PERCENT)

        self.assertTrue(negative_pct <= diff_pct <= positive_pct,
                        f"Tokens Per Second: {avg_tokens_per_second} diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")