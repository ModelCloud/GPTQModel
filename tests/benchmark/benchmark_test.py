# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import unittest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from logbar import LogBar

logger = LogBar.shared()

class BenchmarkTest(unittest.TestCase):
    MODEL_id = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
    MIN_NEW_TOKENS = 10
    MAX_NEW_TOKENS = 20
    NUM_RUNS = 50
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
    MAX_DELTA_FLOOR_PERCENT = 0.25
    MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0

    def benchmark(self, backend, device, tokens_per_second: int, warmup_iter: int = 1):
        model = GPTQModel.load(
            self.MODEL_id,
            device=device,
            backend=backend,
            use_cache=False,
        )

        model.optimize()

        tokenizer = model.tokenizer
        inp = tokenizer(self.PROMPTS, padding=True, padding_side="left", pad_to_multiple_of=16, truncation=True, return_tensors="pt",).to(device)

        print(f"Warming up: warmup_iter = `{warmup_iter}`")
        for i in range(warmup_iter):
            _ = model.generate(**inp, min_new_tokens=self.MIN_NEW_TOKENS,
                               max_new_tokens=self.MAX_NEW_TOKENS)

        times = []
        pb = logger.pb(range(self.NUM_RUNS)).title("Run")
        for _ in pb:
            start_time = time.time()
            _ = model.generate(**inp,min_new_tokens=self.MIN_NEW_TOKENS,
                                 max_new_tokens=self.MAX_NEW_TOKENS)
            end_time = time.time()

            elapsed_time = end_time - start_time
            times.append(elapsed_time)

        sum_time = sum(times)
        sum_tokens = len(self.PROMPTS) * self.MIN_NEW_TOKENS * self.NUM_RUNS
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
