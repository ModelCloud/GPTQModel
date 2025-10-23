# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import time

from gptqmodel.utils.torch import torch_empty_cache


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import unittest

from logbar import LogBar
from transformers import AutoTokenizer

from gptqmodel import GPTQModel


logger = LogBar.shared()

class InferenceSpeed(unittest.TestCase):
    NATIVE_MODEL_ID = "/monster/data/model/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
    BITBLAS_NATIVE_MODEL_ID = "/monster/data/model/opt-125M-autoround-lm_head-false-symTrue"
    MAX_NEW_TOKENS = 10
    NUM_RUNS = 20
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
    MAX_POSITIVE_DELTA_CEIL_PERCENT = 0.25

    def inference(self, model_path, backend, tokens_per_second, assert_result=True, optimize=False, fullgraph=False, warmup_runs=0, device=None):
        model = GPTQModel.from_quantized(
            model_path,
            backend=backend,
            device=device,
        )

        if optimize:
            model.optimize(fullgraph=fullgraph)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inp = tokenizer(self.PROMPTS, padding=True, truncation=True, return_tensors="pt", padding_side='left').to(
            model.device)

        times = []
        tokens = []

        # compile kernels need JIT compile (Bitblas, IPEX, Triton) so we should do some warmup before actual speed run
        if warmup_runs > 0:
            pb = logger.pb(range(warmup_runs)).title("Warmup")
            for _ in pb:
                start_time = time.time()
                result = model.generate(**inp, max_new_tokens=self.MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
                end_time = time.time()
                elapsed_time = end_time - start_time
                times.append(elapsed_time)

                for j in range(result.shape[0]):
                    new_tokens = result[j][inp['input_ids'].shape[1]:]
                    new_token_count = len(new_tokens)
                    tokens.append(new_token_count)

            sum_time = sum(times)
            sum_tokens = sum(tokens)

            avg_tokens_per_second = round(sum_tokens / sum_time, 2)

            print(f"\n**************** {backend} Warm-up Result Info****************")
            print(f"Times: {times}")
            print(f"New Tokens (Size Per Batch Request): {tokens}")
            print(f"Sum Times: {sum_time}")
            print(f"Sum New Tokens: {sum_tokens}")
            print(f"New Token Per Second: {avg_tokens_per_second} token/s")
            print(f"****************  {backend} Warm-up Result Info End****************")

        pb = logger.pb(range(self.NUM_RUNS)).title("Run")
        for _ in pb:
            start_time = time.time()
            result = model.generate(**inp, max_new_tokens=self.MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)

            for j in range(result.shape[0]):
                new_tokens = result[j][inp['input_ids'].shape[1]:]
                new_token_count = len(new_tokens)
                tokens.append(new_token_count)

        sum_time = sum(times)
        sum_tokens = sum(tokens)

        avg_tokens_per_second = round(sum_tokens / sum_time, 2)

        print(f"\n**************** {backend} Result Info****************")
        print(f"Times: {times}")
        print(f"New Tokens: {tokens}")
        print(f"Sum Times: {sum_time}")
        print(f"Sum New Tokens: {sum_tokens}")
        print(f"New Token Per Second: {avg_tokens_per_second} token/s")
        print(f"****************  {backend} Result Info End****************")

        if not assert_result:
            return

        diff_pct = (avg_tokens_per_second / tokens_per_second) * 100
        negative_pct = 100 * (1 - self.MAX_DELTA_FLOOR_PERCENT)
        positive_pct = 100 * (1 + self.MAX_POSITIVE_DELTA_CEIL_PERCENT)

        self.assertTrue(negative_pct <= diff_pct <= positive_pct,
                        f"{backend}: Actual tokens Per Second: {avg_tokens_per_second}, expected = `{tokens_per_second}` diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")

        del model
        torch_empty_cache()
