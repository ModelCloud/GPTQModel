import tempfile

import torch
import unittest

from gptqmodel import GPTQModel, QuantizeConfig

import time
from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class Test_allow_liger_kernel(unittest.TestCase):
    def test_liger_kernel_memory_usage(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        calibration_dataset = [
            tokenizer(
                "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
        )

        # don't use liger_kernel
        torch.cuda.empty_cache()

        model = GPTQModel.from_pretrained(MODEL_ID, quantize_config)
        model.quantize(calibration_dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_quantized(tmpdir)

            del model

            _ = GPTQModel.from_quantized(tmpdir, device="cuda:0")

        memory_no_liger = torch.cuda.memory_allocated()

        # use liger_kernel
        torch.cuda.empty_cache()

        model = GPTQModel.from_pretrained(MODEL_ID, quantize_config, use_liger_kernel=True)
        model.quantize(calibration_dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_quantized(tmpdir)

            del model

            _ = GPTQModel.from_quantized(tmpdir, device="cuda:0")

        memory_use_liger = torch.cuda.memory_allocated()

        print(f'memory usage with memory_use_liger: {memory_use_liger / (1024 ** 3): .4f}GB, memory usage with memory_no_liger: {memory_no_liger / (1024 ** 3): .4f}GB')
        self.assertLess(memory_use_liger, memory_no_liger, f"memory_use_liger: {memory_use_liger}, memory_no_liger: {memory_no_liger}")


    def test_liger_kernel_cost_time(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        calibration_dataset = [
            tokenizer(
                "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
        )

        # don't use liger_kernel
        start_time = time.time()

        model = GPTQModel.from_pretrained(MODEL_ID, quantize_config)
        model.quantize(calibration_dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_quantized(tmpdir)

            del model

            _ = GPTQModel.from_quantized(tmpdir, device="cuda:0")

        duration_no_liger = time.time() - start_time

        # use liger_kernel
        start_time = time.time()

        model = GPTQModel.from_pretrained(MODEL_ID, quantize_config, use_liger_kernel=True)
        model.quantize(calibration_dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_quantized(tmpdir)

            del model

            _ = GPTQModel.from_quantized(tmpdir, device="cuda:0")

        duration_use_liger = time.time() - start_time

        print(f'time with duration_use_liger: {duration_use_liger}, time with duration_no_liger: {duration_no_liger}')
        self.assertLess(duration_use_liger, duration_no_liger, f'duration_use_liger: {duration_use_liger}, duration_no_liger: {duration_no_liger}')
