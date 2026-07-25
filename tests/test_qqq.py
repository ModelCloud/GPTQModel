# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import json
import logging
import os
import tempfile
import unittest

import torch

from datasets import load_dataset
from models.model_test import ModelTest
from parameterized import parameterized
from transformers import AutoTokenizer

from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
from gptqmodel.quantization import FORMAT, METHOD, QUANT_CONFIG_FILENAME
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.utils.torch import torch_empty_cache


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from logbar import LogBar

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402


log = LogBar.shared()

class TestGroupSize(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Llama-3.2-1B"
        #"/monster/data/model/Qwen2.5-0.5B-Instruct/" "/monster/data/model/Qwen2.5-0.5B-Instruct/" #

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_id, use_fast=True)

        traindata = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz", split="train")
        self.calibration_dataset = traindata.select(range(1024))

    def test_load_group_128(self):
        model = GPTQModel.load(
            "/monster/data/model/QQQ-Llama-3-8b-g128",
            device="cuda"
        )

        self.assert_qqq_linear(model)

        result = model.generate("Uncovering deep insights begins with")[0] # tokens
        log.info(f"Output: {model.tokenizer.decode(result)}") # string output

    # TODO FIXME: group_size 128 is failing this CI TEST!
    @parameterized.expand([-1, 128])
    def test_quant_and_inference(self, group_size: int):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=group_size,
            quant_method=METHOD.QQQ,
            format=FORMAT.QQQ,
        )

        model = GPTQModel.load(
            self.pretrained_model_id,
            quantize_config=quantize_config,
        )
        model.quantize(self.calibration_dataset, batch_size=1, calibration_concat_size=2048)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            model.save(tmp_dir_name)

            with open(tmp_dir_name + "/" + QUANT_CONFIG_FILENAME, "r") as f:
                file_dict = json.loads(f.read())

                # make sure the json dict saved to file matches config in memory
                assert model.quantize_config.to_dict() == file_dict
                logging.info(f"Saved config file: {file_dict}")

            del model
            torch_empty_cache()

            model = GPTQModel.load(
                tmp_dir_name,
                device="cuda"
            )

            self.assert_qqq_linear(model)

            result = ModelTest.generate_stable_with_limit(
                model,
                model.tokenizer,
                "The capital city of France is named",
                min_new_tokens=128,
                max_new_tokens=128,
            )
            print(f"BACKEND: {BACKEND.QQQ}, Result: {result}")
            if "paris" not in result.lower() and "city" not in result.lower() and "country" not in result.lower():
                raise AssertionError(" `paris` not found in `result`")

    def assert_qqq_linear(self, model):
        has_qqq = False
        for _, module in model.named_modules():
            linear = QQQLinear
            if isinstance(module, linear):
                has_qqq = True
                break
        self.assertTrue(has_qqq)


class TestQQQHessian(unittest.TestCase):
    """Verify QQQ Hessian accumulation matches the closed-form reference."""

    def _reference_hessian(self, *tensors):
        """Compute 2/N * (X^T X) for one or more (B, S, C) activation tensors."""
        target = tensors[0].device
        x = torch.cat([t.to(target) for t in tensors], dim=0).reshape(-1, tensors[0].shape[-1]).float()
        return (2.0 / x.shape[0]) * x.t().matmul(x)

    def test_single_batch_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32)
        qqq = QQQ(layer)
        x = torch.randn(4, 8, 16, dtype=torch.float32)
        qqq.add_batch(x, None)
        H = qqq.materialize_hessian()
        H_ref = self._reference_hessian(x)
        self.assertTrue(torch.allclose(H, H_ref, rtol=1e-4, atol=1e-5))

    def test_multiple_batches_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32)
        qqq = QQQ(layer)
        batches = [torch.randn(2, 8, 16, dtype=torch.float32) for _ in range(3)]
        for x in batches:
            qqq.add_batch(x, None)
        H = qqq.materialize_hessian()
        H_ref = self._reference_hessian(*batches)
        self.assertTrue(torch.allclose(H, H_ref, rtol=1e-4, atol=1e-5))

    def test_multi_device_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32, device="cpu")
        qqq = QQQ(layer)
        x_cpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cpu")
        x_gpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cuda:0")
        qqq.add_batch(x_cpu, None)
        qqq.add_batch(x_gpu, None)
        H = qqq.materialize_hessian()
        H_ref = self._reference_hessian(x_cpu, x_gpu)
        self.assertTrue(torch.allclose(H, H_ref, rtol=1e-4, atol=1e-5))

    def test_empty_batch_is_noop(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32)
        qqq = QQQ(layer)
        qqq.add_batch(torch.zeros(0, 8, 16, dtype=torch.float32), None)
        H = qqq.materialize_hessian()
        self.assertEqual(H.shape, (16, 16))
        self.assertTrue(torch.allclose(H, torch.zeros_like(H)))
