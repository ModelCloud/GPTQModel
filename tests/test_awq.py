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
from parameterized import parameterized
from transformers import AutoTokenizer

from gptqmodel.nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear
from gptqmodel.nn_modules.qlinear.awq_gemv import AwqGEMVQuantLinear
from gptqmodel.nn_modules.qlinear.awq_gemv_fast import AwqGEMVFastQuantLinear
from gptqmodel.nn_modules.qlinear.awq_machete import AwqMacheteQuantLinear
from gptqmodel.nn_modules.qlinear.awq_marlin import AwqMarlinQuantLinear
from gptqmodel.quantization import FORMAT, METHOD, QUANT_CONFIG_FILENAME
from gptqmodel.utils.machete import _validate_machete_device_support, machete_import_exception


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from logbar import LogBar

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402


log = LogBar.shared()


class TestGroupSize(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pretrained_model_id = "/monster/data/model/Llama-3.2-1B"
        # "/monster/data/model/Qwen2.5-0.5B-Instruct/" "/monster/data/model/Qwen2.5-0.5B-Instruct/" #

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_id, use_fast=True)

        requested_samples = os.getenv("GPTQMODEL_AWQ_CALIB_SAMPLES")
        if requested_samples is not None:
            sample_count = max(1, int(requested_samples))
        else:
            total_mem_gb = 0
            if torch.cuda.is_available():
                try:
                    total_mem_gb = (
                        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                        / (1024 ** 3)
                    )
                except Exception:
                    total_mem_gb = 0

            if total_mem_gb >= 80:
                sample_count = 1024
            elif total_mem_gb >= 48:
                sample_count = 512
            else:
                sample_count = 192

        traindata = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz",
                                 split="train")
        cls.calibration_dataset = traindata.select(range(sample_count))

        cls.quantized_tempdirs = {}
        cls.quantized_model_paths = {}
        cls.quantize_config_dicts = {}

        quantize_targets = {
            (FORMAT.GEMM, 128),
            (FORMAT.GEMV, 128),
            (FORMAT.GEMV_FAST, 128),
        }

        for checkpoint_format, group_size in quantize_targets:
            quantize_config = QuantizeConfig(
                bits=4,
                group_size=group_size,
                quant_method=METHOD.AWQ,
                format=checkpoint_format,
            )

            model = GPTQModel.load(
                cls.pretrained_model_id,
                quantize_config=quantize_config,
            )
            model.quantize(cls.calibration_dataset, batch_size=1, calibration_concat_size=0)

            tmp_dir = tempfile.TemporaryDirectory()
            tmp_dir_name = tmp_dir.name
            model.save(tmp_dir_name)

            with open(tmp_dir_name + "/" + QUANT_CONFIG_FILENAME, "r") as f:
                file_dict = json.loads(f.read())
                assert model.quantize_config.to_dict() == file_dict
                logging.info(f"Saved config file: {file_dict}")

            cls.quantized_tempdirs[(checkpoint_format, group_size)] = tmp_dir
            cls.quantized_model_paths[(checkpoint_format, group_size)] = tmp_dir_name
            cls.quantize_config_dicts[(checkpoint_format, group_size)] = file_dict

            del model
            # torch_empty_cache()

    @classmethod
    def tearDownClass(cls):
        for tmp_dir in getattr(cls, "quantized_tempdirs", {}).values():
            tmp_dir.cleanup()
        # torch_empty_cache()

    # def test_load_group_128(self):
    #     model = GPTQModel.load(
    #         "/monster/data/model/AWQ-Llama-3-8b-g128",
    #     )
    #
    #     self.assert_awq_linear(model)
    #
    #     result = model.generate("Uncovering deep insights begins with")[0] # tokens
    #     log.info(f"Output: {model.tokenizer.decode(result)}") # string output

    @parameterized.expand([
        (FORMAT.GEMM, BACKEND.GEMM, 128),
        (FORMAT.GEMM, BACKEND.MACHETE, 128),
        (FORMAT.GEMM, BACKEND.MARLIN, 128),
        (FORMAT.GEMV, BACKEND.GEMV, 128),
        (FORMAT.GEMV_FAST, BACKEND.GEMV_FAST, 128),
    ])
    def test_quant_and_inference(self, checkpoint_format, backend, group_size: int):
        if backend == BACKEND.MACHETE:
            if machete_import_exception is not None:
                self.skipTest(f"machete unavailable: {machete_import_exception}")
            if not _validate_machete_device_support():
                self.skipTest("Machete requires NVIDIA Hopper or newer (SM90+)")

        key = (checkpoint_format, group_size)
        model_path = self.quantized_model_paths[key]
        expected_config = self.quantize_config_dicts[key]

        model = GPTQModel.load(
            model_path,
            backend=backend,
        )

        self.assertEqual(model.quantize_config.to_dict(), expected_config)

        self.assert_awq_linear(model, backend)

        tokens = model.generate("Capital of France is", max_new_tokens=100)[0]
        result = model.tokenizer.decode(tokens)
        print(f"BACKEND: {backend}, Result: {result}")
        if "paris" not in result.lower() and "city" not in result.lower():
            raise AssertionError(" `paris` not found in `result`")

        del model
        # torch_empty_cache()

    def assert_awq_linear(self, model, backend):
        has_qqq = False
        for _, module in model.named_modules():
            if backend == BACKEND.GEMM:
                linear = AwqGEMMQuantLinear
            elif backend == BACKEND.MACHETE:
                linear = AwqMacheteQuantLinear
            elif backend == BACKEND.MARLIN:
                linear = AwqMarlinQuantLinear
            elif backend == BACKEND.GEMV:
                linear = AwqGEMVQuantLinear
            elif backend == BACKEND.GEMV_FAST:
                linear = AwqGEMVFastQuantLinear
            else:
                raise Exception("unknown backend: " + backend)
            if isinstance(module, linear):
                has_qqq = True
                break
        self.assertTrue(has_qqq)
