# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from gptqmodel import _build_device_thread_pool
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.env import env_flag
from gptqmodel.utils.logger import QuantizationRegionTimer, log_time_block, setup_logger
from gptqmodel.utils.threadx import DeviceThreadPool


class TestQuantTelemetry(unittest.TestCase):
    def test_env_flag_truthiness(self):
        with patch.dict(os.environ, {"GPTQMODEL_TEST_FLAG": "1"}, clear=False):
            self.assertTrue(env_flag("GPTQMODEL_TEST_FLAG"))
        with patch.dict(os.environ, {"GPTQMODEL_TEST_FLAG": "0"}, clear=False):
            self.assertFalse(env_flag("GPTQMODEL_TEST_FLAG"))

    def test_device_thread_pool_cpu_workers_capped_on_free_threading(self):
        with patch.object(DeviceThreadPool, "__init__", return_value=None) as mock_init:
            with patch("gptqmodel.has_gil_disabled", return_value=True):
                _build_device_thread_pool()
        workers = mock_init.call_args.kwargs["workers"]
        cpu = workers["cpu"]
        self.assertGreaterEqual(cpu, 2)
        self.assertLessEqual(cpu, 8)
        self.assertEqual(workers["model_loader:cpu"], 2)

    def test_device_thread_pool_cpu_workers_keep_historical_default_on_gil(self):
        with patch.object(DeviceThreadPool, "__init__", return_value=None) as mock_init:
            with patch("gptqmodel.has_gil_disabled", return_value=False):
                _build_device_thread_pool()
        workers = mock_init.call_args.kwargs["workers"]
        cpu = workers["cpu"]
        self.assertGreaterEqual(cpu, 1)
        self.assertLessEqual(cpu, 12)
        self.assertEqual(workers["model_loader:cpu"], 2)

    def test_log_time_block_is_silent_by_default(self):
        with patch.dict(
            os.environ,
            {"DEBUG": "0", "GPTQMODEL_LOG_TIMES": "0"},
            clear=False,
        ):
            logger = setup_logger()
            with patch.object(logger, "info") as mock_info:
                with log_time_block("silent", logger=logger, module_name="m"):
                    pass
        mock_info.assert_not_called()

    def test_log_time_block_emits_when_env_flag_set(self):
        with patch.dict(os.environ, {"GPTQMODEL_LOG_TIMES": "1"}, clear=False):
            logger = setup_logger()
            with patch.object(logger, "info") as mock_info:
                with log_time_block("visible", logger=logger, module_name="m"):
                    pass
        mock_info.assert_called_once()
        self.assertIn("[time] visible (m) took", mock_info.call_args[0][0])

    def test_hessian_inverse_records_region_timer(self):
        module = nn.Linear(4, 4, bias=False, dtype=torch.float64)
        H = torch.eye(4, dtype=torch.float64) * 2
        qcfg = QuantizeConfig(bits=4, group_size=4, damp_percent=0.01)
        timer = QuantizationRegionTimer()
        gptq = GPTQ(module, qcfg=qcfg, region_timer=timer)
        gptq.name = "test.linear"

        _, damp = gptq.hessian_inverse(H)
        self.assertEqual(damp, 0.01)

        stats = timer.snapshot()["hessian_inverse"]
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["source"], "test.linear")
        self.assertGreater(stats["total"], 0.0)
