# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import concurrent.futures
import os
import sys
from typing import Dict, List

import pytest
import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import QuantizeConfig


def _dummy_prepare_dataset(*, calibration_dataset, calibration_dataset_concat_size, calibration_dataset_sort, batch_size):
    return calibration_dataset


class _DummyProgressBar:
    def title(self, _):
        return self

    def draw(self):
        return None


def _is_free_threaded() -> bool:
    gil_check = getattr(sys, "_is_gil_enabled", None)
    if callable(gil_check):
        return not gil_check()
    env_value = os.environ.get("PYTHON_GIL", "1").lower()
    return env_value in {"0", "false", "off"}


def _run_quant_on_device(device_index: int) -> torch.device:
    torch.cuda.set_device(device_index)
    target = torch.device(f"cuda:{device_index}")
    module = torch.nn.Linear(8, 8, bias=False).to(target)
    named = NamedModule(module, name=f"linear_{device_index}", full_name=f"model.layers.{device_index}.linear", layer_index=device_index)

    qcfg = QuantizeConfig(mock_quantization=True, group_size=-1, desc_act=False)
    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=_dummy_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        require_fwd=False,
        calculate_w_wq_diff=False,
    )
    processor.pb = _DummyProgressBar()

    processor.preprocess(named, fail_safe=False)
    named.module.target_device = target

    processor.process(named)

    return named.weight.data.device


#@pytest.mark.cuda
def test_gptq_quantize_keeps_weight_on_assigned_device_multigpu_free_thread():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for multi-GPU device context test")

    if torch.cuda.device_count() < 8:
        pytest.skip("Requires at least 8 CUDA devices")

    if sys.version_info < (3, 13):
        pytest.skip("Requires Python 3.13 free-threading build")

    if not _is_free_threaded():
        pytest.skip("Requires PYTHON_GIL=0 (free-threading)")

    device_indices: List[int] = list(range(8))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(device_indices)) as pool:
        futures = [pool.submit(_run_quant_on_device, idx) for idx in device_indices]
        results: Dict[int, torch.device] = {idx: future.result() for idx, future in zip(device_indices, futures)}

    for idx, device in results.items():
        assert device.type == "cuda" and device.index == idx
