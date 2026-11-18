# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from gptqmodel.looper.paro_processor import ParoProcessor


def test_parallel_activation_capture_uses_calibration_order():
    processor = object.__new__(ParoProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor.tasks = {}
    processor._ensure_task_bucket("proj", 0)

    def capture(index):
        processor._set_current_batch_index(index)
        processor._record_input_feature("proj", torch.full((1, 2, 3), float(index)))

    # Force a later calibration batch to complete first, then race the others.
    capture(7)
    with ThreadPoolExecutor(max_workers=7) as pool:
        list(pool.map(capture, reversed(range(7))))
    state = SimpleNamespace(modules={"proj": None})
    actual = processor._layer_input_features(state)["proj"]
    expected = torch.arange(8).view(8, 1, 1).expand(8, 2, 3)
    # Captures are joined on the row axis, so the batch axis collapses to 1
    # while every row keeps its calibration order (#5198).
    assert actual.shape == (1, 16, 3)
    assert torch.equal(actual.reshape(8, 2, 3), expected)
    assert torch.equal(processor._layer_input_features(state)["proj"], actual)
    processor._ensure_task_bucket("proj", 1)
    assert processor.tasks["proj"]["inputs"] == []
