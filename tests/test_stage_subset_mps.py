# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

import gptqmodel.looper.stage_subset as stage_subset


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def submit(self, device, fn, *args, **kwargs):
        self.calls.append(("parallel", torch.device(device)))
        return fn(*args, **kwargs)

    def submit_serial(self, device, fn, *args, **kwargs):
        self.calls.append(("serial", torch.device(device)))
        return fn(*args, **kwargs)


def test_mps_module_quantization_uses_single_serial_command_stream(monkeypatch):
    pool = _RecordingPool()
    monkeypatch.setattr(stage_subset, "DEVICE_THREAD_POOL", pool)

    assert stage_subset._submit_quantization_task(torch.device("mps"), lambda: "mps") == "mps"
    assert stage_subset._submit_quantization_task(torch.device("cpu"), lambda: "cpu") == "cpu"

    assert pool.calls == [
        ("serial", torch.device("mps")),
        ("parallel", torch.device("cpu")),
    ]
