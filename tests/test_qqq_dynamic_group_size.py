# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gptqmodel.looper import qqq_processor as qqq_processor_module
from gptqmodel.looper.qqq_processor import QQQProcessor
from gptqmodel.quantization.config import QQQConfig


def test_qqq_dynamic_group_size_configures_the_cloned_quantizer(monkeypatch):
    class FakeQuantizer:
        def __init__(self):
            self.args = None

        def configure(self, *args, **kwargs):
            self.args = (args, kwargs)

    class FakeQQQ:
        def __init__(self, *, module, qcfg):
            del module
            self.qcfg = qcfg
            self.quantizer = FakeQuantizer()
            self.fallback = None
            self.expected_nsamples = None

    monkeypatch.setattr(qqq_processor_module, "QQQ", FakeQQQ)
    processor = object.__new__(QQQProcessor)
    processor.qcfg = QQQConfig(
        bits=4,
        group_size=-1,
        dynamic={"proj": {"group_size": 128}},
        offload_to_disk=False,
    )
    processor.tasks = {}
    processor.total_calibration_tokens = 16

    processor.preprocess(SimpleNamespace(full_name="proj", name="proj"))
    task = processor.tasks["proj"]

    assert task.qcfg.group_size == 128
    args, kwargs = task.quantizer.args
    assert args[0] == 4
    assert kwargs["groupsize"] == 128
    assert kwargs["sym"] == task.qcfg.sym
