# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch.nn as nn
from model_test import ModelTest


class _DummyCompatCase(ModelTest):
    __test__ = False
    MODEL_COMPAT_FAST_LAYER_COUNT = 2

    def runTest(self):
        return None


class _FakeQuantModel:
    def __init__(self, layer_count: int):
        self.model = SimpleNamespace(layers=nn.ModuleList([nn.Linear(1, 1) for _ in range(layer_count)]))
        self.quantize_config = SimpleNamespace(dynamic=None)

    @staticmethod
    def extract_layers_node() -> str:
        return "layers"


def test_model_test_fast_mode_defaults_to_last_layers(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_FAST_LAYER_POSITION", raising=False)
    case = _DummyCompatCase(methodName="runTest")
    model = _FakeQuantModel(layer_count=6)

    with case.model_compat_test_context():
        dynamic = case._build_fast_model_compat_dynamic(model)

    assert sorted(dynamic) == [
        "-:^layers\\.0\\.",
        "-:^layers\\.1\\.",
        "-:^layers\\.2\\.",
        "-:^layers\\.3\\.",
    ]


def test_model_test_fast_mode_first_layers_remain_configurable(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_FAST_LAYER_POSITION", "first")
    case = _DummyCompatCase(methodName="runTest")
    model = _FakeQuantModel(layer_count=6)

    with case.model_compat_test_context():
        dynamic = case._build_fast_model_compat_dynamic(model)

    assert sorted(dynamic) == [
        "-:^layers\\.2\\.",
        "-:^layers\\.3\\.",
        "-:^layers\\.4\\.",
        "-:^layers\\.5\\.",
    ]
