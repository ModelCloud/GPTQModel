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


class _DatasetCompatCase(ModelTest):
    __test__ = False
    DATASET_SIZE = 512
    DATASET_SIZE_FAST = 128
    DATASET_CONCAT_SIZE = 2048
    DATASET_CONCAT_SIZE_FAST = 1024
    OFFLOAD_TO_DISK = True
    OFFLOAD_TO_DISK_FAST = False

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


def test_model_test_fast_mode_uses_fast_dataset_overrides(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_MODEL_TEST_MODE", "fast")
    case = _DatasetCompatCase(methodName="runTest")

    assert case._mode_specific_test_setting("DATASET_SIZE") == 128
    assert case._mode_specific_test_setting("DATASET_CONCAT_SIZE") == 1024
    assert case._mode_specific_test_setting("OFFLOAD_TO_DISK") is False


def test_model_test_slow_mode_uses_default_dataset_settings(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_MODEL_TEST_MODE", "slow")
    case = _DatasetCompatCase(methodName="runTest")

    assert case._mode_specific_test_setting("DATASET_SIZE") == 512
    assert case._mode_specific_test_setting("DATASET_CONCAT_SIZE") == 2048
    assert case._mode_specific_test_setting("OFFLOAD_TO_DISK") is True
