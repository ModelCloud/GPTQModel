import torch.nn as nn
from model_test import ModelTest


class _TypeA(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)


class _TypeB(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 8)
        self.up = nn.Linear(8, 8)


class _TypeC(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(8, 4)
        self.experts = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])


def _build_model_test() -> ModelTest:
    test_case = ModelTest(methodName="runTest")
    test_case.MODEL_COMPAT_FAST_LAYER_COUNT = 2
    return test_case


def test_fast_model_layer_limit_uses_top_two_for_uniform_layers():
    test_case = _build_model_test()
    layers = nn.ModuleList([_TypeA() for _ in range(8)])

    assert test_case._fast_model_layer_limit(layers) == 2


def test_fast_model_layer_limit_covers_all_unique_layer_types_with_minimum_prefix():
    test_case = _build_model_test()
    layers = nn.ModuleList([_TypeA(), _TypeA(), _TypeB(), _TypeB(), _TypeC(), _TypeC(), _TypeC()])

    assert test_case._fast_model_layer_limit(layers) == 5
