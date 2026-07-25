# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.models.loader import _setup_rotation_online_had
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import QuantizeConfig


def test_quantize_config_rotation_roundtrip():
    cfg = QuantizeConfig(bits=4, group_size=128, rotation="hadamard")
    d = cfg.to_dict()
    assert d.get("rotation") == "hadamard"

    cfg2 = QuantizeConfig.from_quant_config(d)
    assert cfg2.rotation == "hadamard"


def test_quantize_config_rotation_none_omitted():
    cfg = QuantizeConfig(bits=4, group_size=128)
    d = cfg.to_dict()
    assert "rotation" not in d


class _DummyQuantLinear(BaseQuantLinear):
    def __init__(self, in_features, out_features):
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = -1
        self.K = 1
        self.register_buffer("had_K", None, persistent=False)


class _DummyMlp(torch.nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.down_proj = _DummyQuantLinear(intermediate_size, hidden_size)


class _DummyLayer(torch.nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.mlp = _DummyMlp(intermediate_size, hidden_size)


class _DummyModel(torch.nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DummyLayer(intermediate_size, hidden_size)])


def test_setup_rotation_online_had_pow2_intermediate():
    model = _DummyModel(8192, 4096)
    _setup_rotation_online_had(model, "hadamard")

    down = model.layers[0].mlp.down_proj
    assert down.online_full_had is True
    assert down.K == 1
    assert down.had_K is None


def test_setup_rotation_online_had_non_pow2_intermediate():
    model = _DummyModel(11008, 4096)
    _setup_rotation_online_had(model, "hadamard")

    down = model.layers[0].mlp.down_proj
    assert down.online_full_had is True
    assert down.K == 172
    assert down.had_K is not None
    assert down.had_K.shape == (172, 172)


def test_apply_rotation_to_input_routes_to_full_and_partial():
    from gptqmodel.quantization.rotation import hadamard_utils

    calls = []

    def fake_matmul_hadU_cuda(X, hadK, K):
        calls.append((X.shape, K, hadK is not None))
        return X

    original = hadamard_utils.matmul_hadU_cuda
    hadamard_utils.matmul_hadU_cuda = fake_matmul_hadU_cuda

    try:
        x = torch.randn(2, 5, 8192)
        mod = _DummyQuantLinear(8192, 4096)
        mod.online_full_had = True
        y = mod._apply_rotation_to_input(x)
        assert len(calls) == 1
        assert calls[-1][0] == (2, 5, 8192)
        assert calls[-1][1] == 1
        assert calls[-1][2] is False
        assert y.shape == x.shape

        calls.clear()
        had_K = torch.eye(128)
        mod2 = _DummyQuantLinear(128, 64)
        mod2.online_partial_had = True
        mod2.had_dim = 128
        mod2.had_K = had_K
        x2 = torch.randn(2, 5, 128)
        y2 = mod2._apply_rotation_to_input(x2)
        assert len(calls) == 1
        assert calls[-1][0] == (10, 128)
        assert calls[-1][1] == 1
        assert calls[-1][2] is True
        assert y2.shape == x2.shape

        calls.clear()
        mod3 = _DummyQuantLinear(128, 64)
        y3 = mod3._apply_rotation_to_input(x2)
        assert len(calls) == 0
        assert y3 is x2
    finally:
        hadamard_utils.matmul_hadU_cuda = original
