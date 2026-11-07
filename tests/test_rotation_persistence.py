# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.models.loader import _setup_rotation_online_had
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import QuantizeConfig


@pytest.mark.parametrize("use_definition_plan", [True, False])
def test_declared_norm_fusion_preserves_real_llama_forward(use_definition_plan):
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.models.module_plan import ModulePlan
    from gptqmodel.quantization.rotation.rotation import fuse_layer_norms

    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=1, vocab_size=64,
        tie_word_embeddings=False,
    )).double().eval()
    layer = model.model.layers[0]
    with torch.no_grad():
        for norm in (layer.input_layernorm, layer.post_attention_layernorm, model.model.norm):
            norm.weight.uniform_(0.5, 1.5)
        tokens = torch.tensor([[1, 4, 9, 12]])
        expected = model(tokens).logits.clone()
        plan = ModulePlan.from_tree(LlamaQModel.module_tree) if use_definition_plan else None
        fuse_layer_norms(model, "model.norm", "model.layers", "lm_head", plan=plan)
        actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for norm in (layer.input_layernorm, layer.post_attention_layernorm, model.model.norm):
        assert torch.equal(norm.weight, torch.ones_like(norm.weight))


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


def test_rotation_with_plan_ignores_undeclared_suffix_matches():
    from gptqmodel.models.module_plan import ModulePlan

    model = _DummyModel(128, 32)
    model.extra = torch.nn.Module()
    model.extra.mlp = _DummyMlp(128, 32)
    plan = ModulePlan.from_tree(["layers", "#", {"mlp": ("down_proj:down",)}])
    _setup_rotation_online_had(model, "hadamard", module_plan=plan, layer_prefixes=["layers"])
    assert model.layers[0].mlp.down_proj.online_full_had
    assert not model.extra.mlp.down_proj.online_full_had


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
    assert down.had_K.abs().min().item() > 0


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


def test_rotation_config_persisted_in_json(tmp_path):
    import json as _json

    cfg = QuantizeConfig(bits=4, group_size=128, rotation="hadamard")
    config_path = tmp_path / "quantize_config.json"
    with open(config_path, "w") as f:
        _json.dump(cfg.to_dict(), f)

    with open(config_path) as f:
        loaded = _json.load(f)

    assert loaded.get("rotation") == "hadamard"
    cfg2 = QuantizeConfig.from_quant_config(loaded)
    assert cfg2.rotation == "hadamard"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fast Hadamard transform")
def test_down_proj_online_hadamard_cancels_in_forward():
    """Fusing a full Hadamard into the weight and re-applying it to the activation
    must cancel, so the quantized forward matches the original (up to dtype noise).
    """
    from gptqmodel.quantization.rotation import hadamard_utils

    in_features, out_features = 4096, 2048
    x = torch.randn(2, 5, in_features, device="cuda", dtype=torch.float32)
    W = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)

    expected = x @ W.T

    # Fuse a full normalized Hadamard into the input dimension of W.
    W_fused = hadamard_utils.matmul_hadU_cuda(W, None, 1)

    # Simulate the online transform that _apply_rotation_to_input applies.
    x_rotated = hadamard_utils.matmul_hadU_cuda(x, None, 1)

    out = x_rotated @ W_fused.T
    assert torch.allclose(out, expected, atol=1e-3)


def test_rotation_loaded_from_quantized_down_proj_has_online_state():
    """If the rotated Llama 3.2 1B checkpoint is present, verify the saved
    rotation value was reloaded and mlp.down_proj carries the online state.
    """
    import os

    path = "/tmp/llama3_2_1b_gptq_rotated"
    if not os.path.isdir(path):
        pytest.skip("Rotated checkpoint not available on this runner")

    with open(os.path.join(path, "quantize_config.json")) as f:
        import json as _json

        raw = _json.load(f)

    assert raw.get("rotation") == "hadamard"
    cfg = QuantizeConfig.from_quant_config(raw)
    assert cfg.rotation == "hadamard"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fast Hadamard transform")
def test_hooked_linear_online_hadamard_cancels_in_forward():
    """HookedLinear must re-apply the online Hadamard during calibration/replay.

    Fusing ``H`` into ``mlp.down_proj`` and applying ``H`` to the activation
    before ``super().forward`` must cancel, matching the original dense matmul.
    """
    from gptqmodel.nn_modules.hooked_linear import HookedLinear
    from gptqmodel.quantization.rotation import hadamard_utils

    in_features, out_features = 4096, 2048
    x = torch.randn(2, 5, in_features, device="cuda", dtype=torch.float32)
    W = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)

    expected = x @ W.T

    # Fuse a full normalized Hadamard into the input dimension of W.
    W_fused = hadamard_utils.matmul_hadU_cuda(W, None, 1)

    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32, device="cuda")
    linear.weight.data = W_fused
    linear.online_full_had = True
    linear.online_partial_had = False
    linear.had_dim = -1
    linear.had_K = None
    linear.K = 1

    hl = HookedLinear.from_linear(linear)
    out = hl(x)
    assert torch.allclose(out, expected, atol=1e-2)

    # The forward hook should observe the Hadamard-transformed input, not the raw one.
    seen_inputs = []
    def capture_hook(module, inp, out):
        seen_inputs.append(inp[0])

    hl.forward_hook = capture_hook
    _ = hl(x)
    assert len(seen_inputs) == 1
    assert torch.allclose(seen_inputs[0], hadamard_utils.matmul_hadU_cuda(x, None, 1), atol=1e-2)
