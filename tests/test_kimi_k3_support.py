# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the native checkpoint and bounded K3 calibration runner."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from gptqmodel.models import auto
from gptqmodel.models.definitions.kimi_k3 import KimiK3QModel, KimiLinearQModel
from gptqmodel.quantization.dtype import dequantize_f4_e2m1


@pytest.fixture(scope="module")
def utils():
    # A separate feature worktree can test a utility being written concurrently.
    path = os.environ.get("GPTQMODEL_KIMI_K3_UTILS")
    if path:
        spec = importlib.util.spec_from_file_location("_kimi_k3_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("gptqmodel.utils.kimi_k3")


@pytest.fixture(scope="module")
def reference(utils):
    path = Path(os.environ.get(
        "KIMI_K3_SNAPSHOT",
        str(Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
            "f831ab66814297da540d832a5235f8e904f29d06"),
    ))
    if not (path / "modeling_kimi_linear.py").is_file():
        pytest.skip("Kimi K3 checkpoint reference code is not cached; set KIMI_K3_SNAPSHOT")
    return utils.load_calibration_reference(path)


@pytest.mark.parametrize("model_type,definition", [
    ("kimi_k3", KimiK3QModel), ("kimi_linear", KimiLinearQModel),
])
def test_registry_selects_kimi_backbone(monkeypatch, model_type, definition):
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda *a, **kw: True)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *a, **kw:
                        SimpleNamespace(model_type=model_type))
    assert auto.check_and_get_model_definition("tiny-kimi", trust_remote_code=True) is definition


@pytest.mark.parametrize("definition,nested", [(KimiK3QModel, True), (KimiLinearQModel, False)])
def test_only_native_routed_projections_are_quantizable(definition, nested):
    text = SimpleNamespace(num_experts=4)
    config = SimpleNamespace(text_config=text) if nested else text
    modules = {name for group in definition.simple_layer_modules(
        config, SimpleNamespace(dynamic=None)) for name in group}
    expected = {f"block_sparse_moe.experts.{expert}.{projection}"
                for expert in range(4) for projection in ("w1", "w3", "w2")}
    assert modules == expected
    assert definition.extract_layers_node() == [
        "language_model.model.layers" if nested else "model.layers"
    ]
    if nested:
        assert definition.out_of_model_tensors == {"prefixes": ["vision_tower.", "mm_projector."]}


def test_ordinary_looper_fails_before_calibration_or_allocations():
    with pytest.raises(NotImplementedError, match="prefix_sum and block_residual"):
        KimiK3QModel.quantize(SimpleNamespace())
    with pytest.raises(NotImplementedError, match="AttnRes"):
        KimiK3QModel.pre_quantize(SimpleNamespace(), SimpleNamespace(use_attn_residuals=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_mxfp4_nibbles_and_e8m0_match_existing_decoder(utils, dtype):
    # Every ordered pair of E2M1 codes occurs, including both signed zeros.
    packed = torch.arange(256, dtype=torch.uint8).reshape(4, 64)
    scale_bytes = torch.tensor([[0, 1, 126, 127], [128, 130, 122, 124],
                                [123, 127, 129, 121], [127, 125, 126, 128]], dtype=torch.uint8)
    decoded = dequantize_f4_e2m1(packed, target_dtype=torch.float32).reshape(4, 4, 32)
    expected = (decoded * torch.exp2(scale_bytes.float() - 127)[..., None]).reshape(4, 128).to(dtype)
    actual = utils.decode_mxfp4(packed, scale_bytes, dtype=dtype)
    assert actual.dtype is dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_mxfp4_extreme_e8m0_matches_float64_oracle(utils):
    packed = torch.full((2, 16), 0x99, dtype=torch.uint8)  # two -0.5 values
    scales = torch.tensor([[0], [254]], dtype=torch.uint8)
    actual = utils.decode_mxfp4(packed, scales, dtype=torch.float32)
    expected = (-0.5 * torch.exp2(scales.double() - 127)).expand(2, 32).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed,scales", [
    (torch.ones(2, 16), torch.ones(2, 1, dtype=torch.uint8)),
    (torch.ones(2, 15, dtype=torch.uint8), torch.ones(2, 1, dtype=torch.uint8)),
    (torch.ones(2, 16, dtype=torch.uint8), torch.ones(2, 2, dtype=torch.uint8)),
    (torch.ones(2, 16, dtype=torch.uint8), torch.ones(2, 1)),
])
def test_mxfp4_rejects_invalid_storage_or_geometry(utils, packed, scales):
    with pytest.raises(ValueError, match="geometry"):
        utils.decode_mxfp4(packed, scales)


def test_mxfp4_rejects_nan_scales_and_overflow(utils):
    with pytest.raises(ValueError, match="NaN"):
        utils.decode_mxfp4(torch.ones(1, 16, dtype=torch.uint8),
                          torch.full((1, 1), 255, dtype=torch.uint8))
    with pytest.raises(ValueError, match="nonfinite"):
        utils.decode_mxfp4(torch.full((1, 16), 0x77, dtype=torch.uint8),
                          torch.full((1, 1), 254, dtype=torch.uint8), dtype=torch.float32)


def test_source_reads_dense_and_packed_tensors_without_loading_other_weights(utils, monkeypatch, tmp_path):
    dense = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
    packed = torch.full((2, 16), 0x21, dtype=torch.uint8)
    scales = torch.tensor([[127], [128]], dtype=torch.uint8)
    names = {
        "dense.weight": dense,
        "expert.w1.weight_packed": packed,
        "expert.w1.weight_scale": scales,
    }
    save_file(names, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "kimi_k3", "text_config": {"num_hidden_layers": 2},
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": dict.fromkeys(names, "model.safetensors"),
    }))
    monkeypatch.setattr(utils, "load_calibration_reference", lambda _: SimpleNamespace())
    source = utils.KimiK3Source(tmp_path)
    torch.testing.assert_close(source.decoded("dense.weight"), dense, rtol=0, atol=0)
    torch.testing.assert_close(source.decoded("expert.w1.weight"),
                               utils.decode_mxfp4(packed, scales), rtol=0, atol=0)
    # The utility intentionally owns a copy rather than a live shard mapping.
    copied = source.tensor("dense.weight")
    pointer = copied.data_ptr()
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            address = line.split()[0]
            start, end = (int(value, 16) for value in address.split("-"))
            if start <= pointer < end:
                assert "model.safetensors" not in line
                break
        else:
            pytest.fail("tensor allocation has no process memory mapping")
    with pytest.raises(KeyError):
        source.decoded("expert.w2.weight")
    with pytest.raises(ValueError, match="layer outside"):
        source.load_block(2, device="cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("linear_beta", [25.0, None])
def test_situ_matches_checkpoint_rounding_and_independent_oracle(utils, reference, dtype, linear_beta):
    gate = torch.tensor([-80, -25, -4, -1, -0.0, 0, 1, 4, 25, 80], dtype=dtype)
    up = torch.tensor([80, 25, -25, 1, -0.0, 0, -1, -4, 4, -80], dtype=dtype)
    native = reference.SituAndMul(beta=4.0, linear_beta=linear_beta)(torch.cat((gate, up)))
    actual = utils.situ(gate, up, beta=4.0, linear_beta=linear_beta)
    g, u = gate.double(), up.double()
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    oracle = (4.0 * torch.tanh(g / 4.0) / (1.0 + torch.exp(-g)) * u)
    assert actual.dtype is dtype
    torch.testing.assert_close(actual, native, rtol=0, atol=0)
    if dtype is torch.float32:
        torch.testing.assert_close(actual.double(), oracle, rtol=1e-6, atol=1e-6)
    else:
        torch.testing.assert_close(actual, oracle.to(dtype), rtol=0, atol=0)


class _TinyAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


def _tiny_config(reference, *, first_dense=0):
    return reference.KimiLinearConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, mla_use_nope=True,
        num_experts=4, num_experts_per_token=2, num_shared_experts=1,
        moe_intermediate_size=6, routed_expert_hidden_size=4,
        latent_moe_use_norm=True, hidden_act="situ", activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0, attn_res_block_size=2,
        first_k_dense_replace=first_dense, routed_scaling_factor=1.7,
        num_expert_group=2, topk_group=1,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("first_dense", [0, 1])
def test_explicit_boundaries_match_native_decoder_and_residual_bank(
    utils, reference, monkeypatch, dtype, first_dense,
):
    monkeypatch.setattr(reference, "KimiMLAAttention", _TinyAttention)
    torch.manual_seed(917)
    config = _tiny_config(reference, first_dense=first_dense)
    blocks = [reference.KimiDecoderLayer(config, layer).to(dtype).eval() for layer in range(4)]
    for block in blocks:
        if hasattr(block, "block_sparse_moe"):
            block.block_sparse_moe.gate.e_score_correction_bias.data.copy_(torch.tensor([.1, -.1, .2, -.2]))
    expected_prefix = torch.randn(2, 3, 8).to(dtype)
    expected_bank = torch.empty(6, 0, 8, dtype=dtype)
    actual_prefix, actual_bank = expected_prefix.clone(), expected_bank.clone()
    for layer, block in enumerate(blocks):
        mlp = getattr(block, "block_sparse_moe", None) or block.mlp
        captured = []
        hook = mlp.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].clone()))
        expected_prefix, expected_bank = block(expected_prefix, block_residual=expected_bank)
        hook.remove()
        actual_prefix, actual_bank, moe_input = utils.prepare_moe(block, actual_prefix, actual_bank)
        torch.testing.assert_close(moe_input, captured[0], rtol=0, atol=0)
        actual_prefix, actual_bank = utils.finish_moe(actual_prefix, actual_bank, mlp(moe_input))
        torch.testing.assert_close(actual_prefix, expected_prefix, rtol=0, atol=0)
        torch.testing.assert_close(actual_bank, expected_bank, rtol=0, atol=0)
        assert actual_bank.shape[1] == layer // 2 + 1


def _linear64(x, module, dtype):
    return F.linear(x.double(), module.weight.double()).to(dtype)


def _mlp_oracle(x, mlp, dtype):
    gate = _linear64(x, mlp.w1 if hasattr(mlp, "w1") else mlp.gate_proj, dtype)
    up = _linear64(x, mlp.w3 if hasattr(mlp, "w3") else mlp.up_proj, dtype)
    activated = (4.0 * torch.tanh(gate.double() / 4.0) * torch.sigmoid(gate.double())
                 * (25.0 * torch.tanh(up.double() / 25.0))).to(dtype)
    return _linear64(activated, mlp.w2 if hasattr(mlp, "w2") else mlp.down_proj, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_latent_moe_router_weighting_and_output_normalization_match_oracle(reference, dtype):
    torch.manual_seed(1927)
    moe = reference.KimiSparseMoeBlock(_tiny_config(reference)).to(dtype).eval()
    moe.gate.e_score_correction_bias.data.copy_(torch.tensor([.4, -.3, .2, -.1]))
    inputs = torch.randn(2, 3, 8).to(dtype)
    native_ids, native_weights = moe.gate(inputs)
    logits = F.linear(inputs.reshape(-1, 8).double(), moe.gate.weight.double()).float()
    scores = torch.sigmoid(logits)
    choices = scores + moe.gate.e_score_correction_bias.float()
    # K3's selected group uses the sum of its best two corrected scores.
    chosen_group = choices.reshape(-1, 2, 2).sum(-1).argmax(-1)
    mask = torch.arange(4).div(2, rounding_mode="floor")[None] == chosen_group[:, None]
    ids = choices.masked_fill(~mask, -torch.inf).topk(2, sorted=False).indices
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 1.7
    assert torch.equal(ids, native_ids)
    torch.testing.assert_close(weights, native_weights, rtol=1e-6, atol=1e-6)
    latent = _linear64(inputs.reshape(-1, 8), moe.routed_expert_down_proj, dtype)
    routed = torch.stack([
        sum((_mlp_oracle(latent[token:token + 1], moe.experts[int(ids[token, rank])], dtype)[0]
             .float() * weights[token, rank]) for rank in range(2))
        for token in range(latent.shape[0])
    ]).to(dtype)
    # Norm is applied AFTER the weighted routed reduction and BEFORE up_proj.
    normed = (routed.double() * torch.rsqrt(routed.double().square().mean(-1, keepdim=True)
              + moe.routed_expert_norm.variance_epsilon)).to(dtype) * moe.routed_expert_norm.weight
    expected = _linear64(normed, moe.routed_expert_up_proj, dtype).reshape_as(inputs)
    expected = expected + _mlp_oracle(inputs, moe.shared_experts, dtype)
    actual = moe(inputs)
    if dtype is torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
