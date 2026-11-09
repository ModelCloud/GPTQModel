# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent Torch oracle for full-GLM MTP envelope and prompt alignment."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm

from gptqmodel.utils.glm5_mtp import MTP_CONTRACT, GlmMTP, aligned_inputs, load_mtp


def _norm(x, weight, eps):
    # Explicit native GLM rounding: FP32 variance/rsqrt, activation cast,
    # then low-precision learned scale. Do not call the changed module.
    f = x.float()
    unit = (f * torch.rsqrt(f.square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype)
    return unit * weight


class Norm(GlmMoeDsaRMSNorm):
    def __init__(self, weight, eps=1e-5):
        super().__init__(weight.numel(), eps=eps)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_shift_is_prompt_local_and_uses_target_positions(dtype):
    hidden = torch.arange(2 * 5 * 8).reshape(2, 5, 8).to(dtype) / 20
    ids = torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]])
    embedding = torch.arange(20 * 8).reshape(20, 8).to(dtype) / 30
    target_norm = Norm(torch.linspace(.5, 1.2, 8).to(dtype))
    aligned = aligned_inputs(hidden, ids, embedding, target_norm)
    assert torch.equal(aligned["input_ids"], ids[:, 1:])
    assert torch.equal(aligned["position_ids"], torch.arange(4).expand(2, 4))
    assert not bool(aligned["inputs_embeds"][:, 0].any())
    assert torch.equal(aligned["inputs_embeds"][:, 1:], embedding[ids[:, 2:]])
    expected = _norm(hidden[:, :-1], target_norm.weight, 1e-5)
    torch.testing.assert_close(aligned["target_hidden"], expected, rtol=2e-3, atol=2e-3)
    expected_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(aligned["attention_mask"], expected_mask.expand(2, 1, 4, 4))
    assert aligned["inputs_embeds"].shape == (2, 4, 8)
    assert aligned["target_hidden"].dtype == dtype
    assert MTP_CONTRACT["target_tap"] == "post-model.norm"


@pytest.mark.cpu
def test_explicit_position_offset_and_post_norm_tap_do_not_double_normalize():
    x = torch.randn(1, 4, 8)
    ids = torch.tensor([[3, 4, 5, 6]])
    table = torch.randn(8, 8)
    pos = torch.arange(17, 21)[None]
    aligned = aligned_inputs(x, ids, table, None, position_ids=pos, target_tap="post_final_norm")
    assert torch.equal(aligned["target_hidden"], x[:, :-1])
    assert torch.equal(aligned["position_ids"], torch.tensor([[17, 18, 19]]))
    assert torch.equal(aligned["inputs_embeds"], table[ids[:, 1:]])
    with pytest.raises(ValueError, match="second final norm"):
        aligned_inputs(x, ids, table, nn.Identity(), target_tap="post_final_norm")
    with pytest.raises(ValueError, match="requires the native"):
        aligned_inputs(x, ids, table, None)


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["one_token", "token_shape", "token_dtype", "bad_tap", "positions", "negative_id", "high_id"])
def test_invalid_alignment_fails_closed(case):
    hidden = torch.randn(1, 4, 8)
    ids = torch.tensor([[0, 1, 2, 3]])
    table = torch.randn(8, 8)
    kwargs = {}
    if case == "one_token":
        hidden, ids = hidden[:, :1], ids[:, :1]
    elif case == "token_shape":
        ids = ids[:, :-1]
    elif case == "token_dtype":
        ids = ids.int()
    elif case == "bad_tap":
        kwargs["target_tap"] = "guess"
    elif case == "positions":
        kwargs["position_ids"] = torch.tensor([[0, 1, 3, 4]])
    elif case == "negative_id":
        ids[0, 2] = -1
    elif case == "high_id":
        ids[0, 2] = 8
    with pytest.raises(ValueError):
        aligned_inputs(hidden, ids, table, nn.Identity(), **kwargs)


def _adapter(dtype, device="cpu", hidden=128, scale=.015):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden)
            self.seen = None

        def forward(self, x, **kwargs):
            self.seen = kwargs
            return x + .125, torch.zeros((*x.shape[:2], 2), dtype=torch.int32, device=x.device)

    gen = torch.Generator().manual_seed(787)
    weights = [torch.rand(hidden, generator=gen).to(dtype=dtype, device=device) + .5 for _ in range(3)]
    eh = nn.Linear(hidden * 2, hidden, bias=False, dtype=dtype, device=device)
    eh.weight.data.copy_(torch.randn(eh.weight.shape, generator=gen).to(device=device, dtype=dtype) * scale)
    eh.requires_grad_(False)
    return GlmMTP(Decoder(), Norm(weights[0]), Norm(weights[1]), eh, Norm(weights[2]), layer_idx=78)


def _check_envelope(dtype, device="cpu", hidden_size=128):
    gen = torch.Generator().manual_seed(41)
    adapter = _adapter(dtype, device, hidden_size, scale=.001 if hidden_size > 128 else .015)
    hidden = torch.randn(2, 5, hidden_size, generator=gen).to(device=device, dtype=dtype) * .1
    embedding = torch.randn(20, hidden_size, generator=gen).to(device=device, dtype=dtype) * .1
    ids = torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]], device=device)
    final_norm = Norm(torch.linspace(.5, 1.2, hidden_size, device=device).to(dtype))
    aligned = aligned_inputs(hidden, ids, embedding, final_norm)
    expected_target = _norm(hidden[:, :-1], final_norm.weight, final_norm.eps)
    expected_embed = embedding[ids[:, 1:]].clone()
    expected_embed[:, 0] = 0
    expected_envelope = torch.cat((_norm(expected_embed, adapter.enorm.weight, adapter.enorm.eps),
        _norm(expected_target, adapter.hnorm.weight, adapter.hnorm.eps)), -1)
    actual = adapter.envelope(aligned)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected_envelope, rtol=2e-3, atol=2e-3)
    # Independent FP32 linear accumulation, separate from final dtype rounding.
    oracle_raw = expected_envelope.float() @ adapter.eh_proj.weight.float().T
    projected = adapter.projected_inputs(aligned)
    torch.testing.assert_close(projected.float(), oracle_raw, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(projected, oracle_raw.to(dtype), rtol=2e-3, atol=2e-3)
    result = adapter.forward_aligned(aligned, lambda x, p: (p, p))
    expected_hidden = projected + .125
    torch.testing.assert_close(result["hidden"], expected_hidden, rtol=0, atol=0)
    expected_normalized = _norm(expected_hidden, adapter.shared_head_norm.weight, adapter.shared_head_norm.eps)
    torch.testing.assert_close(result["normalized_hidden"], expected_normalized, rtol=2e-3, atol=2e-3)
    assert adapter.decoder.seen["prev_topk_indices"] is None
    assert adapter.decoder.seen["past_key_values"] is None
    assert adapter.decoder.seen["use_cache"] is False
    assert torch.equal(adapter.decoder.seen["position_ids"], torch.arange(4, device=device).expand(2, 4))
    return adapter


@pytest.mark.cpu
def test_fp32_envelope_and_projection_match_independent_fp64_arithmetic():
    torch.manual_seed(787)
    module = _adapter(torch.float32)
    hidden, embedding = torch.randn(2, 5, 128), torch.randn(20, 128)
    ids = torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]])
    final = Norm(torch.linspace(.5, 1.2, 128))
    def norm64(x, scale):
        x = x.double()
        return x / (x.square().sum(-1, keepdim=True) / x.shape[-1] + 1e-5).sqrt() * scale.double()
    target = norm64(hidden[:, :-1], final.weight)
    em = embedding[ids[:, 1:]].clone()
    em[:, 0] = 0
    envelope = torch.cat((norm64(em, module.enorm.weight), norm64(target, module.hnorm.weight)), -1)
    aligned = aligned_inputs(hidden, ids, embedding, final)
    actual_envelope = module.envelope(aligned).double()
    assert float(torch.linalg.vector_norm(actual_envelope-envelope)/torch.linalg.vector_norm(envelope)) < 1e-6
    expected = envelope @ module.eh_proj.weight.double().T
    actual = module.projected_inputs(aligned).double()
    assert float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)) < 1e-6


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_envelope_order_projection_and_single_final_norm(dtype):
    adapter = _check_envelope(dtype)
    assert all("embed" not in key and "target_norm" not in key and "lm_head" not in key
               for key in adapter.state_dict())


class ToySource:
    def __init__(self, config):
        self.config = config.to_dict()
        self._config = config
        self.requests = []
        self.closed = False

    def native_config(self):
        return self._config

    def materialize_block(self, block, layer, *, device, include_experts):
        assert layer == 3
        assert block.self_attn.config.indexer_types[-1] == "full"
        assert block.self_attn.config.mlp_layer_types[-1] == "sparse"
        assert block.self_attn.config.num_hidden_layers == 4
        assert not include_experts
        self.materialized = block
        block.layer_idx = layer
        gen = torch.Generator().manual_seed(41)
        for key, old in list(block.named_parameters()):
            weight = torch.randn(old.shape, generator=gen).to(dtype=torch.bfloat16, device=device) * .025
            if "norm" in key:
                weight += 1
            if key == "self_attn.indexer.weights_proj.weight":
                weight = weight.float()
            parent, _, leaf = key.rpartition(".")
            block.get_submodule(parent)._parameters[leaf] = nn.Parameter(weight, requires_grad=False)
        for key, old in list(block.named_buffers()):
            parent, _, leaf = key.rpartition(".")
            block.get_submodule(parent)._buffers[leaf] = torch.zeros(old.shape, dtype=torch.float32, device=device)

    def tensor(self, name, device):
        self.requests.append(name)
        hidden = self.config["hidden_size"]
        shape = (hidden, hidden * 2) if name.endswith("eh_proj.weight") else (hidden,)
        values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        if len(shape) == 2:
            values = values.sin() * .002
        else:
            values = .5 + values / hidden + (sum(name.encode()) % 13) / 30
        return values.bfloat16().to(device)

    def close(self):
        self.closed = True


@pytest.mark.cpu
@pytest.mark.parametrize("count", [0, 2])
def test_loader_rejects_other_native_prediction_depths(count):
    source = SimpleNamespace(native_config=lambda: SimpleNamespace(
        num_hidden_layers=78, num_nextn_predict_layers=count))
    with pytest.raises(ValueError, match="exactly one"):
        load_mtp(source, "cpu")


@pytest.mark.cpu
def test_loader_extends_only_full_glm_schedule_and_loads_own_extra_weights():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    config = GlmMoeDsaConfig(hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        q_lora_rank=128, kv_lora_rank=128, qk_nope_head_dim=64, qk_rope_head_dim=64,
        v_head_dim=128, n_routed_experts=4, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=128, first_k_dense_replace=3,
        indexer_types=["full", "full", "full"], num_nextn_predict_layers=1,
        rope_parameters={"rope_theta": 8000000., "rope_type": "default"})
    config._experts_implementation = "eager"
    source = ToySource(config)
    mtp = load_mtp(source, "cpu")
    assert mtp.layer_idx == 3
    assert mtp.decoder is source.materialized
    assert source.closed
    assert set(source.requests) == {f"model.layers.3.{part}.weight" for part in (
        "enorm", "hnorm", "eh_proj", "shared_head.norm")}
    assert len(source.requests) == 4
    assert len(config.mlp_layer_types) == len(config.indexer_types) == config.num_hidden_layers == 3
    assert mtp.eh_proj.weight.shape == (128, 256)
    assert mtp.eh_proj.weight.dtype == torch.bfloat16
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm, GlmMoeDsaRotaryEmbedding
    final = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).bfloat16()
    rotary = GlmMoeDsaRotaryEmbedding(source.native_config())
    torch.manual_seed(23)
    hidden = torch.randn(1, 5, 128).bfloat16()
    table = torch.randn(20, 128).bfloat16()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    # Separate native baseline, bypassing aligned_inputs/envelope/forward.
    target = final(hidden[:, :-1])
    embedded = table[ids[:, 1:]].clone()
    embedded[:, 0] = 0
    native_input = F.linear(torch.cat((mtp.enorm(embedded), mtp.hnorm(target)), -1), mtp.eh_proj.weight)
    positions = torch.arange(4)[None]
    native_output, native_topk = mtp.decoder(native_input, position_ids=positions,
        position_embeddings=rotary(native_input, positions),
        attention_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool).tril(),
        prev_topk_indices=None, past_key_values=None, use_cache=False)
    actual = mtp(hidden, ids, table, final, rotary)
    assert torch.equal(actual["hidden"], native_output)
    assert torch.equal(actual["prev_topk_indices"], native_topk)
    assert torch.equal(actual["normalized_hidden"], mtp.shared_head_norm(native_output))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires admitted CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [128, 6144])
def test_cuda_envelope_projection_dtype_and_alignment(dtype, hidden_size):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        _check_envelope(dtype, "cuda", hidden_size)
        torch.cuda.synchronize()
    finally:
        torch.set_float32_matmul_precision(previous)
