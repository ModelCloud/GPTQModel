# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Model-level parity and failure-mode tests for LeapQuant state quantization."""

import pytest
import torch
from eager_ssm_kernels import eager_torch_ssm_kernels
from torch import nn
from transformers.cache_utils import DynamicCache, DynamicLayer

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.ports.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
from gptqmodel.models.ports.kimi_k3.modeling_kimi_linear import KimiLinearForCausalLM
from gptqmodel.models.ports.native.glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
from gptqmodel.models.ports.native.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextTextLinearAttention,
    Glm5NextTextModel,
)
from gptqmodel.models.ports.native.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from gptqmodel.models.ports.native.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from gptqmodel.models.ports.native.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from gptqmodel.models.ports.native.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeGatedDeltaNet,
)
from gptqmodel.models.ports.native.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from gptqmodel.models.ports.native.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
)
from gptqmodel.utils.leapquant import (
    LeapQuantConfig,
    LeapQuantDynamicCache,
    LeapQuantLinearAttentionLayer,
    make_leapquant_cache,
)


def _devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _kimi_config():
    return KimiLinearConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        mla_use_nope=True,
        mla_use_output_gate=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=12,
        routed_expert_hidden_size=8,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=2,
        first_k_dense_replace=1,
        pad_token_id=0,
        linear_attn_config={
            "kda_layers": [1, 2, 3, 4],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )


def _qwen_config():
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
    )


def _qwen3_5_moe_config():
    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        pad_token_id=0,
    )


def _qwen3_next_config():
    return Qwen3NextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        pad_token_id=0,
    )


def _glm_config():
    return Glm5NextTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        index_n_heads=1,
        index_head_dim=8,
        index_topk=4,
        index_kpool=4,
        linear_num_heads=2,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        n_routed_experts=2,
        num_experts_per_tok=1,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        vocab_size=64,
        pad_token_id=0,
    )


def _glm_model_config():
    config = Glm5NextTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        index_n_heads=1,
        index_head_dim=8,
        index_topk=4,
        index_kpool=4,
        linear_num_heads=2,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        n_routed_experts=16,
        num_experts_per_tok=8,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        vocab_size=64,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    return config


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    "family",
    [
        "kimi",
        "qwen",
        "qwen-multimodal",
        "qwen3_5_moe",
        "qwen3_5_moe-multimodal",
        "qwen3_next",
        "glm",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generate_uses_packed_recurrent_state(device, family, dtype):
    torch.manual_seed(42)
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32}
    if family == "kimi":
        model = KimiLinearForCausalLM(_kimi_config())
    elif family == "qwen":
        model = Qwen3_5ForCausalLM(_qwen_config())
    elif family == "qwen-multimodal":
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=_qwen_config(), vision_config=vision))
    elif family == "qwen3_5_moe":
        model = Qwen3_5MoeForCausalLM(_qwen3_5_moe_config())
    elif family == "qwen3_5_moe-multimodal":
        model = Qwen3_5MoeForConditionalGeneration(
            Qwen3_5MoeConfig(text_config=_qwen3_5_moe_config(), vision_config=vision)
        )
    elif family == "qwen3_next":
        model = Qwen3NextForCausalLM(_qwen3_next_config())
    else:
        model = Glm5NextForConditionalGeneration(
            Glm5NextConfig(text_config=_glm_model_config(), vision_config={**vision, "projection_intermediate_size": 32}),
        )
    wrapper = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model.to(device=device, dtype=dtype).eval()
    wrapper.tokenizer = None

    ids = torch.tensor([[1, 5, 7, 9]], device=device)
    kwargs = {"max_new_tokens": 3, "min_new_tokens": 3, "return_dict_in_generate": True}
    if family in ("qwen", "qwen3_5_moe", "qwen3_5_moe-multimodal", "qwen3_next"):
        kwargs["cache_implementation"] = "dynamic"
    config = LeapQuantConfig(bits=8, window=4, rank=2)
    with eager_torch_ssm_kernels():
        dense = wrapper.generate(ids, **kwargs)
        wrapper.enable_state_quant(config)
        packed = wrapper.generate(ids, **kwargs)

    assert dense.past_key_values.layers[0].recurrent_states[0] is not None
    layer = packed.past_key_values.layers[0]
    assert getattr(layer, "state_quant", None) == config
    assert layer.recurrent_states[0] is None
    assert layer._leapquant.window_count == 2
    assert packed.past_key_values.get_seq_length() == ids.shape[1] + 2
    assert torch.equal(packed.sequences, dense.sequences)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["qwen", "qwen3_5_moe", "qwen3_next", "glm"])
def test_padded_decode_fails_before_cache_updates(device, family):
    torch.manual_seed(53)
    if family == "qwen":
        model = Qwen3_5ForCausalLM(_qwen_config())
    elif family == "qwen3_5_moe":
        model = Qwen3_5MoeForCausalLM(_qwen3_5_moe_config())
    elif family == "qwen3_next":
        model = Qwen3NextForCausalLM(_qwen3_next_config())
    else:
        model = Glm5NextTextModel(_glm_model_config())
    model = model.to(device).eval()
    wrapper = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.enable_state_quant(LeapQuantConfig(window=4, rank=2))
    ids = torch.tensor([[1, 5, 7, 9]], device=device)

    with eager_torch_ssm_kernels(), torch.no_grad():
        cache = model(ids[:, :-1], use_cache=True).past_key_values
        layer = cache.layers[0]
        before = layer.conv_states[0].clone()
        with pytest.raises(ValueError, match="unpadded"):
            model(ids[:, -1:], past_key_values=cache, attention_mask=ids.ne(9))
        torch.testing.assert_close(layer.conv_states[0], before, rtol=0, atol=0)
        assert layer._leapquant.window_count == 0
        assert cache.get_seq_length() == 3
        model(ids[:, -1:], past_key_values=cache, attention_mask=torch.ones_like(ids))
        assert cache.get_seq_length() == 4
        assert layer._leapquant.window_count == 1


@pytest.mark.parametrize("operation", ["crop", "expand", "repeat", "select"])
def test_unsupported_cache_operation_preserves_attention_prefix(operation):
    config = _qwen_config()
    config.layer_types = ["full_attention", "linear_attention"]
    model = Qwen3_5ForCausalLM(config).eval()
    model.model._gptqmodel_state_quant = LeapQuantConfig(rank=2)
    with eager_torch_ssm_kernels(), torch.no_grad():
        cache = model(torch.tensor([[1, 5, 7, 9]]), use_cache=True).past_key_values
    before = cache.layers[0].keys.clone()

    if operation == "crop":
        with pytest.raises(RuntimeError, match="cannot be cropped"):
            cache.crop(-1)
    elif operation == "repeat":
        with pytest.raises(ValueError, match="cache batch expansion"):
            cache.batch_repeat_interleave(2)
    else:
        reorder = cache.reorder_cache if operation == "expand" else cache.batch_select_indices
        with pytest.raises(ValueError, match="one index per live sequence"):
            reorder(torch.tensor([0, 0]))
    torch.testing.assert_close(cache.layers[0].keys, before, rtol=0, atol=0)


def test_generate_validates_new_cache_policy_and_honors_caller_cache():
    model = Qwen3_5ForCausalLM(_qwen_config()).eval()
    model.model._gptqmodel_state_quant = LeapQuantConfig(rank=2)
    ids = torch.tensor([[1, 5, 7, 9]])
    with eager_torch_ssm_kernels():
        with pytest.raises(ValueError, match="requires a dynamic"):
            model.generate(ids, max_new_tokens=2, cache_implementation="static")
        with pytest.raises(ValueError, match="assisted generation is not supported"):
            model.generate(ids, max_new_tokens=2, prompt_lookup_num_tokens=2)
        cache = DynamicCache(config=model.config)
        output = model.generate(ids, max_new_tokens=2, past_key_values=cache, return_dict_in_generate=True)
        assert output.past_key_values is cache
        assert cache.layers[0].recurrent_states[0] is not None
        no_cache = model.generate(ids, max_new_tokens=2, use_cache=False, return_dict_in_generate=True)
        assert no_cache.past_key_values is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="accelerator required for cache offload")
def test_offloaded_cache_reorder_preserves_next_token():
    torch.manual_seed(19)
    model = Qwen3_5ForCausalLM(_qwen_config()).to("cuda").eval()
    model.model._gptqmodel_state_quant = LeapQuantConfig(rank=2)
    ids = torch.tensor([[1, 5, 7, 9, 11], [2, 4, 6, 8, 12]], device=model.device)
    permutation = torch.tensor([1, 0], device=model.device)

    with eager_torch_ssm_kernels(), torch.no_grad():
        caches = [model(ids[:, :3], use_cache=True).past_key_values for _ in range(2)]
        for cache in caches:
            model(ids[:, 3:4], past_key_values=cache)

        resident, offloaded = caches
        layer = offloaded.layers[0]
        layer.offload()
        assert layer.conv_states[0].device.type == "cpu"
        assert layer._leapquant.device.type == "cpu"
        assert layer._leapquant.window_count == 1
        resident.reorder_cache(permutation)
        offloaded.reorder_cache(permutation)
        layer.prefetch()

        token = ids.index_select(0, permutation)[:, 4:]
        expected = model(token, past_key_values=resident).logits
        actual = model(token, past_key_values=offloaded).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert layer._leapquant.window_count == 2


@pytest.mark.parametrize("device", _devices())
def test_kimi_quantized_cache_matches_fp32(device):
    torch.manual_seed(791)
    model = KimiLinearForCausalLM(_kimi_config()).to(device).eval()
    ids = torch.tensor([[1, 5, 7, 9, 11, 13]], device=device)
    with torch.no_grad():
        baseline = model(ids[:, :4], use_cache=True)
        baseline_logits = model(ids[:, 4:], past_key_values=baseline.past_key_values).logits

    model.model._gptqmodel_state_quant = LeapQuantConfig(bits=8, window=4, rank=2)
    with torch.no_grad():
        quantized = model(ids[:, :4], use_cache=True)
        quantized_logits = model(ids[:, 4:], past_key_values=quantized.past_key_values).logits

    layer = quantized.past_key_values.layers[0]
    assert layer._leapquant is not None
    assert layer.recurrent_states[0] is None
    assert not layer.is_croppable
    assert quantized.past_key_values.get_seq_length() == ids.shape[1]
    assert (baseline_logits - quantized_logits).abs().max().item() < 1e-3


@pytest.mark.parametrize("device", _devices())
def test_qwen_quantized_cache_matches_fp32(device):
    torch.manual_seed(123)
    model = Qwen3_5ForCausalLM(_qwen_config()).to(device).eval()
    ids = torch.tensor([[1, 5, 7, 9, 11, 13]], device=device)
    with eager_torch_ssm_kernels(), torch.no_grad():
        baseline = model(ids[:, :4], use_cache=True)
        baseline_logits = model(ids[:, 4:], past_key_values=baseline.past_key_values).logits

    model.model._gptqmodel_state_quant = LeapQuantConfig(bits=8, window=4, rank=2)
    with eager_torch_ssm_kernels(), torch.no_grad():
        quantized = model(ids[:, :4], use_cache=True)
        quantized_logits = model(ids[:, 4:], past_key_values=quantized.past_key_values).logits

    layer = quantized.past_key_values.layers[0]
    assert layer._leapquant is not None
    assert layer.recurrent_states[0] is None
    assert quantized.past_key_values.get_seq_length() == ids.shape[1]
    assert (baseline_logits - quantized_logits).abs().max().item() < 1e-3


@pytest.mark.parametrize("device", _devices())
def test_qwen_grouped_key_heads_quantized_decode(device):
    torch.manual_seed(99)
    config = _qwen_config()
    config.linear_num_key_heads = 1
    config.linear_num_value_heads = 2
    module = Qwen3_5GatedDeltaNetForTest(config).to(device).eval()
    prefill = torch.randn(1, 4, config.hidden_size, device=device)
    decode = torch.randn(1, 1, config.hidden_size, device=device)
    with eager_torch_ssm_kernels(), torch.no_grad():
        baseline_cache = DynamicCache(config=config)
        baseline_prefill = module(prefill, cache_params=baseline_cache)
        baseline_decode = module(decode, cache_params=baseline_cache)

    with eager_torch_ssm_kernels(), torch.no_grad():
        quantized_cache = make_leapquant_cache(config, LeapQuantConfig(bits=8, window=4, rank=2))
        quantized_prefill = module(prefill, cache_params=quantized_cache)
        quantized_decode = module(decode, cache_params=quantized_cache)

    layer = quantized_cache.layers[0]
    assert layer._leapquant.num_heads == 2
    assert (baseline_prefill - quantized_prefill).abs().max().item() == 0
    assert (baseline_decode - quantized_decode).abs().max().item() < 1e-2


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["qwen3_5_moe", "qwen3_next"])
def test_qwen_gdn_family_quantized_cache_matches_fp32(device, family):
    torch.manual_seed(123)
    if family == "qwen3_5_moe":
        model = Qwen3_5MoeForCausalLM(_qwen3_5_moe_config())
    else:
        model = Qwen3NextForCausalLM(_qwen3_next_config())
    model = model.to(device).eval()
    ids = torch.tensor([[1, 5, 7, 9, 11, 13]], device=device)
    with eager_torch_ssm_kernels(), torch.no_grad():
        baseline = model(ids[:, :4], use_cache=True)
        baseline_logits = model(ids[:, 4:], past_key_values=baseline.past_key_values).logits

    model.model._gptqmodel_state_quant = LeapQuantConfig(bits=8, window=4, rank=2)
    with eager_torch_ssm_kernels(), torch.no_grad():
        quantized = model(ids[:, :4], use_cache=True)
        quantized_logits = model(ids[:, 4:], past_key_values=quantized.past_key_values).logits

    assert isinstance(quantized.past_key_values, LeapQuantDynamicCache)
    layer = quantized.past_key_values.layers[0]
    assert layer._leapquant is not None
    assert layer.recurrent_states[0] is None
    assert quantized.past_key_values.get_seq_length() == ids.shape[1]
    assert (baseline_logits - quantized_logits).abs().max().item() < 1e-3


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("family", ["qwen3_5_moe", "qwen3_next"])
def test_qwen_gdn_family_grouped_key_heads_quantized_decode(device, family):
    torch.manual_seed(99)
    if family == "qwen3_5_moe":
        config = _qwen3_5_moe_config()
        module_cls = Qwen3_5MoeGatedDeltaNet
    else:
        config = _qwen3_next_config()
        module_cls = Qwen3NextGatedDeltaNet
    config.linear_num_key_heads = 1
    config.linear_num_value_heads = 2
    module = module_cls(config, 0).to(device).eval()
    prefill = torch.randn(1, 4, config.hidden_size, device=device)
    decode = torch.randn(1, 1, config.hidden_size, device=device)
    with eager_torch_ssm_kernels(), torch.no_grad():
        baseline_cache = DynamicCache(config=config)
        baseline_prefill = module(prefill, cache_params=baseline_cache)
        baseline_decode = module(decode, cache_params=baseline_cache)

    with eager_torch_ssm_kernels(), torch.no_grad():
        quantized_cache = make_leapquant_cache(config, LeapQuantConfig(bits=8, window=4, rank=2))
        quantized_prefill = module(prefill, cache_params=quantized_cache)
        quantized_decode = module(decode, cache_params=quantized_cache)

    layer = quantized_cache.layers[0]
    assert layer._leapquant.num_heads == 2
    assert (baseline_prefill - quantized_prefill).abs().max().item() == 0
    assert (baseline_decode - quantized_decode).abs().max().item() < 1e-2


@pytest.mark.parametrize("device", _devices())
def test_qwen3_next_quantized_prefill_handles_rank_deficient_state(device):
    # This state is numerically rank one in one head, so the Compensator Gram
    # underflows to subnormals. CUDA eigh returned NaN for it.
    torch.manual_seed(53)
    model = Qwen3NextForCausalLM(_qwen3_next_config()).to(device).eval()
    model.model._gptqmodel_state_quant = LeapQuantConfig(bits=8, window=4, rank=2)
    ids = torch.tensor([[1, 5, 7]], device=device)
    with eager_torch_ssm_kernels(), torch.no_grad():
        cache = model(ids, use_cache=True).past_key_values

    state = cache.layers[0]._leapquant
    assert torch.isfinite(state.key_scale).all()
    assert torch.isfinite(state.value_scale).all()
    assert torch.isfinite(state.materialize()).all()


@pytest.mark.parametrize("device", _devices())
def test_glm_quantized_attention_matches_fp32(device):
    torch.manual_seed(321)
    config = _glm_config()
    module = Glm5NextTextLinearAttention(config, 0).to(device).eval()
    prefill = torch.randn(1, 4, 32, device=device)
    decode = torch.randn(1, 1, 32, device=device)
    with eager_torch_ssm_kernels(), torch.no_grad():
        baseline_cache = DynamicCache(config=config)
        baseline_prefill = module(prefill, cache_params=baseline_cache)
        baseline_decode = module(decode, cache_params=baseline_cache)

    with eager_torch_ssm_kernels(), torch.no_grad():
        quantized_cache = make_leapquant_cache(config, LeapQuantConfig(bits=8, window=4, rank=2))
        quantized_prefill = module(prefill, cache_params=quantized_cache)
        quantized_decode = module(decode, cache_params=quantized_cache)

    layer = quantized_cache.layers[0]
    assert layer._leapquant is not None
    assert layer.recurrent_states[0] is None
    assert (baseline_prefill - quantized_prefill).abs().max().item() == 0
    assert (baseline_decode - quantized_decode).abs().max().item() < 1e-2


@pytest.mark.parametrize("device", _devices())
def test_glm_text_model_quantized_cache_matches_fp32(device):
    torch.manual_seed(77)
    config = _glm_model_config()
    model = Glm5NextTextModel(config).to(device).eval()
    ids = torch.tensor([[1, 5, 7, 9, 11, 13]], device=device)
    with eager_torch_ssm_kernels(), torch.no_grad():
        baseline = model(ids[:, :4], use_cache=True)
        baseline_hidden = model(ids[:, 4:], past_key_values=baseline.past_key_values).last_hidden_state

    model._gptqmodel_state_quant = LeapQuantConfig(bits=8, window=4, rank=2)
    with eager_torch_ssm_kernels(), torch.no_grad():
        quantized = model(ids[:, :4], use_cache=True)
        quantized_hidden = model(ids[:, 4:], past_key_values=quantized.past_key_values).last_hidden_state

    layer = quantized.past_key_values.layers[0]
    assert layer._leapquant is not None
    assert layer.recurrent_states[0] is None
    assert quantized.past_key_values.get_seq_length() == ids.shape[1]
    assert (baseline_hidden - quantized_hidden).abs().max().item() < 1e-2


def test_make_leapquant_cache_replaces_only_linear_layers():
    config = _qwen_config()
    cache = make_leapquant_cache(config, LeapQuantConfig())
    assert isinstance(cache.layers[0], LeapQuantLinearAttentionLayer)
    assert isinstance(cache.layers[1], DynamicLayer)
    assert cache.layers[0].state_quant is not None
    assert not cache.layers[0].is_croppable


def test_quantized_qwen_rejects_padded_mask_and_cu_seqlens():
    config = _qwen_config()
    module = Qwen3_5GatedDeltaNetForTest(config)
    cache = make_leapquant_cache(config, LeapQuantConfig())
    hidden = torch.randn(1, 4, config.hidden_size)
    with pytest.raises(ValueError, match="unpadded"):
        module(hidden, cache_params=cache, attention_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    with pytest.raises(ValueError, match="cu_seqlens"):
        module(hidden, cache_params=cache, cu_seq_lens_q=torch.tensor([0, 4]))


def test_quantized_glm_rejects_padded_mask_and_cu_seqlens():
    config = _glm_config()
    module = Glm5NextTextLinearAttention(config, 0)
    cache = make_leapquant_cache(config, LeapQuantConfig())
    hidden = torch.randn(1, 4, config.hidden_size)
    with pytest.raises(ValueError, match="unpadded"):
        module(hidden, cache_params=cache, attention_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    with pytest.raises(ValueError, match="cu_seqlens"):
        module(hidden, cache_params=cache, cu_seqlens=torch.tensor([0, 4]))


def test_enable_state_quant_wrapper_validation():
    class Stub:
        def __init__(self):
            self.model = nn.Module()
            self.model.supports_state_quant = True

    stub = Stub()
    config = LeapQuantConfig()
    BaseQModel.enable_state_quant(stub, config)
    assert stub.state_quant is config
    assert stub.model._gptqmodel_state_quant is config
    with pytest.raises(TypeError):
        BaseQModel.enable_state_quant(stub, "8bit")
    empty = Stub()
    empty.model = nn.Module()
    with pytest.raises(ValueError, match="no LeapQuant-capable"):
        BaseQModel.enable_state_quant(empty, config)
    stub._runtime_generate = object()
    with pytest.raises(ValueError, match="no LeapQuant-capable"):
        BaseQModel.enable_state_quant(stub, config)


@pytest.mark.parametrize("family", ["kimi", "qwen", "qwen3_5_moe", "qwen3_next", "glm"])
def test_enable_state_quant_requires_a_recurrent_layer(family):
    if family == "kimi":
        config = _kimi_config()
        config.linear_attn_config = None
        model = KimiLinearForCausalLM(config)
    elif family == "qwen":
        config = _qwen_config()
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        model = Qwen3_5ForCausalLM(config)
    elif family == "qwen3_5_moe":
        config = _qwen3_5_moe_config()
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        model = Qwen3_5MoeForCausalLM(config)
    elif family == "qwen3_next":
        config = _qwen3_next_config()
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        model = Qwen3NextForCausalLM(config)
    else:
        config = _glm_model_config()
        config.layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
        model = Glm5NextTextModel(config)

    wrapper = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    with pytest.raises(ValueError, match="no LeapQuant-capable"):
        wrapper.enable_state_quant(LeapQuantConfig())


def test_kimi_rejects_padded_mask():
    model = KimiLinearForCausalLM(_kimi_config()).eval()
    model.model._gptqmodel_state_quant = LeapQuantConfig(bits=8, window=4, rank=2)
    ids = torch.tensor([[1, 5, 7, 9]])
    with pytest.raises(ValueError, match="unpadded"):
        model(ids, attention_mask=torch.tensor([[1, 1, 0, 0]]))


def test_kimi_quantized_reorder_is_atomic():
    model = KimiLinearForCausalLM(_kimi_config()).eval()
    model.model._gptqmodel_state_quant = LeapQuantConfig(bits=8, window=4, rank=2)
    ids = torch.tensor([[1, 5, 7], [1, 4, 6]])
    with torch.no_grad():
        output = model(ids, use_cache=True)
    cache = output.past_key_values
    layer = cache.layers[0]
    before = layer._leapquant.codes.clone()
    cache.reorder_cache(torch.tensor([1, 0]))
    assert torch.equal(layer._leapquant.codes, before.index_select(0, torch.tensor([1, 0])))


def test_quantized_qwen_reorder_is_atomic():
    config = _qwen_config()
    cache = make_leapquant_cache(config, LeapQuantConfig())
    hidden = torch.randn(2, 4, config.hidden_size)
    module = Qwen3_5GatedDeltaNetForTest(config)
    with eager_torch_ssm_kernels(), torch.no_grad():
        module(hidden, cache_params=cache)
    layer = cache.layers[0]
    before = layer._leapquant.codes.clone()
    cache.reorder_cache(torch.tensor([1, 0]))
    assert torch.equal(layer._leapquant.codes, before.index_select(0, torch.tensor([1, 0])))


class Qwen3_5GatedDeltaNetForTest(nn.Module):
    """Small wrapper that keeps the test on the port's GDN module."""

    def __init__(self, config):
        super().__init__()
        from gptqmodel.models.ports.native.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

        self.inner = Qwen3_5GatedDeltaNet(config, 0)

    def forward(self, hidden_states, **kwargs):
        return self.inner(hidden_states, **kwargs)
