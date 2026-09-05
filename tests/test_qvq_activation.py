# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import DynamicCache
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.models.auto import (
    _activation_quantization_mode,
    _is_supported_quantization_config,
)
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear, qvq_dense_oracle_forward
from gptqmodel.nn_modules.qvq_fp8_cache import (
    QVQFP8DynamicCache,
    install_qvq_fp8_kv_cache,
    qvq_fp8_attention_forward,
)
from gptqmodel.quantization import FORMAT, QVQActivationConfig, QVQConfig
from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    quantize_qvq_linear,
    reconstruct_qvq_inner_weight,
)
from gptqmodel.quantization.qvq_activation import (
    dequantize_qvq_fp8_activation,
    fake_quantize_qvq_fp8_activation,
    quantize_qvq_fp8_activation,
)
from gptqmodel.quantization.qvq_codecs import pgc16_levels_for_version
from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b


def _prepared_calibration(**kwargs):
    return kwargs["calibration_dataset"]


def _processor(qcfg: QVQConfig) -> QVQProcessor:
    return QVQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=[
            {
                "input_ids": torch.tensor([[1, 2, 3, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            }
        ],
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )


def test_qvq_fp8_activation_reference_uses_one_e4m3_scale_per_token():
    source = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-3.0, -0.5, 1.25, 6.0],
            [0.001, -0.015, 0.125, -1.0],
        ],
        dtype=torch.float32,
    )

    quantized, scale = quantize_qvq_fp8_activation(source)
    dequantized = dequantize_qvq_fp8_activation(quantized, scale, dtype=source.dtype)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (3, 1)
    assert scale[0].item() == 1.0
    torch.testing.assert_close(scale[1:, 0], source[1:].abs().amax(dim=-1) / 448.0)
    assert torch.equal(dequantized[0], source[0])
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, source, atol=0.0, rtol=0.065)


def test_qvq_fp8_activation_aliases_and_straight_through_gradient():
    source = torch.randn(2, 16, requires_grad=True)
    _, _, dequantized = fake_quantize_qvq_fp8_activation(
        source,
        format="e4m3",
        scale_method="per_token",
        straight_through=True,
    )
    dequantized.sum().backward()
    assert torch.equal(source.grad, torch.ones_like(source))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("shape", ((17, 64), (257, 2048)))
def test_hopper_fused_fp8_row_quantization_is_bit_exact(dtype, shape):
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) < (8, 9):
        pytest.skip("fused E4M3 row quantization requires SM89 or newer")
    generator = torch.Generator(device="cuda").manual_seed(20260904 + shape[0])
    source = torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
    source[0].zero_()
    working = source.float()
    peak = working.abs().amax(dim=-1, keepdim=True)
    expected_scale = torch.where(peak > 0, peak / 448.0, torch.ones_like(peak))
    expected = torch.clamp(working / expected_scale, -448.0, 448.0).to(
        torch.float8_e4m3fn
    )

    actual, scale = quantize_qvq_fp8_activation(source, validate=False)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))


class _TinyCacheConfig:
    num_hidden_layers = 2
    is_encoder_decoder = False
    use_cache = True

    def get_text_config(self, decoder=True):
        assert decoder is True
        return self


def test_qvq_fp8_dynamic_cache_has_no_full_precision_residual():
    torch.manual_seed(13)
    config = _TinyCacheConfig()
    cache = QVQFP8DynamicCache(config, QVQActivationConfig())
    key = torch.randn(2, 4, 3, 16, dtype=torch.bfloat16)
    value = torch.randn(2, 4, 3, 16, dtype=torch.bfloat16)

    returned_key, returned_value = cache.update(key, value, 0)
    returned_key_2, returned_value_2 = cache.update(
        key[..., :1, :], value[..., :1, :], 0
    )
    layer = cache.layers[0]

    assert returned_key.dtype == torch.bfloat16
    assert returned_value.dtype == torch.bfloat16
    assert returned_key_2.shape[-2] == returned_value_2.shape[-2] == 4
    assert layer.keys.dtype == layer.values.dtype == torch.float8_e4m3fn
    assert layer.key_scales.dtype == layer.value_scales.dtype == torch.float32
    assert layer.keys.shape[-2] == layer.key_scales.shape[-2] == 4
    assert layer.keys.stride(-1) == 1
    assert layer.values.stride(-2) == 1
    torch.testing.assert_close(returned_key, key, atol=0.0, rtol=0.065)
    torch.testing.assert_close(returned_value, value, atol=0.0, rtol=0.065)

    telemetry = cache.telemetry()
    expected_payload_bytes = layer.keys.numel() + layer.values.numel()
    expected_scale_bytes = (layer.key_scales.numel() + layer.value_scales.numel()) * 4
    assert telemetry["all_payloads_fp8"] is True
    assert telemetry["no_full_precision_residual"] is True
    assert telemetry["storage_bytes"] == expected_payload_bytes + expected_scale_bytes
    assert telemetry["storage_ratio_vs_dense"] == pytest.approx(0.625)


def test_qvq_fp8_static_cache_preallocates_and_enforces_maximum():
    cache = QVQFP8DynamicCache(
        _TinyCacheConfig(), QVQActivationConfig(), max_cache_length=8
    )
    key = torch.randn(1, 1, 3, 16, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 16, dtype=torch.bfloat16)
    cache.update(key, value, 0)
    cache.update(key, value, 0)
    layer = cache.layers[0]

    assert layer.get_seq_length() == 6
    assert layer.get_max_cache_shape() == 8
    assert layer.capacity == 8
    assert layer.allocations == 1
    assert layer.reallocations == 0
    assert layer.values.stride(-2) == 1
    telemetry = cache.telemetry()
    assert telemetry["allocation_strategy"] == "static"
    assert telemetry["capacities"] == [8]
    assert telemetry["reallocations"] == 0

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        cache.update(key, value, 0)


def test_qvq_a8_runtime_injects_fp8_cache_and_rejects_dense_cache():
    class TinyCacheModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _TinyCacheConfig()

        def forward(self, input_ids, past_key_values=None, use_cache=None):
            if use_cache is False:
                return SimpleNamespace(past_key_values=None)
            key = input_ids.to(torch.bfloat16).reshape(1, 1, -1, 1)
            value = key + 1
            key, value = past_key_values.update(key, value, 0)
            return SimpleNamespace(
                past_key_values=past_key_values, key=key, value=value
            )

        def _prepare_cache_for_generation(self, *args, **kwargs):
            raise AssertionError("the dense cache preparer must be replaced")

    model = TinyCacheModel()
    install_qvq_fp8_kv_cache(model, QVQActivationConfig())
    output = model(torch.tensor([[1, 2, 3]]), use_cache=True)
    assert isinstance(output.past_key_values, QVQFP8DynamicCache)
    assert output.past_key_values.telemetry()["all_payloads_fp8"] is True

    dense_cache = DynamicCache(config=model.config)
    with pytest.raises(TypeError, match="requires QVQFP8DynamicCache"):
        model(torch.tensor([[1]]), past_key_values=dense_cache, use_cache=True)

    generation_config = SimpleNamespace(
        cache_implementation=None,
        cache_config=None,
        use_cache=True,
    )
    model_kwargs = {}
    model._prepare_cache_for_generation(
        generation_config,
        model_kwargs,
        generation_mode=None,
        batch_size=1,
        max_cache_length=16,
    )
    assert isinstance(model_kwargs["past_key_values"], QVQFP8DynamicCache)

    generation_config.cache_implementation = "dynamic"
    with pytest.raises(ValueError, match="fixes `cache_implementation`"):
        model._prepare_cache_for_generation(
            generation_config,
            {},
            generation_mode=None,
            batch_size=1,
            max_cache_length=16,
        )


def test_qvq_a8_registers_additive_padding_and_sliding_window_masks():
    class TinyMaskModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _TinyCacheConfig()

        def forward(self, input_ids, past_key_values=None, use_cache=None):
            del input_ids, past_key_values, use_cache

    model = TinyMaskModel()
    install_qvq_fp8_kv_cache(model, QVQActivationConfig())
    assert ALL_MASK_ATTENTION_FUNCTIONS["qvq_fp8"] is not None

    embeds = torch.zeros((2, 4, 8), dtype=torch.float32)
    left_padding = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    causal = create_causal_mask(
        model.config,
        embeds,
        left_padding,
        past_key_values=None,
    )
    assert tuple(causal.shape) == (2, 1, 4, 4)
    assert bool((causal[0, 0, 3, :2] < -1e20).all())
    assert bool((causal[0, 0, 3, 2:] == 0).all())
    assert bool((causal[1, 0, 3] == 0).all())

    model.config.sliding_window = 2
    sliding = create_sliding_window_causal_mask(
        model.config,
        embeds[:1],
        torch.ones((1, 4), dtype=torch.long),
        past_key_values=None,
    )
    assert bool((sliding[0, 0, 3, :2] < -1e20).all())
    assert bool((sliding[0, 0, 3, 2:] == 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("seed", (17, 29, 41))
def test_h200_fp8_attention_consumes_cache_without_dense_prefix_materialization(seed):
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        pytest.skip("native QVQ FP8 attention validation requires the assigned H200")

    torch.manual_seed(seed)
    cache = QVQFP8DynamicCache(_TinyCacheConfig(), QVQActivationConfig())
    query = torch.randn(1, 8, 3, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 3, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 2, 3, 64, device="cuda", dtype=torch.bfloat16)
    key_view, value_view = cache.update(key, value, 0)
    mask = torch.full((1, 1, 3, 3), float("-inf"), device="cuda")
    mask = torch.triu(mask, diagonal=1)

    output, weights = qvq_fp8_attention_forward(
        SimpleNamespace(num_key_value_groups=4, training=False),
        query,
        key_view,
        value_view,
        None,
        scaling=0.125,
        output_attentions=True,
    )
    layer = cache.layers[0]
    length = layer.get_seq_length()
    dense_key = (
        layer.keys[..., :length, :].float() * layer.key_scales[..., :length, :]
    ).repeat_interleave(4, dim=1)
    dense_value = (
        layer.values[..., :length, :].float() * layer.value_scales[..., :length, :]
    ).repeat_interleave(
        4, dim=1
    )
    reference_weights = torch.softmax(
        query.float() @ dense_key.transpose(2, 3) * 0.125 + mask,
        dim=-1,
    )
    reference = (reference_weights @ dense_value).transpose(1, 2)

    assert output.shape == (1, 3, 8, 64)
    assert weights.shape == (1, 8, 3, 3)
    error = output.float() - reference
    assert error.square().mean().item() < 3e-4
    assert error.abs().max().item() < 0.15

    decode_query = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
    decode_key = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.bfloat16)
    decode_value = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.bfloat16)
    key_view, value_view = cache.update(decode_key, decode_value, 0)
    decode_output, _ = qvq_fp8_attention_forward(
        SimpleNamespace(num_key_value_groups=4, training=False),
        decode_query,
        key_view,
        value_view,
        None,
        scaling=0.125,
    )
    length = layer.get_seq_length()
    dense_key = (
        layer.keys[..., :length, :].float() * layer.key_scales[..., :length, :]
    ).repeat_interleave(4, dim=1)
    dense_value = (
        layer.values[..., :length, :].float() * layer.value_scales[..., :length, :]
    ).repeat_interleave(
        4, dim=1
    )
    decode_weights = torch.softmax(
        decode_query.float() @ dense_key.transpose(2, 3) * 0.125,
        dim=-1,
    )
    decode_reference = (decode_weights @ dense_value).transpose(1, 2)
    decode_error = decode_output.float() - decode_reference
    assert decode_error.square().mean().item() < 3e-4
    assert decode_error.abs().max().item() < 0.15

    telemetry = cache.telemetry()
    assert telemetry["native_fp8_attention"] is True
    assert telemetry["native_attention_calls"] == 2
    assert telemetry["native_grouped_attention_calls"] == 2
    assert telemetry["native_qk_fp8_mm_calls"] == 4
    assert telemetry["native_pv_fp8_mm_calls"] == 4
    assert telemetry["native_qk_fp8_launches"] == 2
    assert telemetry["native_pv_fp8_launches"] == 2
    assert telemetry["dequantized_elements"] == 0
    assert telemetry["dense_kv_prefix_materializations"] == 0
    assert telemetry["capacities"] == [16]
    assert telemetry["allocations"] == 1
    assert telemetry["reallocations"] == 0
    assert layer.values.stride(-2) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_h200_fp8_attention_preserves_left_padding_and_sliding_window_masks():
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        pytest.skip("native QVQ FP8 attention validation requires the assigned H200")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260905)
    query = torch.randn(
        (2, 8, 4, 64), generator=generator, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        (2, 2, 4, 64), generator=generator, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        (2, 2, 4, 64), generator=generator, device=device, dtype=torch.bfloat16
    )
    config = _TinyCacheConfig()
    config._attn_implementation = "qvq_fp8"
    config.sliding_window = 2
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "qvq_fp8", ALL_MASK_ATTENTION_FUNCTIONS["eager"]
    )
    padding = torch.tensor(
        [[0, 0, 1, 1], [1, 1, 1, 1]], device=device, dtype=torch.long
    )
    additive = create_causal_mask(
        config,
        torch.zeros((2, 4, 8), device=device, dtype=torch.bfloat16),
        padding,
        past_key_values=None,
    )
    cache = QVQFP8DynamicCache(config, QVQActivationConfig())
    key_view, value_view = cache.update(key, value, 0)
    module = SimpleNamespace(num_key_value_groups=4, training=False)
    batched, _ = qvq_fp8_attention_forward(
        module,
        query,
        key_view,
        value_view,
        additive,
        scaling=0.125,
    )

    first_cache = QVQFP8DynamicCache(config, QVQActivationConfig())
    first_key, first_value = first_cache.update(
        key[:1, :, 2:], value[:1, :, 2:], 0
    )
    first, _ = qvq_fp8_attention_forward(
        module,
        query[:1, :, 2:],
        first_key,
        first_value,
        None,
        scaling=0.125,
    )
    second_cache = QVQFP8DynamicCache(config, QVQActivationConfig())
    second_key, second_value = second_cache.update(key[1:], value[1:], 0)
    second, _ = qvq_fp8_attention_forward(
        module,
        query[1:],
        second_key,
        second_value,
        None,
        scaling=0.125,
    )
    torch.testing.assert_close(batched[:1, 2:], first, rtol=0, atol=2e-2)
    torch.testing.assert_close(batched[1:], second, rtol=0, atol=2e-2)

    sliding = create_sliding_window_causal_mask(
        config,
        torch.zeros((2, 4, 8), device=device, dtype=torch.bfloat16),
        torch.ones_like(padding),
        past_key_values=None,
    )
    _, weights = qvq_fp8_attention_forward(
        module,
        query,
        key_view,
        value_view,
        sliding,
        scaling=0.125,
        output_attentions=True,
    )
    assert torch.count_nonzero(weights[:, :, 3, :2]).item() == 0
    assert bool((weights[:, :, 3, 2:].sum(dim=-1) > 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_h200_fp8_attention_chunks_large_prefill_without_dense_kv(monkeypatch):
    capability = torch.cuda.get_device_capability()
    if capability[0] != 9:
        pytest.skip("Hopper is required")

    import gptqmodel.nn_modules.qvq_fp8_cache as fp8_cache

    monkeypatch.setattr(fp8_cache, "_QVQ_FP8_PREFILL_QUERY_CHUNK", 16)
    torch.manual_seed(23)
    cache = QVQFP8DynamicCache(_TinyCacheConfig(), QVQActivationConfig())
    query = torch.randn(1, 8, 33, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 33, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 2, 33, 64, device="cuda", dtype=torch.bfloat16)
    key_view, value_view = cache.update(key, value, 0)

    output, weights = qvq_fp8_attention_forward(
        SimpleNamespace(num_key_value_groups=4, training=False),
        query,
        key_view,
        value_view,
        None,
        scaling=0.125,
        output_attentions=True,
    )

    layer = cache.layers[0]
    dense_key = (
        layer.keys[..., :33, :].float() * layer.key_scales[..., :33, :]
    ).repeat_interleave(4, dim=1)
    dense_value = (
        layer.values[..., :33, :].float() * layer.value_scales[..., :33, :]
    ).repeat_interleave(4, dim=1)
    causal_mask = torch.full((33, 33), float("-inf"), device="cuda")
    causal_mask = torch.triu(causal_mask, diagonal=1)
    reference_weights = torch.softmax(
        query.float() @ dense_key.transpose(2, 3) * 0.125 + causal_mask,
        dim=-1,
    )
    reference = (reference_weights @ dense_value).transpose(1, 2)

    assert output.shape == (1, 33, 8, 64)
    assert weights.shape == (1, 8, 33, 33)
    error = output.float() - reference
    assert error.square().mean().item() < 3e-4
    assert error.abs().max().item() < 0.15

    telemetry = cache.telemetry()
    expected_launches = 2 * 3
    assert telemetry["native_qk_fp8_launches"] == expected_launches
    assert telemetry["native_pv_fp8_launches"] == expected_launches
    assert telemetry["dequantized_elements"] == 0
    assert telemetry["dense_kv_prefix_materializations"] == 0


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
def test_qvq_v2b2_g32_a8_config_round_trip(bits):
    config = QVQConfig(
        bits=bits,
        format="v2b2-g32",
        rounding="block_ldlq",
        activation=True,
        offload_to_disk=False,
    )

    assert config.format == FORMAT.QVQ_V2B2_P32
    assert config.bank_count == 2
    assert config.activation == QVQActivationConfig()
    assert "activation_quantization" not in config.to_dict()
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.activation == config.activation
    assert reloaded.quant_linear_init_kwargs()["activation"] == {
        "bits": 8,
        "format": "float8_e4m3fn",
        "kernel_mode": "auto",
        "replay_max_rows": 2048,
        "replay_passes": 1,
        "replay_validation_fraction": 0.125,
        "scale_method": "dynamic_per_token",
        "target": "p32_operand",
    }
    legacy = config.to_dict()
    legacy["activation_quantization"] = legacy.pop("activation")
    assert QVQConfig.from_quant_config(legacy).activation == config.activation
    with pytest.raises(ValueError, match="both `activation` and legacy"):
        QVQConfig.from_quant_config({**config.to_dict(), "activation_quantization": {}})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_h200_fp8_replay_reencodes_from_immutable_dense_teacher_and_executes_native_kernel():
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        pytest.skip("true QVQ P32 FP8 replay validation requires the assigned H200")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260952)
    linear = torch.nn.Linear(32, 64, bias=True, device=device, dtype=torch.float16).eval()
    with torch.no_grad():
        linear.weight.copy_(
            torch.randn(linear.weight.shape, generator=generator, device=device, dtype=torch.float16) * 0.02
        )
        linear.bias.copy_(
            torch.randn(linear.bias.shape, generator=generator, device=device, dtype=torch.float16) * 0.01
        )
    calibration_input = torch.randn(
        (16, 32),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    with torch.inference_mode():
        native_teacher = linear(calibration_input)
    canonical_weight = linear.weight.detach().clone()
    immutable_weight = canonical_weight.clone()
    source_hessian = calibration_input.float().t() @ calibration_input.float()
    quantization_kwargs = {
        "bias": linear.bias.detach(),
        "vector_size": 2,
        "trellis_window": 16,
        "bank_count": 2,
        "v2b2_p32": True,
        "rounding": "block_ldlq",
        "trellis_batch_size": 1,
        "input_hadamard": True,
        "output_hadamard": True,
    }
    first_result = quantize_qvq_linear(
        canonical_weight,
        source_hessian,
        bits=3.5,
        **quantization_kwargs,
    )
    qcfg = QVQConfig(
        bits=3.5,
        format="v2b2-g32",
        rounding="block_ldlq",
        activation={
            "kernel_mode": "require",
            "replay_max_rows": 16,
            "replay_validation_fraction": 0.125,
        },
        offload_to_disk=False,
    )
    named = NamedModule(linear, name="proj", full_name="proj", layer_index=0)
    task_entry = {
        "fp8_replay_rows": [(calibration_input, native_teacher)],
        "fp8_replay_stats": None,
    }
    processor = object.__new__(QVQProcessor)

    selected = processor._fp8_replay_reencode(
        named,
        qcfg,
        canonical_weight,
        source_hessian,
        quantization_kwargs,
        first_result,
        task_entry,
    )

    assert torch.equal(canonical_weight, immutable_weight)
    assert torch.equal(selected.SU, first_result.SU)
    assert torch.equal(selected.SV, first_result.SV)
    stats = task_entry["fp8_replay_stats"]
    assert stats["schema"] == "qvq.fp8-target-replay.v1"
    assert stats["source"] == "immutable_original_dense_weight"
    assert stats["rows"] == 16
    assert stats["train_rows"] == 14
    assert stats["validation_rows"] == 2
    assert stats["ridge_prior"] == "immutable_original_dense_inner_target"
    assert stats["native_first_executed"] == 1
    assert stats["native_second_executed"] == 1
    assert selected.telemetry["fp8_target_replay"] == stats


def test_qvq_a8_rejects_incompatible_weight_formats_and_rates():
    with pytest.raises(ValueError, match="requires `format=qvq_v2b2_p32`"):
        QVQConfig(
            bits=2,
            format="qvq",
            rounding="block_ldlq",
            activation=True,
            offload_to_disk=False,
        )
    with pytest.raises(ValueError, match="W2 through W3.5"):
        QVQConfig(
            bits=1.5,
            format="qvq_v2b2_p32",
            rounding="block_ldlq",
            activation=True,
            offload_to_disk=False,
        )


def test_qvq_v2b2_g32_alias_normalizes_in_dynamic_overrides():
    config = QVQConfig(
        bits=2,
        format="v2b2-g32",
        rounding="block_ldlq",
        activation=True,
        dynamic={"model.layers.0.*": {"format": "v2b2-g32", "bits": 3}},
        offload_to_disk=False,
    )

    assert config.dynamic["model.layers.0.*"]["format"] == FORMAT.QVQ_V2B2_P32.value


def test_qvq_processor_accumulates_hessian_on_dequantized_linear_input_a8_values():
    torch.manual_seed(7)
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    config = QVQConfig(
        bits=2,
        format="qvq_v2b2_p32",
        rounding="block_ldlq",
        activation={"target": "linear_input"},
        device="cpu",
        offload_to_disk=False,
    )
    processor = _processor(config)
    processor.preprocess(named)
    source = torch.randn((1, 4, 16)) * 3.0
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.tensor([[True, True, True, False]])
    processor._set_current_batch_index(0)

    processor.pre_process_fwd_hook("proj")(root.proj, (source,), root.proj(source))

    capture = processor.tasks["proj"]["capture"]
    _, _, expected_source = fake_quantize_qvq_fp8_activation(source[:, :3])
    expected = capture.compute_hessian_xtx(expected_source.reshape(-1, 16))
    actual = capture._device_hessian_partials[torch.device("cpu")]
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    stats = QVQProcessor._activation_quantization_error_summary(processor.tasks["proj"])
    assert stats["format"] == "float8_e4m3fn"
    assert stats["elements"] == 48
    assert stats["relative_rmse"] > 0.0


def test_qvq_fp8_replay_teacher_capture_survives_completed_pristine_hessian():
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    config = QVQConfig(
        bits=2,
        format="qvq_v2b2_p32",
        rounding="block_ldlq",
        activation={"replay_max_rows": 16},
        device="cpu",
        offload_to_disk=False,
    )
    processor = _processor(config)
    processor.preprocess(named)
    task_entry = processor.tasks["proj"]
    task_entry["pristine_hessian_complete"] = True
    source = torch.randn((1, 4, 16))
    native_output = root.proj(source)

    processor.pre_process_fwd_hook("proj")(root.proj, (source,), native_output)

    assert task_entry["fp8_replay_row_count"] == 4
    assert len(task_entry["fp8_replay_rows"]) == 1
    captured_source, captured_output = task_entry["fp8_replay_rows"][0]
    assert torch.equal(captured_source, source.reshape(-1, 16))
    assert torch.equal(captured_output, native_output.reshape(-1, 16))
    assert not task_entry["capture"]._device_hessian_partials


def test_qvq_processor_merges_device_local_activation_error_statistics():
    accumulated = {
        "elements": 4,
        "source_square_sum": torch.tensor(20.0),
        "error_square_sum": torch.tensor(2.0),
        "maximum_absolute_error": torch.tensor(0.75),
        "minimum_scale": torch.tensor(0.25),
        "maximum_scale": torch.tensor(1.0),
        "finite": torch.tensor(True),
    }
    stats = QVQProcessor._activation_quantization_error_summary(
        {
            "activation_quantization_error": {
                "replica-0": accumulated,
                "replica-1": {
                    **accumulated,
                    "elements": 6,
                    "source_square_sum": torch.tensor(30.0),
                    "error_square_sum": torch.tensor(3.0),
                    "maximum_absolute_error": torch.tensor(0.5),
                    "minimum_scale": torch.tensor(0.5),
                    "maximum_scale": torch.tensor(2.0),
                },
            },
            "qcfg": SimpleNamespace(activation=QVQActivationConfig(target="linear_input")),
        }
    )

    assert stats["elements"] == 10
    assert stats["rmse"] == pytest.approx(5.0**0.5 / 10.0**0.5)
    assert stats["relative_rmse"] == pytest.approx(0.1**0.5)
    assert stats["maximum_absolute_error"] == 0.75
    assert stats["minimum_scale"] == 0.25
    assert stats["maximum_scale"] == 2.0


def test_auto_loader_accepts_only_the_qvq_a8_contract():
    payload = QVQConfig(
        bits=2,
        format="qvq_v2b2_p32",
        rounding="block_ldlq",
        activation=True,
        offload_to_disk=False,
    ).to_dict()
    assert _activation_quantization_mode(payload) is None
    assert (
        _is_supported_quantization_config(SimpleNamespace(quantization_config=payload))
        is True
    )

    invalid = dict(payload)
    invalid["activation"] = {
        "bits": 4,
        "format": "float",
        "scale_method": "dynamic",
    }
    assert _activation_quantization_mode(invalid) == "activation"

    with_kv_cache = dict(payload, kv_cache_scheme={"num_bits": 8, "type": "float"})
    assert _activation_quantization_mode(with_kv_cache) == "kv_cache_scheme"


def test_yaqa_collects_fisher_factors_under_the_a8_forward_contract():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(16, 16, bias=False)

        def forward(self, hidden_states):
            return self.proj(hidden_states).tanh()

    class TinyCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(32, 16)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList((Layer(),))
            self.head = torch.nn.Linear(16, 32, bias=False)

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            hidden_states = self.embed(input_ids)
            hidden_states = self.model.layers[0](hidden_states)
            return SimpleNamespace(logits=self.head(hidden_states))

    torch.manual_seed(19)
    model = TinyCausalModel().eval()
    module = model.model.layers[0].proj
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }

    input_hessians, output_hessians, stats = capture_yaqa_sketch_b(
        model,
        [batch],
        {"proj": module},
        device=torch.device("cpu"),
        seed=23,
        activation=QVQActivationConfig(target="linear_input"),
        activation_modules={"proj": module},
    )

    assert torch.isfinite(input_hessians["proj"]).all()
    assert torch.isfinite(output_hessians["proj"]).all()
    activation_stats = stats["activation_quantization_error"]
    assert activation_stats["format"] == "float8_e4m3fn"
    assert activation_stats["elements"] == 64
    assert activation_stats["relative_rmse"] > 0.0
    assert set(activation_stats["modules"]) == {"proj"}
    assert not module._forward_pre_hooks


@pytest.mark.parametrize(
    "activation",
    (QVQActivationConfig(), {"bits": 8, "target": "p32_operand"}),
)
def test_yaqa_rejects_pretransform_hooks_for_p32_operand_target(activation):
    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False)).eval()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    with pytest.raises(ValueError, match="post-SU/Hadamard collector"):
        capture_yaqa_sketch_b(
            model,
            [batch],
            {"proj": model[0]},
            device=torch.device("cpu"),
            first_decoder_layer=model[0],
            activation=activation,
            activation_modules={"proj": model[0]},
        )


def test_qvq_config_rejects_yaqa_p32_operand_but_accepts_linear_input():
    with pytest.raises(ValueError, match="post-SU/Hadamard Sketch-B collector"):
        QVQConfig(
            bits=3.5,
            format="v2b2-g32",
            rounding="yaqa",
            activation=True,
            offload_to_disk=False,
        )
    config = QVQConfig(
        bits=3.5,
        format="v2b2-g32",
        rounding="yaqa",
        activation={"target": "linear_input"},
        offload_to_disk=False,
    )
    assert config.activation.target == "linear_input"


def _p32_a8_layer(kernel_mode: str) -> QVQLinear:
    bits = 3.5
    in_features = 32
    out_features = 64
    tile_count = (in_features // 16) * (out_features // 16)
    tensors = {
        "trellis": torch.zeros((tile_count, int(8 * bits)), dtype=torch.int32),
        "SU": torch.linspace(0.75, 1.25, in_features),
        "SV": torch.linspace(0.5, 1.0, out_features),
        "bank_ids": pack_qvq_binary_bank_ids(
            torch.zeros(tile_count * 8, dtype=torch.uint8)
        ),
        "bank_alt_id": torch.tensor([1], dtype=torch.uint8),
    }
    return QVQLinear.from_tensors(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        name="proj",
        tensors=tensors,
        bank_count=2,
        v2b2_p32=True,
        activation={"kernel_mode": kernel_mode},
    ).eval()


def test_qvq_fp8_fallback_uses_the_deployed_e4m3_weight_levels():
    layer = _p32_a8_layer("disable")
    source = torch.randn((3, 32), generator=torch.Generator().manual_seed(20260905))
    transformed = layer.transform_input(source)
    quantized, scale = quantize_qvq_fp8_activation(
        transformed,
        format=layer.activation.format,
        scale_method=layer.activation.scale_method,
    )

    actual = layer.forward_prequantized_fp8(
        quantized,
        scale,
        output_dtype=source.dtype,
    )
    fp8_levels, level_scale = layer._prepare_hopper_fp8_levels(source.device)
    canonical_inner = reconstruct_qvq_inner_weight(
        layer.trellis,
        bits=layer.bits,
        in_features=layer.in_features,
        out_features=layer.out_features,
        bank_ids=layer.bank_ids,
        v2b2_p32=True,
        bank_alt_id=layer.bank_alt_id,
    )
    canonical_levels = pgc16_levels_for_version(layer.codebook_version)
    deployed_inner = (fp8_levels.float() * level_scale)[
        torch.searchsorted(canonical_levels, canonical_inner)
    ]
    dequantized = dequantize_qvq_fp8_activation(
        quantized, scale, dtype=torch.float32
    )
    expected = layer.recover_output(
        dequantized @ deployed_inner,
        output_dtype=source.dtype,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        layer(source),
        qvq_dense_oracle_forward(layer, source),
        rtol=0,
        atol=2e-5,
    )

    auto = _p32_a8_layer("auto")
    auto.load_state_dict(layer.state_dict(), strict=True)
    with patch.object(
        QVQLinear,
        "_fp8_kernel_ineligible_reason",
        return_value="forced_test_fallback",
    ):
        auto_actual = auto.forward_prequantized_fp8(
            quantized,
            scale,
            output_dtype=source.dtype,
        )
    torch.testing.assert_close(auto_actual, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_h200_p32_a8_preserves_finite_bf16_range_and_matches_deployed_fallback():
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        pytest.skip("native QVQ P32 FP8 validation requires the assigned H200")

    native = _p32_a8_layer("require").to("cuda")
    fallback = _p32_a8_layer("disable").to("cuda")
    fallback.load_state_dict(native.state_dict(), strict=True)
    source = torch.empty((2, 32), device="cuda", dtype=torch.bfloat16)
    source[0].fill_(65536.0)
    source[1, ::2] = 65536.0
    source[1, 1::2] = -65536.0

    transformed = native._qvq_prepare_inference_input(source, torch.float32)
    assert transformed.dtype == torch.float32
    assert torch.isfinite(transformed).all()
    with torch.inference_mode():
        actual = native(source)
        expected = fallback(source)

    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=256.0)
    telemetry = native.qvq_fp8_kernel_telemetry()
    assert telemetry["executed"] == 1
    assert telemetry["fallback"] == 0
