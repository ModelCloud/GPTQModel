# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.models.auto import (
    _activation_quantization_mode,
    _is_supported_quantization_config,
)
from gptqmodel.nn_modules.qvq_fp8_cache import (
    QVQFP8DynamicCache,
    install_qvq_fp8_kv_cache,
    qvq_fp8_attention_forward,
)
from gptqmodel.quantization import FORMAT, QVQActivationConfig, QVQConfig
from gptqmodel.quantization.qvq import quantize_qvq_linear
from gptqmodel.quantization.qvq_activation import (
    dequantize_qvq_fp8_activation,
    fake_quantize_qvq_fp8_activation,
    quantize_qvq_fp8_activation,
)
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
    torch.testing.assert_close(returned_key, key, atol=0.0, rtol=0.065)
    torch.testing.assert_close(returned_value, value, atol=0.0, rtol=0.065)

    telemetry = cache.telemetry()
    expected_payload_bytes = layer.keys.numel() + layer.values.numel()
    expected_scale_bytes = (layer.key_scales.numel() + layer.value_scales.numel()) * 4
    assert telemetry["all_payloads_fp8"] is True
    assert telemetry["no_full_precision_residual"] is True
    assert telemetry["storage_bytes"] == expected_payload_bytes + expected_scale_bytes
    assert telemetry["storage_ratio_vs_dense"] == pytest.approx(0.625)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("seed", (17, 29, 41))
def test_h200_fp8_attention_consumes_cache_without_dense_prefix_materialization(seed):
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        pytest.skip("native QVQ FP8 attention validation requires the assigned H200")

    torch.manual_seed(seed)
    cache = QVQFP8DynamicCache(_TinyCacheConfig(), QVQActivationConfig())
    query = torch.randn(1, 4, 3, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 64, device="cuda", dtype=torch.bfloat16)
    key_view, value_view = cache.update(key, value, 0)
    mask = torch.full((1, 1, 3, 3), float("-inf"), device="cuda")
    mask = torch.triu(mask, diagonal=1)

    output, weights = qvq_fp8_attention_forward(
        SimpleNamespace(num_key_value_groups=4, training=False),
        query,
        key_view,
        value_view,
        mask,
        scaling=0.125,
    )
    layer = cache.layers[0]
    dense_key = (layer.keys.float() * layer.key_scales).repeat_interleave(4, dim=1)
    dense_value = (layer.values.float() * layer.value_scales).repeat_interleave(
        4, dim=1
    )
    reference_weights = torch.softmax(
        query.float() @ dense_key.transpose(2, 3) * 0.125 + mask,
        dim=-1,
    )
    reference = (reference_weights @ dense_value).transpose(1, 2)

    assert output.shape == (1, 3, 4, 64)
    assert weights.shape == (1, 4, 3, 3)
    error = output.float() - reference
    assert error.square().mean().item() < 3e-4
    assert error.abs().max().item() < 0.1
    telemetry = cache.telemetry()
    assert telemetry["native_fp8_attention"] is True
    assert telemetry["native_attention_calls"] == 1
    assert telemetry["native_qk_fp8_mm_calls"] == 4
    assert telemetry["native_pv_fp8_mm_calls"] == 4
    assert telemetry["dequantized_elements"] == 0
    assert telemetry["dense_kv_prefix_materializations"] == 0


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
def test_qvq_v2b2_g32_a8_config_round_trip(bits):
    config = QVQConfig(
        bits=bits,
        format="v2b2-g32",
        rounding="block_ldlq",
        activation_quantization=True,
        offload_to_disk=False,
    )

    assert config.format == FORMAT.QVQ_V2B2_P32
    assert config.bank_count == 2
    assert config.activation_quantization == QVQActivationConfig()
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.activation_quantization == config.activation_quantization
    assert reloaded.quant_linear_init_kwargs()["activation_quantization"] == {
        "bits": 8,
        "format": "float8_e4m3fn",
        "kernel_mode": "auto",
        "replay_max_rows": 2048,
        "replay_passes": 1,
        "replay_validation_fraction": 0.125,
        "scale_method": "dynamic_per_token",
        "target": "p32_operand",
    }


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
        activation_quantization={
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
            activation_quantization=True,
            offload_to_disk=False,
        )
    with pytest.raises(ValueError, match="W2 through W3.5"):
        QVQConfig(
            bits=1.5,
            format="qvq_v2b2_p32",
            rounding="block_ldlq",
            activation_quantization=True,
            offload_to_disk=False,
        )


def test_qvq_v2b2_g32_alias_normalizes_in_dynamic_overrides():
    config = QVQConfig(
        bits=2,
        format="v2b2-g32",
        rounding="block_ldlq",
        activation_quantization=True,
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
        activation_quantization={"target": "linear_input"},
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
        activation_quantization={"replay_max_rows": 16},
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
            "qcfg": SimpleNamespace(
                activation_quantization=QVQActivationConfig(target="linear_input")
            ),
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
        activation_quantization=True,
        offload_to_disk=False,
    ).to_dict()
    assert _activation_quantization_mode(payload) is None
    assert (
        _is_supported_quantization_config(SimpleNamespace(quantization_config=payload))
        is True
    )

    invalid = dict(payload)
    invalid["activation_quantization"] = {
        "bits": 4,
        "format": "float",
        "scale_method": "dynamic",
    }
    assert _activation_quantization_mode(invalid) == "activation_quantization"

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
        activation_quantization=QVQActivationConfig(target="linear_input"),
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
