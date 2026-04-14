# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch
from torch import nn

import gptqmodel.looper.awq_processor as awq_processor_module
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.input_activations import (
    calibrate_input_scale_inv,
    quantize_dequantize_input,
)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="Current PyTorch build does not provide FP8 dtypes.")
class TestAwqInputActivations:
    class _RuntimeTestAWQProcessor(AWQProcessor):
        """Small AWQ harness that exercises calibration/replay code without a full model."""

        def __init__(self, qcfg: QuantizeConfig):
            model = nn.Sequential()
            super().__init__(
                tokenizer=None,
                qcfg=qcfg,
                calibration=None,
                prepare_dataset_func=None,
                calibration_concat_size=None,
                calibration_sort=None,
                batch_size=1,
                gptq_model=types.SimpleNamespace(
                    model=model,
                    lm_head="lm_head",
                    qlinear_kernel=AwqTorchLinear,
                    rotary_embedding=None,
                    quant_region_timer=None,
                ),
                model=model,
                require_fwd=True,
                calculate_w_wq_diff=False,
                calibration_concat_separator=None,
            )

    @staticmethod
    def _input_activations_config() -> dict:
        return {
            "num_bits": 8,
            "type": "float",
            "format": "float8_e4m3fn",
            "strategy": "tensor",
            "dynamic": True,
            "implementation": "reference",
        }

    def test_awq_search_best_scale_uses_quantized_input_activations(self, monkeypatch):
        qcfg = QuantizeConfig(
            quant_method=METHOD.AWQ,
            format=FORMAT.GEMM,
            bits=4,
            group_size=16,
            sym=False,
            input_activations=self._input_activations_config(),
        )
        processor = self._RuntimeTestAWQProcessor(qcfg)

        parent = nn.Module()
        parent.prev = nn.LayerNorm(16, elementwise_affine=False, dtype=torch.float16)
        parent.linear = nn.Linear(16, 16, bias=False, dtype=torch.float16)

        x = torch.randn(2, 3, 16, dtype=torch.float16)
        expected = processor.quantize_dequantize_input(x.to(parent.linear.weight.device))
        expected_mean = expected.abs().view(-1, expected.shape[-1]).to(torch.float32).mean(dim=0).to(expected.dtype)
        expected_output = parent.linear(expected)
        observed = {}

        def fake_compute_best_scale(self, x_arg, w_mean, x_mean, module2inspect, linears2scale, fp16_output, kwargs):
            observed["x"] = x_arg.detach().clone()
            observed["x_mean"] = x_mean.detach().clone()
            observed["fp16_output"] = fp16_output.detach().clone()
            return torch.ones(x_arg.shape[-1], dtype=torch.float32), 0.0

        monkeypatch.setattr(AWQProcessor, "_compute_best_scale", fake_compute_best_scale)

        AWQProcessor._search_best_scale(
            processor,
            module=parent,
            prev_op=parent.prev,
            layers=[parent.linear],
            inp=x,
            module2inspect=parent.linear,
            kwargs={},
        )

        torch.testing.assert_close(observed["x"], expected, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(observed["x_mean"], expected_mean, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(observed["fp16_output"], expected_output, atol=1e-3, rtol=1e-3)

    def test_awq_static_activation_path_caches_calibrated_input_scale(self, monkeypatch):
        qcfg = QuantizeConfig(
            bits=4,
            group_size=16,
            sym=False,
            activation={
                "method": "fp8",
                "format": "f8_e4m3",
            },
        )
        processor = self._RuntimeTestAWQProcessor(qcfg)

        first = torch.randn(2, 3, 16, dtype=torch.float16)
        second = torch.randn(2, 3, 16, dtype=torch.float16) * 2.0
        expected_scale_inv = calibrate_input_scale_inv(first, qcfg.activation)
        expected_second = quantize_dequantize_input(second, qcfg.activation, scale_inv=expected_scale_inv)

        observed_calls = {"count": 0}
        original = awq_processor_module.calibrate_input_scale_inv

        def wrapped_calibrate(*args, **kwargs):
            observed_calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(awq_processor_module, "calibrate_input_scale_inv", wrapped_calibrate)

        processor.quantize_dequantize_input(first, module_names=["linear"])
        actual_second = processor.quantize_dequantize_input(second, module_names=["linear"])

        assert observed_calls["count"] == 1
        assert "linear" in processor._activation_scale_inv_by_module
        torch.testing.assert_close(
            processor._activation_scale_inv_by_module["linear"],
            expected_scale_inv.detach().to(device="cpu", dtype=torch.float32).reshape(()),
        )
        torch.testing.assert_close(actual_second, expected_second, atol=1e-3, rtol=1e-3)

    def test_awq_torch_linear_applies_runtime_input_activation_qdq(self):
        qcfg = QuantizeConfig(
            quant_method=METHOD.AWQ,
            format=FORMAT.GEMM,
            bits=4,
            group_size=16,
            sym=False,
        )
        processor = self._RuntimeTestAWQProcessor(qcfg)

        linear = nn.Linear(16, 32, bias=True, dtype=torch.float16)
        _, scales, zeros = processor.pseudo_quantize_tensor(linear.weight.detach().clone())

        baseline = AwqTorchLinear(
            bits=4,
            group_size=16,
            sym=False,
            desc_act=False,
            in_features=16,
            out_features=32,
            bias=True,
        )
        baseline.pack(linear, scales, zeros)

        candidate = AwqTorchLinear(
            bits=4,
            group_size=16,
            sym=False,
            desc_act=False,
            in_features=16,
            out_features=32,
            bias=True,
            input_activations=self._input_activations_config(),
        )
        candidate.pack(linear, scales, zeros)

        x = torch.randn(4, 16, dtype=torch.float16)
        qdq_x = candidate.quantize_dequantize_input(x)

        torch.testing.assert_close(candidate(x), baseline(qdq_x), atol=1e-3, rtol=1e-3)

    def test_awq_pack_module_passes_input_activation_init_kwargs(self, monkeypatch):
        qcfg = QuantizeConfig(
            quant_method=METHOD.AWQ,
            format=FORMAT.GEMM,
            bits=4,
            group_size=16,
            sym=False,
            input_activations=self._input_activations_config(),
        )
        processor = self._RuntimeTestAWQProcessor(qcfg)

        created = {}

        def fake_create_quant_module(*args, **kwargs):
            created["kwargs"] = kwargs

        def fake_pack_module(*args, **kwargs):
            return "fake_pack"

        monkeypatch.setattr(awq_processor_module, "create_quant_module", fake_create_quant_module)
        monkeypatch.setattr(awq_processor_module, "pack_module", fake_pack_module)

        class _FakeNamedModule:
            """Minimal named-module stub for AWQ pack tests."""

            def __init__(self):
                self.name = "linear"
                self.full_name = "linear"
                self.layer_index = 0
                self.module = nn.Linear(16, 8, bias=True, dtype=torch.float16)
                self.weight = self.module.weight
                self.state = {
                    "q_zeros": torch.zeros(8, 1, dtype=torch.float16),
                    "q_scales": torch.ones(8, 1, dtype=torch.float16),
                }

            def stream_sync(self):
                return None

        processor.pack_module(_FakeNamedModule())

        assert "kwargs" in created
        assert created["kwargs"]["init_kwargs"]["input_activations"]["format"] == "float8_e4m3fn"

    def test_awq_pack_module_sets_calibrated_static_input_scale(self, monkeypatch):
        qcfg = QuantizeConfig(
            bits=4,
            group_size=16,
            sym=False,
            activation={
                "method": "fp8",
                "format": "f8_e4m3",
            },
        )
        processor = self._RuntimeTestAWQProcessor(qcfg)
        processor._activation_scale_inv_by_module["linear"] = torch.tensor(321.0, dtype=torch.float32)

        created = {}
        packed = {}

        def fake_create_quant_module(*args, **kwargs):
            created["kwargs"] = kwargs

        def fake_pack_module(*args, **kwargs):
            packed["called"] = True
            return "fake_pack"

        class _FakeQuantModule:
            """Track the calibrated static activation scale passed into the packed module."""

            def __init__(self):
                self.input_scale_inv = None

            def set_input_scale_inv(self, scale_inv):
                self.input_scale_inv = scale_inv.detach().clone()

        fake_qmodule = _FakeQuantModule()

        def fake_find_modules(*args, **kwargs):
            return {"linear": fake_qmodule}

        monkeypatch.setattr(awq_processor_module, "create_quant_module", fake_create_quant_module)
        monkeypatch.setattr(awq_processor_module, "pack_module", fake_pack_module)
        monkeypatch.setattr(awq_processor_module, "find_modules", fake_find_modules)

        class _FakeNamedModule:
            """Minimal named-module stub for AWQ static activation pack tests."""

            def __init__(self):
                self.name = "linear"
                self.full_name = "linear"
                self.layer_index = 0
                self.module = nn.Linear(16, 8, bias=True, dtype=torch.float16)
                self.weight = self.module.weight
                self.state = {
                    "q_zeros": torch.zeros(8, 1, dtype=torch.float16),
                    "q_scales": torch.ones(8, 1, dtype=torch.float16),
                }

            def stream_sync(self):
                return None

        processor.pack_module(_FakeNamedModule())

        assert "kwargs" in created
        assert packed["called"] is True
        torch.testing.assert_close(fake_qmodule.input_scale_inv, torch.tensor(321.0, dtype=torch.float32))
