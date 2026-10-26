# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent Torch-oracle checks for EXL3's MLX quantization pipeline."""

import gc
import sys
import threading

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_codebook import _torch_codebook_oracle
from tests.test_mlx_exl3_gss import (
    _torch_global_scale_search_oracle,
    _torch_gss_sample_oracle,
)
from tests.test_mlx_exl3_hadamard import _torch_hadamard_oracle
from tests.test_mlx_exl3_hessian import _torch_finalize_hessian_oracle
from tests.test_mlx_exl3_ldlq import _normalized_drift, _torch_ldlq_oracle
from tests.test_mlx_exl3_quantization import _torch_pack_trellis_oracle
from tests.test_mlx_exl3_regularize import (
    _assert_composed_weight_matches,
    _torch_regularize_transforms_oracle,
)
from tests.test_mlx_exl3_viterbi import _half_rounding_boundary


if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.looper import exllamav3_processor as processor_module  # noqa: E402
from gptqmodel.looper.exllamav3_processor import EXL3Processor  # noqa: E402
from gptqmodel.looper.named_module import NamedModule  # noqa: E402
from gptqmodel.quantization.config import EXL3Config  # noqa: E402
from gptqmodel.quantization.mlx_exl3_pipeline import (  # noqa: E402
    exl3_quantize_weight_mlx,
    exl3_quantize_weight_mlx_to_torch,
)


def _signs(size, seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        (torch.randn(size, generator=generator).sign() + 1e-5)
        .sign()
        .numpy()
        .astype(np.float32)
    )


def _pipeline_oracle(
    weight,
    hessian,
    input_signs,
    output_signs,
    *,
    sample_count,
    bits,
    codebook,
    force_output_scales=None,
):
    if sample_count and not np.any(hessian):
        fallback = True
        transformed_hessian = None
        factor = None
        diagonal = np.zeros(hessian.shape[0], dtype=np.float32)
    else:
        fallback, transformed_hessian, factor, diagonal_tensor = (
            _torch_finalize_hessian_oracle(
                hessian,
                input_signs[:, 0],
                sample_count=sample_count,
            )
        )
        diagonal = diagonal_tensor.numpy()
    apply_output_scales, regularized, input_scales, output_scales = (
        _torch_regularize_transforms_oracle(
            weight,
            input_signs,
            output_signs,
            force_output_scales=force_output_scales,
            hessian_diagonal=diagonal,
            fallback=fallback,
        )
    )
    global_scale, _ = _torch_global_scale_search_oracle(
        _torch_gss_sample_oracle(regularized),
        bits=bits,
        codebook=codebook,
    )
    regularized = np.ascontiguousarray(
        regularized * np.float32(global_scale), dtype=np.float32
    )
    input_scales = np.ascontiguousarray(
        input_scales / np.float32(global_scale), dtype=np.float32
    )
    if fallback:
        from tests.test_mlx_exl3_fallback import _torch_fallback_quantize_oracle

        quantized, encoded = _torch_fallback_quantize_oracle(
            regularized,
            bits=bits,
            codebook=codebook,
        )
        proxy_error = 0.0
    else:
        quantized, encoded = _torch_ldlq_oracle(
            regularized,
            factor.numpy(),
            bits=bits,
            codebook=codebook,
        )
        error = torch.from_numpy(regularized - quantized)
        source = torch.from_numpy(regularized)
        numerator = torch.sum(error * (transformed_hessian @ error))
        denominator = torch.sum(source * (transformed_hessian @ source))
        proxy_error = float((numerator / denominator.clamp_min(1e-8)).item())

    reconstructed = _torch_hadamard_oracle(quantized, 0) * input_scales
    reconstructed = _torch_hadamard_oracle(reconstructed, 1) * output_scales
    trellis = _torch_pack_trellis_oracle(encoded, bits).numpy()
    return (
        reconstructed,
        proxy_error,
        trellis,
        input_scales.reshape(-1).astype(np.float16),
        output_scales.reshape(-1).astype(np.float16),
        fallback,
        global_scale,
        apply_output_scales,
    )


def _inputs(seed, *, fallback=False):
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.2, (128, 128)).astype(np.float32)
    if fallback:
        hessian = np.zeros((128, 128), dtype=np.float32)
    else:
        calibration = torch.from_numpy(
            rng.normal(0.0, 0.1, (128, 192)).astype(np.float32)
        )
        hessian = (calibration @ calibration.T).numpy()
        hessian += np.eye(128, dtype=np.float32) * np.float32(2.0)
    return (
        weight,
        hessian.astype(np.float32),
        _signs(128, seed + 1)[:, None],
        _signs(128, seed + 2)[None, :],
    )


@pytest.mark.parametrize("codebook", ("3inst", "mcg", "mul1"))
def test_exl3_pipeline_matches_torch_oracle(codebook):
    inputs = _inputs(8317)
    expected = _pipeline_oracle(
        *inputs,
        sample_count=8,
        bits=4,
        codebook=codebook,
    )
    actual = exl3_quantize_weight_mlx(
        *(mx.array(value) for value in inputs),
        sample_count=8,
        bits=4,
        codebook=codebook,
    )
    actual_values = tuple(
        np.asarray(value) if hasattr(value, "shape") else value for value in actual
    )

    np.testing.assert_array_equal(actual_values[2], expected[2])
    np.testing.assert_array_equal(actual_values[3], expected[3])
    np.testing.assert_array_equal(actual_values[4], expected[4])
    np.testing.assert_allclose(actual_values[0], expected[0], atol=1e-6, rtol=1e-6)
    assert _normalized_drift(actual_values[0], expected[0]) <= 1e-6
    assert abs(float(actual_values[1]) - expected[1]) <= 1e-6
    assert actual[5:] == expected[5:]


def test_exl3_pipeline_fallback_matches_torch_oracle():
    inputs = _inputs(9317, fallback=True)
    expected = _pipeline_oracle(
        *inputs,
        sample_count=1,
        bits=4,
        codebook="mcg",
    )
    actual = exl3_quantize_weight_mlx(
        *(mx.array(value) for value in inputs),
        sample_count=1,
        bits=4,
        codebook="mcg",
    )
    np.testing.assert_array_equal(np.asarray(actual[2]), expected[2])
    _assert_composed_weight_matches(np.asarray(actual[0]), expected[0])
    assert float(actual[1].item()) == 0.0
    assert actual[5] is True


@pytest.mark.parametrize("codebook", ("3inst", "mcg", "mul1"))
def test_exl3_pipeline_rounding_boundaries(codebook):
    tie = _half_rounding_boundary(codebook)
    codebook_values = _torch_codebook_oracle(codebook).numpy()
    values = np.array(
        [
            np.nextafter(tie, np.float16(-np.inf), dtype=np.float16),
            tie,
            np.nextafter(tie, np.float16(np.inf), dtype=np.float16),
            -0.0,
            0.0,
            codebook_values.min(),
            codebook_values.max(),
        ],
        dtype=np.float32,
    )
    weight = np.resize(values, (128, 128)).astype(np.float32, copy=False)
    hessian = np.eye(128, dtype=np.float32)
    input_signs = np.ones((128, 1), dtype=np.float32)
    output_signs = np.ones((1, 128), dtype=np.float32)
    expected = _pipeline_oracle(
        weight,
        hessian,
        input_signs,
        output_signs,
        sample_count=1,
        bits=4,
        codebook=codebook,
        force_output_scales=False,
    )
    actual = exl3_quantize_weight_mlx(
        mx.array(weight),
        mx.array(hessian),
        mx.array(input_signs),
        mx.array(output_signs),
        sample_count=1,
        bits=4,
        codebook=codebook,
        force_output_scales=False,
    )
    np.testing.assert_array_equal(np.asarray(actual[2]), expected[2])
    _assert_composed_weight_matches(np.asarray(actual[0]), expected[0])


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_exl3_processor_bridge_preserves_outputs(dtype):
    weight, hessian, _, _ = _inputs(10317)
    weight_t = torch.from_numpy(weight).to(dtype).float().contiguous()
    hessian_t = torch.from_numpy(hessian)
    reconstructed, proxy_error, tensors = exl3_quantize_weight_mlx_to_torch(
        weight_t,
        hessian_t,
        sample_count=8,
        bits=4,
        codebook="mcg",
        force_output_scales=None,
        sigma_reg=0.025,
        seed=787,
    )
    input_signs = _signs(128, 787)[:, None]
    output_signs = _signs(128, 788)[None, :]
    expected = _pipeline_oracle(
        weight_t.numpy(),
        hessian,
        input_signs,
        output_signs,
        sample_count=8,
        bits=4,
        codebook="mcg",
    )
    np.testing.assert_allclose(reconstructed.numpy(), expected[0], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(tensors["trellis"].numpy(), expected[2])
    np.testing.assert_array_equal(tensors["suh"].numpy(), expected[3])
    np.testing.assert_array_equal(tensors["svh"].numpy(), expected[4])
    assert abs(proxy_error - expected[1]) <= 1e-6
    assert "mcg" in tensors and "mul1" not in tensors


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_exl3_processor_bridge_preserves_mps_values():
    weight, hessian, _, _ = _inputs(11317)
    expected = exl3_quantize_weight_mlx_to_torch(
        torch.from_numpy(weight),
        torch.from_numpy(hessian),
        sample_count=8,
        bits=4,
        codebook="mcg",
        force_output_scales=None,
        sigma_reg=0.025,
        seed=787,
    )
    actual = exl3_quantize_weight_mlx_to_torch(
        torch.from_numpy(weight).to("mps"),
        torch.from_numpy(hessian).to("mps"),
        sample_count=8,
        bits=4,
        codebook="mcg",
        force_output_scales=None,
        sigma_reg=0.025,
        seed=787,
    )

    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=1e-6, rtol=1e-6)
    assert abs(actual[1] - expected[1]) <= 1e-6
    for key in ("trellis", "suh", "svh", "mcg"):
        torch.testing.assert_close(actual[2][key].cpu(), expected[2][key], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_pipeline_qwen38_projection_oracle(
    name, out_features, in_features, dtype
):
    source_tensor = torch.full(
        (in_features, out_features), 0.2, dtype=dtype
    ).float()
    source = source_tensor.numpy()
    hessian = np.zeros((in_features, in_features), dtype=np.float32)
    input_signs = np.ones((in_features, 1), dtype=np.float32)
    output_signs = np.ones((1, out_features), dtype=np.float32)
    expected = _pipeline_oracle(
        source,
        hessian,
        input_signs,
        output_signs,
        sample_count=1,
        bits=4,
        codebook="mcg",
        force_output_scales=False,
    )
    actual = exl3_quantize_weight_mlx(
        mx.array(source),
        mx.array(hessian),
        mx.array(input_signs),
        mx.array(output_signs),
        sample_count=1,
        bits=4,
        codebook="mcg",
        force_output_scales=False,
    )
    actual_weight = np.asarray(actual[0])
    np.testing.assert_array_equal(
        np.asarray(actual[2]), expected[2], err_msg=f"{name} {dtype} trellis"
    )
    np.testing.assert_array_equal(
        np.asarray(actual[3]), expected[3], err_msg=f"{name} {dtype} input scales"
    )
    np.testing.assert_array_equal(
        np.asarray(actual[4]), expected[4], err_msg=f"{name} {dtype} output scales"
    )
    np.testing.assert_allclose(
        actual_weight,
        expected[0],
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"{name} {dtype} reconstructed weight",
    )
    assert _normalized_drift(actual_weight, expected[0]) <= 1e-6
    assert float(actual[1].item()) == expected[1] == 0.0

    del source_tensor, source, hessian, input_signs, output_signs
    del expected, actual, actual_weight
    gc.collect()
    mx.clear_cache()


def test_exl3_pipeline_rejects_invalid_inputs():
    valid = mx.zeros((128, 128), dtype=mx.float32)
    signs = mx.ones((128, 1), dtype=mx.float32)
    with pytest.raises(ValueError, match="dimensions"):
        exl3_quantize_weight_mlx(
            mx.zeros((128, 64), dtype=mx.float32),
            valid,
            signs,
            mx.ones((1, 64), dtype=mx.float32),
            sample_count=1,
            bits=4,
        )
    with pytest.raises(ValueError, match="hessian"):
        exl3_quantize_weight_mlx(
            valid,
            mx.eye(64, dtype=mx.float32),
            signs,
            signs.T,
            sample_count=1,
            bits=4,
        )


def test_exl3_processor_routes_cpu_quantization_through_mlx(monkeypatch):
    linear = torch.nn.Linear(128, 128, bias=False)
    named = NamedModule(linear, "proj", "layers.0.proj", 0)

    class Capture:
        nsamples = 8
        H = torch.eye(128, dtype=torch.float32)
        freed = False

        def finalize_hessian(self, target_device=None):
            self.H = self.H.to(target_device)
            return self.H

        def clone_module(self, copy=True, device=None):
            return linear.weight.detach().to(device=device, copy=copy).float()

        def free(self):
            self.freed = True

    capture = Capture()
    config = EXL3Config(bits=4, codebook="mcg")
    processor = object.__new__(EXL3Processor)
    processor.tasks = {named.name: {"capture": capture, "qcfg": config}}
    processor.qcfg = config
    processor.lm_head_name = "lm_head"
    processor.avg_losses = []
    processor.durations = []
    processor.module_names = []
    processor.log = []
    processor._stats_lock = threading.Lock()
    processor.draw_progress = lambda *args, **kwargs: None
    processor.module_feature_summary = lambda module: ""
    processor.module_dtype_size_summary = lambda module: ""
    processor.formatted_fwd_time = lambda: "0.000"
    processor.device_memory_report = lambda: ""
    processor.log_new_row = lambda stat: None

    calls = []

    def fake_quantize(weight, hessian, **kwargs):
        calls.append((weight.clone(), hessian.clone(), kwargs))
        tensors = {
            "trellis": torch.zeros((8, 8, 64), dtype=torch.int16),
            "suh": torch.ones(128, dtype=torch.float16),
            "svh": torch.ones(128, dtype=torch.float16),
            "mcg": torch.tensor(-878_191_635, dtype=torch.int32),
        }
        return weight + 0.25, 0.125, tensors

    monkeypatch.setattr(
        processor_module, "exl3_mlx_quantization_available", lambda: True
    )
    monkeypatch.setattr(
        processor_module, "exl3_quantize_weight_mlx_to_torch", fake_quantize
    )
    processor.process(named, device=torch.device("cpu"))

    assert len(calls) == 1
    assert calls[0][0].shape == (128, 128)
    assert calls[0][2]["bits"] == 4
    assert calls[0][2]["codebook"] == "mcg"
    assert capture.freed
    assert {"trellis", "suh", "svh", "mcg"} <= named.state.keys()
    torch.testing.assert_close(
        named.weight,
        calls[0][0].T.to(named.weight.dtype) + 0.25,
    )


def test_exl3_processor_rejects_cpu_without_mlx(monkeypatch):
    processor = object.__new__(EXL3Processor)
    processor.tasks = {
        "proj": {
            "capture": object(),
            "qcfg": EXL3Config(bits=4),
        }
    }
    processor.draw_progress = lambda *args, **kwargs: None
    module = NamedModule(
        torch.nn.Linear(128, 128, bias=False),
        "proj",
        "layers.0.proj",
        0,
    )
    monkeypatch.setattr(
        processor_module, "exl3_mlx_quantization_available", lambda: False
    )
    with pytest.raises(ValueError, match="requires CUDA/HIP"):
        processor.process(module, device=torch.device("cpu"))
