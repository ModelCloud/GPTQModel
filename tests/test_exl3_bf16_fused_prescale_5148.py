# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for #5148: the BF16 EXL3 pre-scale is one launch.

#4989 removed the host sync from ``ExllamaV3Linear._native_bf16_input``, but the
BF16 decode path was still ~2x slower than FP16 because that method and
``_rescale_native_output`` ran about a dozen eager elementwise kernels per
linear per forward (``vector_norm``, ``.float()``, ``div``, ``frexp``,
``ones_like`` x2, ``ldexp``, ``where``, ``x / scale``, two casts, then
``.to(fp32)``, ``* scale``, ``.to(bf16)``). At M=1 each is a microsecond kernel
behind a full host dispatch, so the GEMM starved: 96 us/linear BF16 against
33 us FP16 on sm_120.

``gptqmodel_ext/exllamav3/bf16_scale.cu`` now performs each half in one launch
and the eager sequence stays as the reference and fallback. This module pins:

* bitwise equality of the fused pre-scale, re-scale and complete forward
  against that fallback, including adversarial rows and every bias dtype;
* the fallback itself (missing fused op, non-contiguous or non-CUDA input);
* that no dense/expanded weight is retained across forwards (AGENTS.md);
* an M=1 ``bf16 <= 1.3x fp16`` per-linear bound that the pre-fix eager chain
  fails by ~2.7x in the same process.
"""

import gc

import pytest
import torch

from gptqmodel.nn_modules import exllamav3 as exl3_bridge
from gptqmodel.nn_modules import exllamav3_bf16_kernels as fused_kernels
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from tests.accelerator_gates import requires_nvidia_cuda


requires_cuda = pytest.mark.skipif(
    not requires_nvidia_cuda(),
    reason="the native EXL3 dispatch needs an NVIDIA CUDA device",
)

# The issue measured 2x on gate_proj at M=1; the fused path must come close to
# the FP16 kernel. 1.3x is the bound from the issue report.
_MAX_BF16_FP16_RATIO = 1.3


def _bridge(out_features: int = 64) -> ExllamaV3Linear:
    """An ``ExllamaV3Linear`` without ``nn.Module.__init__`` for op tests."""
    module = object.__new__(ExllamaV3Linear)
    module.out_features = out_features
    module.bias = None
    return module


@pytest.fixture(scope="module")
def fused_ops():
    """The compiled fused ops, or a precise skip when this build lacks them."""
    from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

    if not prewarm_exllamav3_extension():
        pytest.skip("the EXL3 torch.ops extension is not available in this build")
    if not fused_kernels.fused_available():
        pytest.skip(
            "this EXL3 extension build predates bf16_prescale/bf16_rescale (#5148)"
        )
    return fused_kernels


@requires_cuda
def test_fused_ops_are_registered_in_the_extension():
    """The shipped extension must expose both kernels (red on 17a488e41)."""
    from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

    if not prewarm_exllamav3_extension():
        pytest.skip("the EXL3 torch.ops extension is not available in this build")
    assert fused_kernels.fused_available(), (
        "gptqmodel_exllamav3 must register bf16_prescale and bf16_rescale; "
        "a build without them silently keeps the ~12-launch eager BF16 chain"
    )


@pytest.fixture(scope="module")
def quantized_layer(fused_ops):
    """A real EXL3 W4 layer quantized on device by the production quantizer."""
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    device = torch.device("cuda:0")
    in_features, out_features = 512, 512
    torch.manual_seed(5148)
    weight = torch.randn((in_features, out_features), dtype=torch.float32, device=device) * 0.02
    activations = torch.randn((64, in_features), dtype=torch.float32, device=device)
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    quant_args = {
        "K": 4,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": 5148,
        "mcg": True,
    }
    _, _, tensors = quantize_exl3(
        weight,
        {"H": hessian, "count": activations.shape[0], "finalized": False},
        quant_args,
        return_weight_q=False,
    )
    tensors = dict(tensors)
    tensors["bias"] = (
        torch.randn(out_features, device=device, dtype=torch.float16) * 0.01
    )
    module = ExllamaV3Linear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name="layer.mlp.gate_proj",
        tensors=tensors,
    ).to(device)
    module.post_init()
    return module


def _bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    # Compare raw bits: +0.0/-0.0 and NaN payloads must match too.
    return torch.equal(
        actual.view(torch.int16 if actual.element_size() == 2 else torch.int32),
        expected.view(torch.int16 if expected.element_size() == 2 else torch.int32),
    )


def _disable_fused_ops(monkeypatch):
    """Force the eager reference path in ``ExllamaV3Linear``.

    ``raising=False`` keeps this usable against a pre-#5148 checkout, where the
    names do not exist yet, so the timing bound can be run red.
    """
    monkeypatch.setattr(exl3_bridge, "_try_fused_prescale", lambda x: None, raising=False)
    monkeypatch.setattr(
        exl3_bridge, "_try_fused_rescale", lambda *a, **kw: None, raising=False
    )


# --------------------------------------------------------------------------
# Adversarial activation fixtures.  Random rows alone never reach the
# subnormal, overflow and all-zero branches the power-of-two decision has.
# --------------------------------------------------------------------------


def _activation_cases(in_features: int):
    """(name, tensor) pairs covering the pre-scale decision boundaries."""
    device = torch.device("cuda:0")
    torch.manual_seed(5148)
    bound = exl3_bridge._FP16_SAFE_AMAX
    rows = [
        ("zero-row", torch.zeros(1, in_features)),
        ("denormal", torch.full((1, in_features), 1e-40)),
        ("fp16-max", torch.full((1, in_features), 65504.0)),
        ("just-below-bound", torch.full((1, in_features), bound * 0.99)),
        ("just-above-bound", torch.full((1, in_features), bound * 1.01)),
        ("bf16-max", torch.full((1, in_features), 3.0e38)),
        ("mixed-outlier", torch.cat([
            torch.full((1, in_features), 1e5),
            torch.randn(3, in_features) * 0.3,
        ])),
        ("negative", -torch.randn(2, in_features) * 1e4),
        ("random-wide", torch.randn(2, in_features) * 1e6),
        ("inf-row", torch.tensor([[float("inf")] * in_features])),
        ("nan-row", torch.tensor([[float("nan")] * in_features])),
    ]
    return [(name, row.to(device=device, dtype=torch.bfloat16)) for name, row in rows]


# --------------------------------------------------------------------------
# Bitwise equality with the eager reference.
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("rows", [1, 8, 512, 2048])
def test_prescale_matches_the_eager_reference_bitwise(fused_ops, rows):
    """The fused amax/scale/cast equals ``_native_bf16_input``'s eager math."""
    device = torch.device("cuda:0")
    module = _bridge()
    module.__dict__["_native_bf16_inputs_finite"] = True
    torch.manual_seed(5148 + rows)
    for magnitude in (0.5, 1e5, 3.0e38):
        x = (torch.randn(rows, 4096, device=device) * magnitude).to(torch.bfloat16)
        actual_input, actual_scale = fused_kernels.try_prescale(x)
        expected_input, expected_scale = module._prescale_from_row_amax(
            x, module._row_amax(x)
        )
        assert _bits_equal(actual_input, expected_input), (rows, magnitude)
        assert _bits_equal(actual_scale, expected_scale), (rows, magnitude)


@requires_cuda
def test_prescale_matches_the_eager_reference_on_adversarial_rows(fused_ops):
    device = torch.device("cuda:0")
    module = _bridge()
    for name, x in _activation_cases(4096):
        x = x.to(device)
        actual = fused_kernels.try_prescale(x)
        expected = module._prescale_from_row_amax(x, module._row_amax(x))
        assert actual is not None, name
        assert _bits_equal(actual[0], expected[0]), name
        assert _bits_equal(actual[1], expected[1]), name


@requires_cuda
@pytest.mark.parametrize("result_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32, None])
@pytest.mark.parametrize("stored_padding", [0, 128])
def test_rescale_matches_the_eager_reference_bitwise(
    fused_ops, result_dtype, bias_dtype, stored_padding
):
    """Bias subtract/scale/add stay three separate FP32 roundings, then one cast.

    The BF16 native path keeps the GEMM result in FP32 (#5331), so the fused
    re-scale must accept both result widths.
    """
    device = torch.device("cuda:0")
    rows, out_features = 8, 1024
    module = _bridge(out_features)
    stored = out_features + stored_padding
    torch.manual_seed(5148)
    for scale_value in (1.0, 2.0**17, 2.0**60):
        result = (torch.randn(rows, stored, device=device) * 0.5).to(result_dtype)
        scale = torch.full((rows, 1), scale_value, device=device, dtype=torch.float32)
        bias = (
            None
            if bias_dtype is None
            else (torch.randn(stored, device=device) * 0.1).to(bias_dtype)
        )
        module.bias = bias
        expected = module._rescale_native_output(
            result[..., :out_features], scale
        ).to(torch.bfloat16)
        actual = fused_kernels.try_rescale(result, scale, bias, out_features)
        assert actual is not None
        assert _bits_equal(actual, expected), (bias_dtype, stored_padding, scale_value)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 8, 512, 2048])
def test_forward_matches_the_eager_fallback_bitwise(
    fused_ops, quantized_layer, monkeypatch, rows
):
    """The shipped fallback branch is the pre-fix code, and it must agree."""
    device = torch.device("cuda:0")
    for magnitude in (0.5, 1e5):
        x = (torch.randn(rows, 512, device=device) * magnitude).to(torch.bfloat16)
        with torch.inference_mode():
            actual = quantized_layer(x)
            _disable_fused_ops(monkeypatch)
            expected = quantized_layer(x)
            monkeypatch.undo()
        assert _bits_equal(actual, expected), (rows, magnitude)
        assert actual.dtype == torch.bfloat16


@requires_cuda
def test_forward_matches_the_eager_fallback_on_adversarial_rows(
    fused_ops, quantized_layer, monkeypatch
):
    device = torch.device("cuda:0")
    for name, x in _activation_cases(512):
        x = x.to(device)
        with torch.inference_mode():
            actual = quantized_layer(x)
            _disable_fused_ops(monkeypatch)
            expected = quantized_layer(x)
            monkeypatch.undo()
        assert _bits_equal(actual, expected), name


# --------------------------------------------------------------------------
# Fallbacks: the fused path must decline rather than change the result.
# --------------------------------------------------------------------------


@requires_cuda
def test_missing_fused_ops_fall_back_to_the_eager_path(fused_ops, quantized_layer, monkeypatch):
    """A build without the new ops (e.g. a stale prebuilt wheel) still works."""
    monkeypatch.setattr(fused_kernels, "_op", lambda name: None)
    probe = torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16)
    assert fused_kernels.try_prescale(probe) is None
    assert fused_kernels.fused_available() is False
    x = (torch.randn(4, 512, device="cuda") * 0.5).to(torch.bfloat16)
    with torch.inference_mode():
        fallback = quantized_layer(x)
    monkeypatch.undo()
    with torch.inference_mode():
        fused = quantized_layer(x)
    assert _bits_equal(fallback, fused)


@requires_cuda
def test_unsupported_inputs_decline_the_fused_path(fused_ops):
    """CPU, FP16/FP32, empty and non-contiguous activations keep the eager path."""
    cuda = torch.device("cuda")
    assert fused_kernels.try_prescale(
        torch.zeros(2, 512, dtype=torch.bfloat16)
    ) is None
    assert fused_kernels.try_prescale(
        torch.zeros(2, 512, device=cuda, dtype=torch.float16)
    ) is None
    assert fused_kernels.try_prescale(
        torch.zeros(2, 512, device=cuda, dtype=torch.float32)
    ) is None
    assert fused_kernels.try_prescale(
        torch.zeros(0, 512, device=cuda, dtype=torch.bfloat16)
    ) is None
    strided = torch.zeros(4, 1024, device=cuda, dtype=torch.bfloat16)[:, ::2]
    assert not strided.is_contiguous()
    assert fused_kernels.try_prescale(strided) is None
    # A contiguous FP16 result with a mismatched scale row count is rejected
    # rather than read out of bounds.
    result = torch.zeros(4, 512, device=cuda, dtype=torch.float16)
    assert fused_kernels.try_rescale(result, torch.ones(3, 1, device=cuda), None, 512) is None


# --------------------------------------------------------------------------
# Storage rule: no dense/expanded weight may outlive a forward.
# --------------------------------------------------------------------------


@requires_cuda
def test_forward_retains_no_expanded_weight(fused_ops, quantized_layer):
    """Only packed checkpoint buffers may stay resident (AGENTS.md)."""
    dense_bytes = quantized_layer.in_features * quantized_layer.out_features * 2
    x = (torch.randn(4, 512, device="cuda") * 0.5).to(torch.bfloat16)
    with torch.inference_mode():
        for _ in range(3):
            quantized_layer(x)
    torch.cuda.synchronize()
    gc.collect()

    retained = {}
    for name, value in vars(quantized_layer).items():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            retained[name] = value.numel() * value.element_size()
    for name, value in quantized_layer._buffers.items():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            retained[name] = value.numel() * value.element_size()
    for name, value in quantized_layer._parameters.items():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            retained[name] = value.numel() * value.element_size()
    assert not [name for name, size in retained.items() if size >= dense_bytes], retained
    # The packed trellis is the only multi-hundred-KB resident tensor.
    assert quantized_layer.trellis.dtype == torch.int16
    assert retained["trellis"] < dense_bytes


# --------------------------------------------------------------------------
# The bound from the issue: bf16 decode within 1.3x of fp16 per linear.
# --------------------------------------------------------------------------


def _per_call_us(module, x, iters: int = 400) -> float:
    """Minimum of several event-timed loops: the GPU-resident call cost."""
    with torch.inference_mode():
        for _ in range(50):
            module(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(7):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                module(x)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iters * 1e3)
    return min(samples)


@requires_cuda
@pytest.mark.slow
def test_bf16_decode_is_within_1_3x_of_fp16_per_linear(monkeypatch):
    """M=1 ``gate_proj`` must not pay the eager pre-scale launch chain (#5148).

    The issue's repro: ``gate_proj`` 4096 -> 14336 at M=1 was 96.2 us in BF16
    against 33.0 us in FP16. The test deliberately does not require the fused
    ops to exist: on 17a488e41 the module itself runs the eager chain and
    breaches the bound (2.7x), while the fused path meets it (1.1x). The same
    eager chain is also measured in-process with the fused ops disabled, so the
    detection does not rest on a historical number.
    """
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    cuda = torch.device("cuda:0")
    in_features, out_features = 4096, 14336
    torch.manual_seed(5148)
    weight = torch.randn((in_features, out_features), dtype=torch.float32, device=cuda) * 0.02
    activations = torch.randn((64, in_features), dtype=torch.float32, device=cuda)
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    _, _, tensors = quantize_exl3(
        weight,
        {"H": hessian, "count": activations.shape[0], "finalized": False},
        {
            "K": 4,
            "devices": [cuda],
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 5148,
            "mcg": True,
        },
        return_weight_q=False,
    )
    del weight, activations, hessian
    module = ExllamaV3Linear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name="layer.mlp.gate_proj",
        tensors=dict(tensors),
    ).to(cuda)
    module.post_init()

    x_bf16 = (torch.randn(1, in_features, device=cuda) * 0.5).to(torch.bfloat16)
    x_fp16 = x_bf16.to(torch.float16)
    fp16_us = _per_call_us(module, x_fp16)
    fused_us = _per_call_us(module, x_bf16)
    _disable_fused_ops(monkeypatch)
    eager_us = _per_call_us(module, x_bf16)
    monkeypatch.undo()

    message = (
        f"fp16 {fp16_us:.2f} us, bf16 fused {fused_us:.2f} us "
        f"({fused_us / fp16_us:.2f}x), bf16 eager {eager_us:.2f} us "
        f"({eager_us / fp16_us:.2f}x)"
    )
    if fp16_us < 20.0:
        pytest.skip(
            "fp16 per-linear cost is below 20 us, so a fixed ~10 us of fused "
            f"launches cannot meet a ratio bound on this host: {message}"
        )
    # Detection evidence: the pre-fix chain must breach the bound here.
    assert eager_us / fp16_us > _MAX_BF16_FP16_RATIO, message
    assert fused_us / fp16_us <= _MAX_BF16_FP16_RATIO, message
    assert fused_us < eager_us, message
