# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bound and pin the GGUF_CPP native-matmul activation-quantization loss (#4219).

`ggml_mul_mat` has no full-precision activation path. The CPU backend converts
the activations to Q8_0/Q8_1 (legacy layouts) or Q8_K (K-quant layouts) before
the integer dot product, and the CUDA backend additionally accumulates in low
precision. The native result therefore differs from an FP32 dequantized-weight
oracle by roughly 5e-3..1.5e-2 even for FP32 activations.

These tests establish *where* that error comes from instead of assuming it: the
native result must reproduce an independent Q8 round trip of the activations
(ggml's own `quantize_row_q8_*_ref` followed by `dequantize_row_*`) to within
1e-6 on the CPU. The error bound asserted below is derived from that simulation,
not from a hand-tuned tolerance, so loosening a constant cannot make the suite
pass. `GGUF_TORCH` remains the exact path.
"""

import ctypes

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import _GGUF_TYPE_INFO, GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import (
    GGML_NATIVE_MATMUL_ERROR_BOUND,
    GGUFCppKernel,
    GGUFCudaKernel,
    _get_ggml_bridge,
    _ggml_native_output_needs_exact_epilogue,
)


IN_FEATURES = 256
OUT_FEATURES = 64

# Layout -> the activation type ggml's CPU vec_dot converts F32 activations to.
# Q4_1/Q5_1 use Q8_1, which shares Q8_0's layout and rounding; its dot product
# also consumes a per-block activation sum, so the round trip matches to 2e-4
# instead of 1e-6 (measured max 1.48e-4).
_ACTIVATION_QTYPE = {
    "Q1_0": "Q8_0",
    "Q1_0_g128": "Q8_0",
    "Q2_0": "Q8_0",
    "Q4_0": "Q8_0",
    "Q4_1": "Q8_0",
    "Q5_0": "Q8_0",
    "Q5_1": "Q8_0",
    "Q8_0": "Q8_0",
    "IQ4_NL": "Q8_0",
    "MXFP4": "Q8_0",
    "NVFP4": "Q8_0",
    "Q2_K": "Q8_K",
    "Q3_K": "Q8_K",
    "Q4_K": "Q8_K",
    "Q5_K": "Q8_K",
    "Q6_K": "Q8_K",
    "TQ1_0": "Q8_K",
    "TQ2_0": "Q8_K",
    "IQ1_S": "Q8_K",
    "IQ1_M": "Q8_K",
    "IQ2_XXS": "Q8_K",
    "IQ2_XS": "Q8_K",
    "IQ2_S": "Q8_K",
    "IQ3_XXS": "Q8_K",
    "IQ3_S": "Q8_K",
    "IQ4_XS": "Q8_K",
}
# Layouts whose vec_dot also consumes a quantized activation sum.
_SUM_QTYPES = frozenset({"Q4_1", "Q5_1"})
_CPU_BOUND = 1e-2
_CUDA_BOUND = 2e-2


def _bridge():
    try:
        bridge = _get_ggml_bridge()
        ok, error = bridge.cpu_available()
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"llama-cpp-python GGML bridge unavailable: {exc}")
    if not ok:
        pytest.skip(f"GGML CPU bridge unavailable: {error}")
    return bridge


def _kernels(qtype, *, bias=False, cuda=False):
    torch.manual_seed(7)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias)
    reference = GGUFTorchLinear(
        bits=qtype,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
    )
    reference.pack_original(linear, scales=None, zeros=None)
    assert reference.gguf_tensor_qtype == qtype, (reference.gguf_tensor_qtype, qtype)

    def build(cls):
        kernel = cls(
            bits=qtype,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            bias=bias,
        )
        kernel.load_state_dict(reference.state_dict(), strict=True)
        return kernel.eval()

    cpp = build(GGUFCppKernel)
    gpu = None
    if cuda:
        ok, error = GGUFCudaKernel.cached_validate_once()
        if not ok:
            pytest.skip(f"llama-cpp-python GGML CUDA unavailable: {error}")
        gpu = build(GGUFCudaKernel).to("cuda:0")
    return linear, reference, cpp, gpu


def _ggml_dequantized_weight(bridge, reference):
    """Dequantize with ggml's native kernels; cross-checked against the torch one."""

    raw = reference.qweight.detach().cpu().numpy()
    # ggml's dequantizer emits (out_features, in_features); the oracle matmul
    # below is written against the (in_features, out_features) convention that
    # `GGUFTorchLinear.dequantize_weight` returns.
    weight = bridge.dequantize_rows(raw, reference.gguf_tensor_qtype, (OUT_FEATURES, IN_FEATURES))
    native = torch.from_numpy(weight).double().T.contiguous()
    torch_weight = reference.dequantize_weight(device="cpu", dtype=torch.float32).double()
    # Both dequantizers must agree, otherwise the oracle below is meaningless.
    assert torch.allclose(native, torch_weight, rtol=0, atol=1e-6), "dequantizers disagree"
    return native


def _oracle(weight, x, bias=None):
    out = x.double() @ weight
    if bias is not None:
        out = out + bias.double()
    return out


def _q8_simulation(bridge, weight, x, activation_qtype):
    """Independent Q8 round trip of the activations, then an FP64 matmul."""

    info = _GGUF_TYPE_INFO[activation_qtype]
    source = np.ascontiguousarray(x.detach().to(torch.float32).cpu().numpy())
    rows, columns = source.shape
    assert columns % info["block_size"] == 0
    packed = np.empty((rows, columns // info["block_size"] * info["type_size"]), dtype=np.uint8)
    symbol = {
        "Q8_0": "quantize_row_q8_0_ref",
        "Q8_K": "quantize_row_q8_K_ref",
    }[activation_qtype]
    quantize = getattr(bridge._ggml_base, symbol, None)
    assert quantize is not None, f"ggml build lacks {symbol}"
    quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    quantize.restype = None
    for row in range(rows):
        quantize(
            source[row].ctypes.data_as(ctypes.c_void_p),
            packed[row].ctypes.data_as(ctypes.c_void_p),
            columns,
        )
    dequantized = bridge.dequantize_rows(packed, activation_qtype, (rows, columns))
    return torch.from_numpy(dequantized).double() @ weight


def _relative(actual, expected):
    difference = actual.detach().double() - expected.detach().double()
    return float(difference.norm() / expected.detach().double().norm())


@pytest.mark.parametrize("qtype", sorted(_ACTIVATION_QTYPE))
def test_cpu_native_result_equals_independent_q8_activation_simulation(qtype):
    """The whole CPU error is the Q8 activation quantization, not a layout bug."""

    bridge = _bridge()
    _, reference, cpp, _ = _kernels(qtype)
    weight = _ggml_dequantized_weight(bridge, reference)
    torch.manual_seed(11)
    x = torch.randn(4, IN_FEATURES)

    with torch.inference_mode():
        native = cpp(x.clone())

    simulated = _q8_simulation(bridge, weight, x, _ACTIVATION_QTYPE[qtype])
    residual = _relative(native, simulated)
    assert residual <= (2e-4 if qtype in _SUM_QTYPES else 1e-6), (
        f"{qtype}: native result differs from the Q8 activation simulation by {residual:.3e}; "
        "the error is then not (only) activation quantization"
    )
    # The other activation type must be clearly worse, which pins the mapping.
    other = "Q8_K" if _ACTIVATION_QTYPE[qtype] == "Q8_0" else "Q8_0"
    assert _relative(native, _q8_simulation(bridge, weight, x, other)) > 1e-3


@pytest.mark.parametrize("qtype", sorted(_ACTIVATION_QTYPE))
def test_cpu_native_error_stays_within_the_documented_bound(qtype):
    bridge = _bridge()
    _, reference, cpp, _ = _kernels(qtype)
    weight = _ggml_dequantized_weight(bridge, reference)
    torch.manual_seed(11)
    x = torch.randn(4, IN_FEATURES)

    with torch.inference_mode():
        native = cpp(x.clone())

    error = _relative(native, _oracle(weight, x))
    assert error <= _CPU_BOUND, f"{qtype}: CPU native error {error:.3e} exceeds {_CPU_BOUND:.0e}"
    # The documented constant must still cover every supported layout.
    assert error <= GGML_NATIVE_MATMUL_ERROR_BOUND


@pytest.mark.parametrize("qtype", ["Q2_K", "Q4_0", "Q4_1", "Q4_K", "Q6_K", "Q8_0"])
@pytest.mark.parametrize("rows", [1, 8])
def test_cuda_native_error_stays_within_the_documented_bound(qtype, rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    bridge = _bridge()
    _, reference, _, gpu = _kernels(qtype, cuda=True)
    weight = _ggml_dequantized_weight(bridge, reference)
    torch.manual_seed(11)
    x = torch.randn(rows, IN_FEATURES)

    with torch.inference_mode():
        native = gpu(x.clone().to("cuda:0")).float().cpu()

    error = _relative(native, _oracle(weight, x))
    assert error <= _CUDA_BOUND, f"{qtype} M={rows}: CUDA native error {error:.3e} exceeds {_CUDA_BOUND:.0e}"


@pytest.mark.parametrize("qtype", ["Q4_0", "Q4_K", "Q6_K"])
def test_biased_cuda_layer_is_recomputed_exactly(qtype):
    """A biased CUDA layer never returns the lossy native product (#3789)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    bridge = _bridge()
    linear, reference, _, gpu = _kernels(qtype, bias=True, cuda=True)
    weight = _ggml_dequantized_weight(bridge, reference)
    torch.manual_seed(11)
    x = torch.randn(4, IN_FEATURES)

    with torch.inference_mode():
        actual = gpu(x.clone().to("cuda:0")).float().cpu()

    expected = _oracle(weight, x, linear.bias)
    assert _relative(actual, expected) <= 1e-6


def test_exact_epilogue_rule_is_a_static_function_of_bias_presence():
    """No timing, no activation values, no call order decide the exact epilogue."""

    for device_type in ("cpu", "cuda"):
        for native_dtype in (torch.float32, torch.float16, torch.bfloat16):
            assert _ggml_native_output_needs_exact_epilogue(
                device_type=device_type, bias_present=False, native_dtype=native_dtype
            ) is False
    assert _ggml_native_output_needs_exact_epilogue(
        device_type="cuda", bias_present=True, native_dtype=torch.float32
    ) is True
    assert _ggml_native_output_needs_exact_epilogue(
        device_type="cpu", bias_present=True, native_dtype=torch.float32
    ) is False
    for native_dtype in (torch.float16, torch.bfloat16):
        assert _ggml_native_output_needs_exact_epilogue(
            device_type="cpu", bias_present=True, native_dtype=native_dtype
        ) is True


def test_native_path_is_not_chosen_by_timing(monkeypatch):
    """The CPU path choice must not time the two implementations (#4219)."""

    _bridge()
    _, _, cpp, _ = _kernels("Q4_K")

    def timing_forbidden(*args, **kwargs):
        raise AssertionError("the ggml/dense choice must not benchmark the paths")

    from gptqmodel.utils.kernel import KernelConfig, configure_kernel

    monkeypatch.setattr(cpp, "_benchmark_forward_runner", timing_forbidden)
    monkeypatch.setattr(cpp, "_autotune", timing_forbidden)
    configure_kernel(KernelConfig(gguf_cpp_native_matmul=None))
    x = torch.zeros(2, IN_FEATURES)
    assert cpp._use_ggml_cpu_matmul(x) is True
    assert cpp._use_ggml_cpu_matmul(x) is True


def test_native_output_is_bitwise_repeatable():
    """The same input must produce the same bytes on every call."""

    _bridge()
    _, _, cpp, _ = _kernels("Q4_K")
    torch.manual_seed(11)
    x = torch.randn(4, IN_FEATURES)
    with torch.inference_mode():
        first = cpp(x.clone())
        second = cpp(x.clone())
    assert torch.equal(first, second)


def test_zero_bias_does_not_change_the_cpu_matmul_path(monkeypatch):
    """Bias *presence* must not silently switch the CPU product to the dense path."""

    _bridge()
    _, _, cpp, _ = _kernels("Q4_K", bias=True)
    calls = {"native": 0}
    original = cpp._forward_ggml_cpu

    def counting(x_flat):
        calls["native"] += 1
        return original(x_flat)

    monkeypatch.setattr(cpp, "_forward_ggml_cpu", counting)
    torch.manual_seed(11)
    x = torch.randn(2, IN_FEATURES)
    with torch.inference_mode():
        with torch.no_grad():
            cpp.bias.zero_()
            cpp(x.clone())
            cpp.bias.fill_(1.0)
            cpp(x.clone())
    assert calls["native"] == 2
