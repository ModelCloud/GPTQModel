# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gptqmodel.nn_modules.qlinear.exllamav2 as exllamav2_module
import gptqmodel.utils.exllamav2 as exllamav2_utils
from gptqmodel.utils import cpp as cpp_module
from gptqmodel.utils.model import gptqmodel_post_init
from tests.accelerator_gates import requires_nvidia_cuda


def _build_module(monkeypatch) -> exllamav2_module.ExllamaV2Linear:
    # These tests stub the native ops; construction must not compile CUDA on
    # CPU/ROCm hosts before those stubs can be installed.
    monkeypatch.setattr(
        exllamav2_module.ExllamaV2Linear, "cached_validate_once", classmethod(lambda cls: (True, None))
    )
    return exllamav2_module.ExllamaV2Linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
    )


def test_exllamav2_make_q_matrix_uses_jit_op(monkeypatch):
    module = _build_module(monkeypatch)
    calls = {}

    def fake_make_q_matrix(qweight, q_perm, q_invperm, q_scale, q_scale_max, q_groups, qzeros, scales, g_idx):
        calls["make_q_matrix"] = {
            "qweight_shape": tuple(qweight.shape),
            "scales_dtype": scales.dtype,
            "g_idx_is_none": g_idx is None,
        }
        return 123

    monkeypatch.setattr(exllamav2_module, "exllamav2_make_q_matrix", fake_make_q_matrix)

    weights = {
        "qweight": module.qweight,
        "qzeros": module.qzeros,
        "scales": module.scales.to(dtype=torch.float32),
        "g_idx": torch.zeros_like(module.g_idx),
    }

    handle = module.ext_make_q_matrix(weights)

    assert handle == 123
    assert calls["make_q_matrix"] == {
        "qweight_shape": tuple(module.qweight.shape),
        "scales_dtype": torch.float16,
        "g_idx_is_none": True,
    }


def test_exllamav2_forward_uses_jit_gemm(monkeypatch):
    module = _build_module(monkeypatch)
    module.q_handle = 77
    calls = {}

    def fake_gemm(x, q_handle, output, force_cuda):
        calls["gemm"] = {
            "x_shape": tuple(x.shape),
            "x_dtype": x.dtype,
            "q_handle": q_handle,
            "output_shape": tuple(output.shape),
            "force_cuda": force_cuda,
        }
        output.copy_(torch.full_like(output, 5.0))

    monkeypatch.setattr(exllamav2_module, "exllamav2_gemm_half_q_half", fake_gemm)

    x = torch.randn((2, module.in_features), dtype=torch.float32)
    out = module(x)

    assert calls["gemm"] == {
        "x_shape": (2, module.in_features),
        "x_dtype": torch.float32,
        "q_handle": 77,
        "output_shape": (2, module.out_features),
        "force_cuda": False,
    }
    assert out.shape == (2, module.out_features)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full_like(out, 5.0))


def test_exllamav2_act_order_handle_is_not_rebuilt_each_forward(monkeypatch):
    module = _build_module(monkeypatch)
    module.g_idx[0] = 1
    make_calls = []
    free_calls = []

    def fake_make(*args):
        make_calls.append(args)
        return 100 + len(make_calls)

    monkeypatch.setattr(
        exllamav2_module,
        "exllamav2_make_q_matrix",
        fake_make,
    )
    monkeypatch.setattr(
        exllamav2_module,
        "exllamav2_free_q_matrix",
        free_calls.append,
    )

    first = module._ensure_q_handle()
    second = module._ensure_q_handle()

    assert first == second == 101
    assert len(make_calls) == 1
    assert free_calls == []


def test_exllamav2_validate_once_surfaces_jit_error(monkeypatch):
    # Pin the host family so this exercises the non-ROCm JIT-error path even
    # when the suite runs on a HIP device.
    monkeypatch.setattr(exllamav2_module, "IS_ROCM", False)
    monkeypatch.setattr(exllamav2_module, "exllamav2_gptq_runtime_available", lambda: False)
    monkeypatch.setattr(exllamav2_module, "exllamav2_gptq_runtime_error", lambda: "missing exllamav2 jit ops")

    ok, err = exllamav2_module.ExllamaV2Linear.validate_once()

    assert ok is False
    assert isinstance(err, ImportError)
    assert "missing exllamav2 jit ops" in str(err)


def test_exllamav2_include_paths_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_module, "IS_ROCM", False)
    root = tmp_path / "exllamav2"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (wheel_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(exllamav2_utils, "_exllamav2_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = exllamav2_utils._exllamav2_include_paths()

    assert include_paths[0] == str(root)
    assert str(wheel_cuda_include) in include_paths


def test_exllamav2_include_paths_skip_wheel_headers_when_local_cuda_has_required_headers(monkeypatch, tmp_path):
    root = tmp_path / "exllamav2"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (local_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(exllamav2_utils, "_exllamav2_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = exllamav2_utils._exllamav2_include_paths()

    assert include_paths == [str(root)]


# --- #5715: FP16 prefill with a bias must keep FP16 GEMM operands -----------
#
# `MAX_Q_GEMM_ROWS` (50) sends a larger row count to the extension's dense
# fallback, which reconstructs the FP16 weight and hands it to cuBLAS. A biased
# FP16 call asks for an FP32 output (the bias is added before the single
# visible cast), and that used to fall through to
# `at::matmul(a.to(kFloat), weights.to(kFloat))`: an FP32 copy of the weight
# (4*K*N bytes), an FP32 copy of the activations (4*M*K) and an FP32 SGEMM.
# The fixed path asks cuBLAS for an FP32 accumulator (`gemm_half_half_float_cuda`)
# and allocates only the requested FP32 output.

_PREFILL_IN = 4096
_PREFILL_OUT = 4096
_PREFILL_GROUP = 128
_PREFILL_ROWS = 64  # > MAX_Q_GEMM_ROWS (50): the dense fallback
# The FP32-copy path adds 4*K*N = 64 MiB for the weight alone at this shape, so
# 24 MiB separates it from the FP16-operand path with a wide margin.
_MAX_BIAS_EXTRA_MIB = 24.0


def _prefill_codes(rows, cols):
    """Deterministic 4-bit codes; no version-dependent RNG stream."""
    row = torch.arange(rows, dtype=torch.int64).reshape(rows, 1)
    col = torch.arange(cols, dtype=torch.int64).reshape(1, cols)
    return (row * 7 + col * 3) % 16


def _pack_input_axis(codes, bits=4):
    """GPTQ layout: pack consecutive input rows into one int32 word."""
    factor = 32 // bits
    k, n = codes.shape
    words = torch.zeros(k // factor, n, dtype=torch.int64)
    for lane in range(factor):
        words.bitwise_or_(codes[lane::factor] << (bits * lane))
    return words.to(torch.int32)


def _pack_output_axis(codes, bits=4):
    """GPTQ layout: pack consecutive output columns into one int32 word."""
    factor = 32 // bits
    groups, n = codes.shape
    words = torch.zeros(groups, n // factor, dtype=torch.int64)
    for lane in range(factor):
        words.bitwise_or_(codes[:, lane::factor] << (bits * lane))
    return words.to(torch.int32)


def _dequant_oracle(layer, codes, zeros):
    """Independent FP32 dequantization of the checkpoint words, rounded like
    the extension's FP16 reconstruction."""
    scales = layer.scales.detach().to("cpu").to(torch.float32)
    group_of_row = torch.arange(codes.shape[0]) // _PREFILL_GROUP
    weight = (codes.to(torch.float32) - zeros.to(torch.float32)[group_of_row]) * scales[group_of_row]
    return weight.to(torch.float16)


def _seed_prefill_buffers(layer, codes, zeros):
    torch.manual_seed(5715)
    layer.qweight = _pack_input_axis(codes)
    layer.qzeros = _pack_output_axis(zeros)
    layer.scales = (torch.rand(_PREFILL_IN // _PREFILL_GROUP, _PREFILL_OUT) * 0.001 + 0.0005).to(
        torch.float16
    )
    layer.g_idx = torch.zeros(_PREFILL_IN, dtype=torch.int32)


def _build_gptq_prefill_layer(device, *, bias):
    layer = exllamav2_module.ExllamaV2Linear(
        bits=4,
        group_size=_PREFILL_GROUP,
        desc_act=False,
        sym=True,
        in_features=_PREFILL_IN,
        out_features=_PREFILL_OUT,
        bias=bias,
        pack_dtype=torch.int32,
    )
    codes = _prefill_codes(_PREFILL_IN, _PREFILL_OUT)
    zeros = torch.ones(_PREFILL_IN // _PREFILL_GROUP, _PREFILL_OUT, dtype=torch.int64)
    _seed_prefill_buffers(layer, codes, zeros)
    if bias:
        layer.bias = (torch.randn(_PREFILL_OUT, dtype=torch.float32) * 0.5).to(torch.float16)
    layer = layer.eval().to(device)
    layer = gptqmodel_post_init(layer, use_act_order=False)
    return layer, codes, zeros


def _build_awq_prefill_layer(device, *, bias):
    from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
    from gptqmodel.quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns
    from gptqmodel.utils.exllamav2 import ScratchSpace

    layer = AwqExllamaV2Linear(
        bits=4,
        group_size=_PREFILL_GROUP,
        desc_act=False,
        sym=True,
        in_features=_PREFILL_IN,
        out_features=_PREFILL_OUT,
        bias=bias,
        register_buffers=True,
    ).to(device)
    codes = _prefill_codes(_PREFILL_IN, _PREFILL_OUT)
    zeros = torch.ones(_PREFILL_IN // _PREFILL_GROUP, _PREFILL_OUT, dtype=torch.int32)
    torch.manual_seed(5715)
    layer.qweight.copy_(pack_awq_columns(codes.to(torch.int32), 4, order=AWQ_ORDER).to(device))
    layer.qzeros.copy_(pack_awq_columns(zeros, 4, order=AWQ_ORDER).to(device))
    layer.scales.copy_(
        (torch.rand(_PREFILL_IN // _PREFILL_GROUP, _PREFILL_OUT) * 0.001 + 0.0005)
        .to(torch.float16)
        .to(device)
    )
    if bias:
        layer.bias.copy_((torch.randn(_PREFILL_OUT, dtype=torch.float32) * 0.5).to(torch.float16).to(device))
    layer = layer.eval()
    layer.post_init(ScratchSpace(layer.scratch_space_fixed(), device))
    return layer, codes, zeros


def _forward_peak_mib(layer, x):
    """Peak live device allocation (MiB) of one forward call."""
    with torch.inference_mode():
        layer(x)  # warm: handle lookup, allocator pools
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        out = layer(x)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / (1024 * 1024)
    del out
    return peak


def _assert_wide_prefill(monkeypatch, gemm_name, builder, bias, device):
    layer, codes, zeros = builder(device, bias=bias)
    control, _, _ = builder(device, bias=False)

    torch.manual_seed(5715 + _PREFILL_ROWS)
    x = (torch.randn(_PREFILL_ROWS, _PREFILL_IN, dtype=torch.float16) * 0.5).to(device)

    # Trigger evidence: the layer must ask the extension for an FP32 output.
    requested = []
    if hasattr(exllamav2_module, gemm_name):
        real_gemm = getattr(exllamav2_module, gemm_name)
        monkeypatch.setattr(exllamav2_module, gemm_name, _recorder(real_gemm, requested))
    else:
        import gptqmodel.nn_modules.qlinear.exllamav2_awq as awq_module

        real_gemm = getattr(awq_module, gemm_name)
        monkeypatch.setattr(awq_module, gemm_name, _recorder(real_gemm, requested))

    with torch.inference_mode():
        out = layer(x)
    assert requested and requested[0] is torch.float32

    assert out.dtype is torch.float16
    assert out.shape == (_PREFILL_ROWS, _PREFILL_OUT)

    oracle = x.to(torch.float64) @ _dequant_oracle(layer, codes, zeros).to(device).to(torch.float64)
    oracle = oracle + layer.bias.to(torch.float64)
    torch.testing.assert_close(out, oracle.to(torch.float16), rtol=2e-3, atol=2e-3)

    # Allocation regression: the biased prefill may not pay for an FP32 copy of
    # the reconstructed weight. Measure the delta against an identical unbiased
    # layer so the transient FP16 reconstruction cancels.
    biased_peak = _forward_peak_mib(layer, x)
    unbiased_peak = _forward_peak_mib(control, x)
    extra = biased_peak - unbiased_peak
    assert extra < _MAX_BIAS_EXTRA_MIB, (
        f"biased FP16 prefill allocated {extra:.1f} MiB more than the unbiased "
        f"forward ({biased_peak:.1f} vs {unbiased_peak:.1f} MiB); the FP32 "
        f"weight copy is back"
    )


def _recorder(real_gemm, requested):
    def wrapped(a, handle, c, force_cuda=False):
        requested.append(c.dtype)
        return real_gemm(a, handle, c, force_cuda)

    return wrapped


@pytest.mark.skipif(not requires_nvidia_cuda(), reason="ExLlamaV2 kernels are CUDA-only")
def test_exllamav2_biased_fp16_prefill_keeps_fp16_gemm_operands(monkeypatch):
    _assert_wide_prefill(
        monkeypatch,
        "exllamav2_gemm_half_q_half",
        _build_gptq_prefill_layer,
        True,
        torch.device("cuda"),
    )


@pytest.mark.skipif(not requires_nvidia_cuda(), reason="ExLlamaV2 kernels are CUDA-only")
def test_exllamav2_awq_biased_fp16_prefill_keeps_fp16_gemm_operands(monkeypatch):
    _assert_wide_prefill(
        monkeypatch,
        "exllamav2_awq_gemm_half_q_half",
        _build_awq_prefill_layer,
        True,
        torch.device("cuda"),
    )
