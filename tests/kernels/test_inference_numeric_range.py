# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Finite-output range regressions, with independent FP64 arithmetic oracles."""

import numpy as np
import pytest
import torch
from test_torch_weight_storage import state_tensors

from gptqmodel.quantization import FORMAT


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
DTYPES = [torch.float16, torch.bfloat16]
NVIDIA = torch.cuda.is_available() and not torch.version.hip


def check(module, x, expected):
    """Check repeat forwards, dtype, rounded oracle, and reachable weight storage."""
    expected = torch.as_tensor(expected, device=x.device, dtype=torch.float64)
    before = {t.data_ptr() for t in state_tensors(module)}
    for _ in range(2):
        with torch.inference_mode():
            actual = module(x)
        assert actual.dtype == x.dtype
        assert torch.isfinite(actual).all()
        oracle = expected.expand_as(actual)
        torch.testing.assert_close(actual, oracle.to(x.dtype), rtol=2e-3, atol=2e-3)
        # Scaled relative error separates output rounding from arithmetic.
        scale = oracle.abs().amax().clamp_min(1)
        print("raw_oracle_normalized_error", float((actual.double() - oracle).abs().amax() / scale))
        for tensor in state_tensors(module):
            if tensor.data_ptr() not in before:
                assert not (tensor.is_floating_point() and tensor.numel() >= x.shape[-1] * actual.shape[-1]), (
                    'Forward retained a newly expanded weight tensor', tensor.shape, tensor.dtype
                )
    return actual


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('fmt', ['float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz'])
@pytest.mark.parametrize('method', ['tensor', 'row', 'block'])
def test_fp8_small_weight_scale(device, dtype, fmt, method):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    m = TorchFP8Linear(bits=8, group_size=-1, desc_act=False, sym=True,
                       in_features=128, out_features=64, bias=False, format=fmt,
                       weight_scale_method=method, weight_block_size=(16, 16) if method == "block" else None)
    linear = torch.nn.Linear(128, 64, bias=False)
    linear.weight.data.fill_(2**-20)
    m.pack(linear, None, None)
    m = m.to(device).eval()
    x = torch.full((1, 128), 1024., dtype=dtype, device=device)
    check(m, x, .125)
    # The explicit dense fallback must also preserve range when scaled_mm exists.
    weight = m.dequantize_weight(device=device, dtype=dtype)
    torch.testing.assert_close(weight, torch.full_like(weight, 2**-20), rtol=2e-3, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('value', [0., 2**-20, -(2**-20), 2**-16])
def test_qqq_small_activation_scale(device, dtype, value):
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
    m = QQQTorchLinear(bits=4, group_size=-1, sym=True, desc_act=False,
                       in_features=128, out_features=128, bias=False, register_buffers=False)
    linear = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16)
    linear.weight.data.fill_(1024.)
    m.pack(linear, torch.full((128, 1), 1024., dtype=torch.float16))
    m = m.to(device).eval()
    x = torch.full((1, 128), value, dtype=dtype, device=device)
    check(m, x, 128 * value * 1024)
    q, s = m.dynamic_quant(x)
    assert torch.isfinite(s).all() and (s > 0).all()
    torch.testing.assert_close(q.float() * s, x.float(), rtol=1e-6, atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('backend', ['gptq', 'awq', 'awq_fused'])
@pytest.mark.parametrize('scale', [16384., 9358.5, 2**-30])
def test_cpu_int4_affine_offset_range(dtype, backend, scale):
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
    from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
    cls = {'gptq': TorchAtenLinear, 'awq': TorchAtenAwqLinear, 'awq_fused': TorchFusedAwqLinear}[backend]
    m = cls(bits=4, group_size=128, sym=False, desc_act=False, in_features=128,
            out_features=128, bias=False,
            checkpoint_format=FORMAT.GPTQ_V2 if backend == 'gptq' else FORMAT.GEMM).eval()
    m.qweight.fill_(-1)
    m.qzeros.fill_(-1)
    m.scales = m.scales.float()
    m.scales.fill_(scale)
    m.post_init()
    check(m, torch.ones((1, 128), dtype=dtype), 0.)


@pytest.mark.parametrize('dtype', DTYPES + [torch.float32])
def test_cpu_int8_tiny_channel_scale(dtype):
    from gptqmodel.nn_modules.qlinear.torch_int8 import Int8PackedModule
    m = Int8PackedModule(torch.full((64, 128), 127, dtype=torch.int8),
                         torch.full((64,), 2**-24 / 127))
    check(m, torch.full((1, 128), 1024., dtype=dtype), .0078125)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('fmt', ['q4_k', 'q5_k'])
@pytest.mark.parametrize('dtype', DTYPES)
def test_gguf_superblock_scale_range(device, fmt, dtype):
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    m = GGUFTorchLinear(bits=fmt, group_size=-1, sym=True, desc_act=False,
                        in_features=256, out_features=64, bias=False).eval()
    block = np.zeros((64, m.gguf_type_size), dtype=np.uint8)
    block[:, :2] = np.array([60000.], np.float16).view(np.uint8)
    block[:, 2:4] = block[:, :2]
    # Independently encode eight 6-bit subscales=2 and minima=1.
    block[:, 4:8] = 2
    block[:, 8:12] = 1
    block[:, 12:16] = 0x12
    block[:, 16 if fmt == 'q4_k' else 48:] = 0x11
    m.qweight = torch.from_numpy(block).to(device)
    x = torch.full((1, 256), 2**-16, dtype=dtype, device=device)
    # Weight rounding is part of the dense fallback contract.
    weight = torch.tensor(60000., dtype=dtype).double().item()
    check(m, x, weight * 256 * 2**-16)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('fmt', ['q4_k', 'q5_k'])
@pytest.mark.parametrize('dtype', DTYPES)
def test_gguf_superblock_scale_times_code_range(device, fmt, dtype):
    """A scale term can overflow only after the code multiplier is applied.

    `d * 63` fits FP16 here, but `d * 63 * 15` (Q4_K) or `d * 63 * 31` (Q5_K)
    does not. Bounding the bare scale product would let this reconstruct to
    inf, so the guard must include the code extreme. `d_min` is chosen per
    format so the final weight stays representable: only the intermediate
    overflows, which is exactly the case a range guard has to catch.
    """
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    code_max = 31. if fmt == 'q5_k' else 15.
    d = 74.
    assert d * 63 <= torch.finfo(torch.float16).max
    assert d * 63 * code_max > torch.finfo(torch.float16).max
    # Keep d * 63 * code - d_min * 63 inside the target range, with d_min a
    # whole FP16 value so the encoded bytes round-trip exactly.
    d_min = float(round((d * 63 * code_max - 30000.) / 63))
    assert abs(d * 63 * code_max - d_min * 63) <= torch.finfo(dtype).max

    m = GGUFTorchLinear(bits=fmt, group_size=-1, sym=True, desc_act=False,
                        in_features=256, out_features=64, bias=False).eval()
    block = np.zeros((64, m.gguf_type_size), dtype=np.uint8)
    block[:, :2] = np.array([d], np.float16).view(np.uint8)
    block[:, 2:4] = np.array([d_min], np.float16).view(np.uint8)
    # 6-bit subscale and minimum both at their 63 maximum.
    block[:, 4:8] = 63
    block[:, 8:12] = 63
    block[:, 12:16] = 0
    if fmt == 'q4_k':
        block[:, 16:] = 0xFF
    else:
        # Q5_K carries a fifth bit plane before the low nibbles.
        block[:, 16:48] = 0xFF
        block[:, 48:] = 0xFF
    m.qweight = torch.from_numpy(block).to(device)

    blocks, _, _ = m._reshape_blocks(device=torch.device(device))
    # The guard must track the real condition: whether the scale term leaves
    # `dtype` once the code extreme is applied.
    assert m._k_scale_overflows(blocks, dtype) == (d * 63 * code_max > torch.finfo(dtype).max)
    if dtype is torch.float16:
        assert m._k_scale_overflows(blocks, dtype)
    dequant = (m._dequantize_q5_k_blocks if fmt == 'q5_k' else m._dequantize_q4_k_blocks)
    weight = dequant(blocks, torch.float32, compute_dtype=torch.float32)
    assert torch.isfinite(weight).all()
    weight = weight[:, : m.in_features].to(dtype).transpose(0, 1).contiguous()

    x = torch.full((1, 256), 2**-16, dtype=dtype, device=device)
    with torch.inference_mode():
        actual = m(x)
    expected = (x.to(torch.float64) @ weight.to(torch.float64)).to(dtype)
    assert torch.isfinite(actual).all()
    # BF16 carries ~8 bits of mantissa, so its representable relative spacing
    # is ~0.008; the 2e-3 inference tolerance is unreachable for that dtype.
    tol = max(2e-3, 2 * torch.finfo(dtype).eps)
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize('device', DEVICES)
def test_gguf_chunk_cancellation(device):
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    m = GGUFTorchLinear(bits='q4_0', group_size=-1, sym=True, desc_act=False,
                        in_features=512, out_features=16384, bias=False).eval()
    m.autotune_enabled = False
    block = torch.zeros((16384, 16, 18), dtype=torch.uint8)
    block[:, :, :2] = torch.tensor([1.], dtype=torch.float16).view(torch.uint8)
    block[:, :8, 2:] = 0x99
    block[:, 8:, 2:] = 0x77
    m.qweight = block.reshape(16384, -1).to(device)
    x = torch.full((1, 512), 512., dtype=torch.float16, device=device)
    assert m._is_fused_k_forward_candidate(x)
    check(m, x, 0.)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,value', [(torch.float16, 60000.), (torch.bfloat16, 100000.)])
def test_exl3_transform_range(device, dtype, value):
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

    from ..test_exllamav3_cpu import _independent_weight_oracle, _pack_symbols
    symbols = torch.randint(0, 16, (8, 8, 256), generator=torch.Generator().manual_seed(21))
    suh = torch.ones(128, dtype=torch.float16)
    svh = torch.full((128,), 2**-12, dtype=torch.float16)
    weight = _independent_weight_oracle(symbols, 4, suh, svh)
    m = ExllamaV3TorchLinear.from_tensors(in_features=128, out_features=128, name='range',
        tensors={'trellis': _pack_symbols(symbols, 4), 'suh': suh, 'svh': svh}).to(device).eval()
    x = torch.full((1, 128), value, dtype=dtype, device=device)
    check(m, x, (x.cpu().double() @ weight).to(device))


@pytest.mark.parametrize('backend', ['cpu', 'cuda', 'native'])
@pytest.mark.parametrize('rows', [1, 8])
@pytest.mark.parametrize('dtype', DTYPES)
def test_vptq_residual_normalization_range(backend, rows, dtype):
    from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear
    from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
    if backend != 'cpu' and not torch.cuda.is_available():
        pytest.skip('GPU required')
    if backend == 'native' and not NVIDIA:
        pytest.skip('Native VPTQ requires CUDA')
    cls = VPTQCudaLinear if backend == 'native' else VPTQLinear
    device = 'cpu' if backend == 'cpu' else 'cuda'
    m = cls(bits=1, in_features=128, out_features=64, bias=False, vector_lens=[-1, 4],
            num_centroids=[-1, 16], num_res_centroids=[-1, 16], group_size=128,
            group_num=-1, enable_norm=True, enable_perm=False, dtype=dtype).to(device).eval()
    m.indices.zero_()
    m.centroids.weight.data.fill_(40000.)
    m.res_centroids.weight.data.fill_(40000.)
    m.weight_scale.data.fill_(.5)
    m.weight_bias.data.zero_()
    x = torch.full((rows, 128), 2**-16, device=device, dtype=dtype)
    check(m, x, float(torch.tensor(40000., dtype=dtype)) * 128 * 2**-16)


@pytest.mark.skipif(not NVIDIA, reason='ExLlamaV2 requires NVIDIA CUDA')
@pytest.mark.parametrize('backend', ['gptq', 'awq'])
@pytest.mark.parametrize('rows', [1, 16, 64])
@pytest.mark.parametrize('dtype,value', [(torch.float16, 10000.), (torch.bfloat16, 100000.),
                                        (torch.bfloat16, 2**-30), (torch.float32, 100000.)])
def test_exllamav2_dot_and_activation_range(backend, rows, dtype, value):
    from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
    from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
    from gptqmodel.utils.exllamav2 import ScratchSpace
    cls = ExllamaV2Linear if backend == 'gptq' else AwqExllamaV2Linear
    m = cls(bits=4, group_size=128, sym=True, desc_act=False, in_features=256,
            out_features=256, bias=False, register_buffers=True,
            checkpoint_format=FORMAT.GPTQ_V2 if backend == 'gptq' else FORMAT.GEMM).cuda().eval()
    m.qweight.fill_(-1)
    m.qzeros.fill_(-2004318072)  # Every zero code=8, every weight code=15.
    m.scales.fill_(2**-10)
    scratch = ScratchSpace(m.scratch_space_fixed(), 0)
    m.post_init(scratch)
    x = torch.full((rows, 256), value, dtype=dtype, device='cuda')
    check(m, x, float(x[0, 0]) * 256 * 7 / 1024)
    assert not any(t.numel() >= 256 * 256 for t in state_tensors(vars(scratch)))


@pytest.mark.skipif(not NVIDIA, reason='Native AWQ kernels require NVIDIA CUDA')
@pytest.mark.parametrize('backend', ['fast', 'marlin'])
@pytest.mark.parametrize('rows', [1, 4, 16, 64])
@pytest.mark.parametrize('rank', [2, 3])
def test_awq_partial_sum_range(backend, rows, rank):
    from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
    from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
    k, n = 2048, 256
    cls = AwqGEMVFastLinear if backend == 'fast' else AwqMarlinLinear
    kwargs = {'pack_dtype': torch.int16} if backend == 'fast' else {}
    m = cls(bits=4, group_size=128, sym=False, desc_act=False, in_features=k,
            out_features=n, bias=False, register_buffers=True, **kwargs)
    if backend == 'fast':
        linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
        linear.weight.data.fill_(1.)
        m.pack(linear, torch.ones((n, k // 128), dtype=torch.float16),
               torch.zeros((n, k // 128), dtype=torch.float16))
    else:
        m.qweight.data.fill_(0x11111111)
        m.qzeros.data.zero_()
        m.scales.data.fill_(1.)
    m = m.cuda().eval()
    m.post_init()
    x = torch.full((rows, k), 10000., device='cuda', dtype=torch.float16)
    x[:, k // 2:] *= -1
    if rank == 3:
        x = x.unsqueeze(1)
    check(m, x, 0.)


@pytest.mark.skipif(not NVIDIA, reason='Native QQQ requires NVIDIA CUDA')
def test_native_qqq_small_activation_scale():
    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
    m = QQQLinear(bits=4, group_size=-1, sym=True, desc_act=False,
                   in_features=128, out_features=128, bias=False, register_buffers=False)
    linear = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16)
    linear.weight.data.fill_(1024.)
    m.pack(linear, torch.full((128, 1), 1024., dtype=torch.float16))
    m = m.cuda().eval()
    check(m, torch.full((1, 128), 2**-20, dtype=torch.float16, device='cuda'), .125)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('fmt', ['float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz', 'float8_e8m0fnu'])
@pytest.mark.parametrize('layout', ['scalar', 'row', 'column', 'block', 'full'])
@pytest.mark.parametrize('simd', ['default', 'avx2', 'scalar'])
def test_floatx_cpu_scale_range(monkeypatch, dtype, fmt, layout, simd):
    from gptqmodel.quantization.dtype import dequantize_fp8
    monkeypatch.setenv('GPTQMODEL_FLOATX_CPU_FORCE_NATIVE_FP8', '1')
    if simd in ('avx2', 'scalar'):
        monkeypatch.setenv('GPTQMODEL_FLOATX_CPU_DISABLE_AVX512', '1')
    if simd == 'scalar':
        monkeypatch.setenv('GPTQMODEL_FLOATX_CPU_DISABLE_AVX2', '1')
    # All byte codes plus tails; mask NaN/Inf codes, which have no finite oracle.
    codes = torch.arange(256, dtype=torch.uint8).repeat(3)[:2 * 257].reshape(2, 257)
    source = codes.view(getattr(torch, fmt))
    values = source.float()
    codes[~torch.isfinite(values)] = 0
    values = source.double()
    scale_value = 2**-120 if fmt == 'float8_e8m0fnu' else 2**-25
    shapes = {'scalar': (), 'row': (2,), 'column': (257,), 'block': (2, 1), 'full': (2, 257)}
    scale = torch.full(shapes[layout], scale_value)
    actual = dequantize_fp8(source, scale=scale, axis=1 if layout == 'column' else 0, target_dtype=dtype)
    expected = (values * scale_value).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('simd', ['default', 'avx2', 'scalar'])
def test_fp4_cpu_tiny_scale(monkeypatch, dtype, simd):
    from gptqmodel.quantization.dtype import dequantize_f4_e2m1
    if simd in ('avx2', 'scalar'):
        monkeypatch.setenv('GPTQMODEL_FLOATX_CPU_DISABLE_AVX512', '1')
    if simd == 'scalar':
        monkeypatch.setenv('GPTQMODEL_FLOATX_CPU_DISABLE_AVX2', '1')
    packed = torch.arange(256, dtype=torch.uint8).reshape(2, 128)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
    expected = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1).reshape(2, 256)
    actual = dequantize_f4_e2m1(packed, scale=torch.full((2, 16), 2**-25), axis=None, target_dtype=dtype)
    torch.testing.assert_close(actual, (expected * 2**-25).to(dtype), rtol=0, atol=0)


@pytest.mark.skipif(not NVIDIA, reason='cuBLAS reduced-precision regression')
@pytest.mark.parametrize('backend', ['gptq', 'awq'])
def test_torch_fallback_ignores_unsafe_reduction_flag(request, backend):
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    request.addfinalizer(lambda: setattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction', previous))
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    cls = TorchLinear if backend == 'gptq' else AwqTorchLinear
    m = cls(bits=4, group_size=128, sym=False, desc_act=False, in_features=2048,
            out_features=256, bias=False, register_buffers=True,
            checkpoint_format=FORMAT.GPTQ_V2 if backend == 'gptq' else FORMAT.GEMM).cuda().eval()
    m.qweight.fill_(0x11111111)
    m.qzeros.zero_()
    m.scales.fill_(1.)
    m.post_init()
    x = torch.full((16, 2048), 10000., dtype=torch.float16, device='cuda')
    x[:, 1024:] *= -1
    check(m, x, 0.)
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction


@pytest.mark.skipif(not NVIDIA, reason='ExLlamaV2 CUDA split reduction')
@pytest.mark.parametrize('backend', ['gptq', 'awq'])
def test_exllamav2_split_reduction_and_stream(backend):
    from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
    from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
    from gptqmodel.utils.exllamav2 import ScratchSpace
    cls = ExllamaV2Linear if backend == 'gptq' else AwqExllamaV2Linear
    m = cls(bits=4, group_size=128, sym=True, desc_act=False, in_features=2048,
            out_features=256, bias=False, register_buffers=True,
            checkpoint_format=FORMAT.GPTQ_V2 if backend == 'gptq' else FORMAT.GEMM).cuda().eval()
    m.qweight.fill_(-1717986919)  # 0x99999999
    m.qzeros.fill_(-2004318072)   # 0x88888888
    m.scales.fill_(1.)
    scratch = ScratchSpace(m.scratch_space_fixed(), 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        m.post_init(scratch)
        x = torch.full((16, 2048), 10000., dtype=torch.float16, device='cuda')
        x[:, 1024:] *= -1
        check(m, x, 0.)
    stream.synchronize()
    assert not state_tensors(vars(scratch))


@pytest.mark.skipif(not NVIDIA, reason='ExLlamaV2 requires NVIDIA CUDA')
@pytest.mark.parametrize('backend', ['gptq', 'awq'])
@pytest.mark.parametrize('rows', [64, 65])
def test_exllamav2_retains_no_dense_weight_scratch(backend, rows):
    """Prefill must not reconstruct weights into model-owned storage.

    The row-count fallback used to write a full FP16 weight into the scratch
    slice handed to ``make_q_matrix`` at construction. That slice was sized from
    the largest layer and kept on ``model.device_tensors`` for the model's whole
    lifetime. The GEMM now reconstructs into a transient per-call tensor, so the
    handle, the module, and the model scratch must all stay free of a dense
    weight.
    """
    from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
    from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
    from gptqmodel.utils.model import gptqmodel_post_init

    cls = ExllamaV2Linear if backend == 'gptq' else AwqExllamaV2Linear
    in_features, out_features = 1024, 256
    m = cls(bits=4, group_size=128, sym=True, desc_act=False,
            in_features=in_features, out_features=out_features, bias=False,
            register_buffers=True,
            checkpoint_format=FORMAT.GPTQ_V2 if backend == 'gptq' else FORMAT.GEMM).cuda().eval()
    # Every code is 15 and every zero point is 8, so the dequantized weight is
    # exactly 7 * 2**-10 in FP64; the sum over K is analytic.
    m.qweight.fill_(-1)
    m.qzeros.fill_(-2004318072)  # 0x88888888
    m.scales.fill_(2 ** -10)

    model = torch.nn.Module()
    model.layer = m
    # The model-level path owns the scratch allocation, so let it call
    # post_init exactly once: the AWQ post_init re-packs qweight and is not
    # idempotent.
    gptqmodel_post_init(model, use_act_order=False)

    dense_bytes = in_features * out_features * 2
    retained = [t for t in state_tensors(vars(model)) if t.numel() * t.element_size() >= dense_bytes]
    assert retained == [], [(tuple(t.shape), t.dtype) for t in retained]
    assert m.temp_dq_size() == 0

    x = torch.full((rows, in_features), 2 ** -8, dtype=torch.float16, device='cuda')
    expected = float(x[0, 0]) * in_features * 7 / 1024
    check(m, x, expected)
    after = [t for t in state_tensors(vars(model)) if t.numel() * t.element_size() >= dense_bytes]
    assert after == [], 'prefill retained a dense weight'
