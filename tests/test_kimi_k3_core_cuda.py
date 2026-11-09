# SPDX-License-Identifier: Apache-2.0
"""CUDA oracle gates for the standard K6 padded-output calibration path.

Run on the admitted Oliver GPU with ``pytest -q -s tests/test_kimi_k3_core_cuda.py``.
KIMI_SOURCE_SNAPSHOT may name the cached base K2 checkpoint. This module never
changes CUDA kernel arithmetic; its Torch decoder deliberately reads the
serialized bitstream independently of the CUDA unpack/reconstruct helpers.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch.nn import functional as F

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.utils import kimi_k3_core_format as core


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires admitted CUDA GPU')


@pytest.fixture(scope='module', autouse=True)
def disable_tf32_for_numerical_oracle():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def oracle_states(trellis):
    """Read circular 16-bit states from MSB-first symbols, including word swap."""
    bits = trellis.shape[-1] // 16
    words = (trellis.detach().cpu().to(torch.int64) & 65535).reshape(-1, 16 * bits)
    result = []
    # Each uint32 in the file holds its two uint16 words in reversed order.
    # Materializing bits makes this decoder independent of CUDA funnel shifts.
    bit_positions = torch.arange(15, -1, -1)
    ends = torch.arange(256) * bits + bits
    windows = (ends[:, None] - 16 + torch.arange(16)[None, :]) % (256 * bits)
    significance = 1 << bit_positions
    for chunk in words.split(256):
        stream = chunk.reshape(-1, 16 * bits // 2, 2).flip(-1).reshape(-1, 16 * bits)
        stream = ((stream[..., None] >> bit_positions) & 1).reshape(-1, 256 * bits)
        states = (stream[:, windows] * significance).sum(-1)
        result.append(states.to(torch.int32))
    return torch.cat(result).reshape(*trellis.shape[:2], 256)


def oracle_inner(packed):
    """Independent MCG/FP16 codebook decode and tensor-core tile permutation."""
    states = oracle_states(packed['trellis']).to(torch.int64)
    scrambled = (states * 0xCBAC1FED) & 0xFFFFFFFF
    half_bits = ((scrambled & 0x8FFF8FFF) ^ 0x3B603B60)
    halves = torch.stack((half_bits & 65535, half_bits >> 16), -1).to(torch.uint16).contiguous()
    values = halves.view(torch.float16).sum(-1, dtype=torch.float16)
    # Stored fragment order -> each 16x16 tile's row-major order.
    row_major = [0] * 256
    for lane in range(32):
        r, c = (lane % 4) * 2, lane // 4
        coords = [(r,c),(r+1,c),(r+8,c),(r+9,c),
                  (r,c+8),(r+1,c+8),(r+8,c+8),(r+9,c+8)]
        for element,(row,col) in enumerate(coords):
            row_major[row * 16 + col] = lane * 8 + element
    tiles = values[..., row_major]
    k16,n16 = tiles.shape[:2]
    return tiles.reshape(k16,n16,16,16).permute(0,2,1,3).reshape(k16 * 16,n16 * 16).contiguous()


def oracle_hadamard(device):
    had = torch.ones((1,1),dtype=torch.float64)
    while had.shape[0] < 128:
        had = torch.cat((torch.cat((had,had),1),torch.cat((had,-had),1)),0)
    # The runtime uses an FP32 normalized matrix, FP32 GEMMs, and an FP16
    # rounding boundary after each transform and scale multiplication.
    return (had / math.sqrt(128)).float().to(device)


def oracle_reconstruction(packed, *, device='cuda:0', dtype=torch.float16):
    inner = oracle_inner(packed).to(device)
    k,n = inner.shape
    had = oracle_hadamard(device)
    left = (had @ inner.float().reshape(-1,128,n)).reshape(k,n).half()
    left = (left * packed['suh'].to(device).reshape(k,1)).half()
    right = (left.float().reshape(k,-1,128) @ had).reshape(k,n).half()
    return (right * packed['svh'].to(device).reshape(1,n)).half().to(dtype)


def print_result(name, **values):
    print('KIMI_CORE_CUDA_ORACLE ' + json.dumps(dict(test=name,**values),sort_keys=True))


def captured_hessian(activations, *, device='cuda:0'):
    """FP32 XTX checked in bounded row blocks against independent CPU FP64."""
    x = activations.float().to(device)
    actual = x.T @ x
    x64 = activations.double().cpu()
    error_sq,reference_sq = 0.,0.
    for start in range(0,x.shape[1],256):
        reference = x64[:,start:start+256].T @ x64
        candidate = actual[start:start+256].double().cpu()
        error_sq += float((candidate-reference).square().sum())
        reference_sq += float(reference.square().sum())
    normalized = math.sqrt(error_sq / reference_sq)
    assert normalized < 1e-6 and bool(torch.isfinite(actual).all())
    return {'H': actual,'count': len(activations),'finalized': False},normalized


def direct_quantizer_baseline(weight, raw):
    # An independent call to the unchanged quantizer gives a discrete byte
    # oracle for the new wrapper's transpose, padding, seed and raw-H handling.
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    outputs,inputs = weight.shape
    output_pad = math.ceil(outputs / 128) * 128 - outputs
    padded = F.pad(weight.float(),(0,0,0,output_pad)).T.contiguous()
    args = {'K': 6,'seed': 787,'sigma_reg': .025,'devices': [torch.device('cuda:0')],
                'mcg': True,'apply_out_scales': None}
    h = {'H': raw['H'].clone(),'count': raw['count'],'finalized': False}
    _,_,packed = quantize_exl3(padded,h,args,return_weight_q=False)
    return {key:value.detach().cpu() for key,value in packed.items()}


@pytest.mark.cpu
def test_independent_bitstream_decoder_matches_clean_room_reference():
    # CPU validation checks both K2 and K6 and uint16 endpoints. The actual
    # CUDA gate below independently checks every decoded half against CUDA.
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    gen = torch.Generator().manual_seed(19)
    for bits in (2,6):
        trellis = torch.randint(-32768,32768,(8,40,16 * bits),generator=gen,dtype=torch.int16)
        trellis.reshape(-1)[:8] = torch.tensor([0,1,32767,-32768,-1,0x5555,-21846,31],dtype=torch.int16)
        packed = {'trellis': trellis,'suh': torch.ones(128,dtype=torch.float16),
                      'svh': torch.ones(640,dtype=torch.float16),
                      'mcg': torch.tensor(0xCBAC1FED,dtype=torch.uint32).view(torch.int32)}
        reference = ExllamaV3TorchLinear.from_tensors(in_features=128,out_features=640,name='oracle',tensors=packed)
        assert torch.equal(oracle_states(trellis).long(),reference._unpack_indices())
        # This prior clean-room reference emits FP32 codebook sums. Account
        # explicitly for the specified CUDA half addition, then compare bits.
        assert torch.equal(oracle_inner(packed),reference.get_inner_weight_tensor().half())


@pytest.fixture(scope='module')
def small_projection():
    gen = torch.Generator().manual_seed(787)
    weight = (torch.randn((576,128),generator=gen) * .02).bfloat16().cuda()
    activations = torch.randn((512,128),generator=gen).bfloat16()
    raw,norm = captured_hessian(activations)
    original = raw['H'].clone()
    packed,metrics,evidence = core.quantize_core_projection(weight,raw)
    repeated,_,_ = core.quantize_core_projection(weight,raw)
    baseline = direct_quantizer_baseline(weight,raw)
    assert all(torch.equal(packed[k],repeated[k]) and torch.equal(packed[k],baseline[k]) for k in packed)
    assert torch.equal(raw['H'],original) and raw['finalized'] is False
    assert metrics['hessian_sample_count']==512 and evidence['identity_rows']==0
    print_result('small_quantization',native_shape=list(weight.shape),stored_shape=evidence['stored_shape'],
                 hessian_fp64_normalized_error=norm,packed_mismatches=0,identity_rows=0)
    return packed


@pytest.mark.cuda
@requires_cuda
def test_small_padded_reconstruction_matches_independent_torch(small_projection):
    from gptqmodel.exllamav3.ext import exllamav3_ext as ext
    packed = small_projection
    device_packed = {key:value.cuda() for key,value in packed.items()}
    actual_inner = torch.empty((128,640),dtype=torch.float16,device='cuda:0')
    ext.reconstruct(actual_inner,device_packed['trellis'],6,True,False)
    expected_inner = oracle_inner(packed).cuda()
    assert torch.equal(actual_inner,expected_inner)
    for dtype in (torch.float16,torch.bfloat16):
        expected = oracle_reconstruction(packed,dtype=dtype).T[:576].contiguous()
        actual = core.reconstruct_core_projection(packed,(576,128),dtype=dtype)
        torch.cuda.synchronize()
        normalized = float((actual.float()-expected.float()).norm() / expected.float().norm())
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        assert normalized <= 1e-6
        print_result('small_reconstruction',dtype=str(dtype),inner_half_mismatches=0,
                     output_mismatches=int(torch.count_nonzero(actual!=expected)),normalized_error=normalized,
                     max_abs_error=float((actual.float()-expected.float()).abs().max()))


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('batch',[1,33])
def test_padded_native_and_reconstruct_wrapper_inference(small_projection,dtype,batch):
    gen = torch.Generator().manual_seed(41)
    x = (torch.randn(batch,128,generator=gen) * .125).to(device='cuda:0',dtype=dtype)
    # Threshold neighbors and signed zeros survive input casting; output
    # quantization/codebook behavior is covered separately by exact decode.
    x[0,:8] = torch.tensor([0.,-0.,.125,-.125,.25,-.25,.5,-.5],device=x.device,dtype=dtype)
    packed = {key:value.cuda() for key,value in small_projection.items()}
    module = ExllamaV3Linear.from_tensors(in_features=128,out_features=576,name='kv_a',tensors=packed)
    expected_weight = oracle_reconstruction(small_projection).T[:576].contiguous()
    expected_raw = F.linear(x.half().float(),expected_weight.float())
    expected_visible = expected_raw.half().to(dtype)
    actual = module(x)
    torch.cuda.synchronize()
    assert actual.shape==(batch,576) and actual.dtype==dtype
    # Check visible output against the same rounding boundary, and also
    # retain raw arithmetic error so dtype rounding cannot hide a failure.
    torch.testing.assert_close(actual.float(),expected_raw,rtol=2e-3,atol=2e-3)
    torch.testing.assert_close(actual,expected_visible,rtol=2e-3,atol=2e-3)
    print_result('wrapper_inference',dtype=str(dtype),batch=batch,
                 path='native' if batch<=32 else 'reconstructed_inner_gemm',
                 raw_max_abs_error=float((actual.float()-expected_raw).abs().max()),
                 visible_max_abs_error=float((actual.float()-expected_visible.float()).abs().max()))
    module._drop_inner()


@pytest.mark.cuda
@pytest.mark.model
@requires_cuda
def test_native_kimi_mla_shape_calibrated_quantization_and_crop():
    root = Path(os.environ.get('KIMI_SOURCE_SNAPSHOT',str(Path.home()/'.cache/huggingface/cuteafd-exports/Kimi-K3-EXL3-K2')))
    if not (root/'model.safetensors.index.json').is_file():
        pytest.skip('set KIMI_SOURCE_SNAPSHOT to the cached base K2 model')
    name = 'language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight'
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(root/index[name],framework='pt',device='cpu') as shard:
        weight = shard.get_tensor(name).clone()
    assert weight.shape==(576,7168) and weight.dtype==torch.bfloat16
    gen = torch.Generator().manual_seed(787)
    # Dense random samples only; no identity rows or cold-expert recovery.
    activations = torch.randn((128,7168),generator=gen).bfloat16()
    raw,norm = captured_hessian(activations)
    original = raw['H'].clone()
    started = time.monotonic()
    packed,metrics,evidence = core.quantize_core_projection(weight,raw)
    repeated,_,_ = core.quantize_core_projection(weight,raw)
    assert all(torch.equal(packed[key],repeated[key]) for key in packed)
    assert torch.equal(raw['H'],original) and raw['finalized'] is False
    expected = oracle_reconstruction(packed,dtype=torch.bfloat16).T[:576].contiguous()
    actual = core.reconstruct_core_projection(packed,weight.shape)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    assert evidence['stored_shape']==[640,7168] and evidence['identity_rows']==0
    assert packed['trellis'].shape==(448,40,96)
    assert torch.count_nonzero(packed['svh'][576:])==0
    print_result('native_mla_projection',native_shape=list(weight.shape),stored_shape=evidence['stored_shape'],
                 dtype=str(actual.dtype),hessian_fp64_normalized_error=norm,packed_mismatches=0,
                 reconstruction_max_abs_error=float((actual.float()-expected.float()).abs().max()),
                 hessian_relative_error=metrics['hessian_weighted_relative_error'],
                 elapsed_seconds=time.monotonic()-started,identity_rows=0)
