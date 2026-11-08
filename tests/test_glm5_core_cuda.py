# SPDX-License-Identifier: Apache-2.0
"""Admitted RTX checks; parent runs these after CPU qualification."""
import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.utils import glm5_core as core


pytestmark=[pytest.mark.cuda,pytest.mark.skipif(not torch.cuda.is_available(),reason="admitted CUDA GPU required")]


@pytest.mark.parametrize("bits",[4,6])
def test_real_glm_kv_a_geometry_owns_raw_hessian_and_repeats_exactly(bits):
    generator=torch.Generator(device="cuda").manual_seed(787)
    weight=(torch.randn(576,6144,device="cuda",generator=generator)*.01).bfloat16()
    raw={"H": torch.eye(6144,device="cuda"),"count": 4096,"finalized": False}
    original=raw["H"].clone()
    first,_,evidence=core.quantize_projection(weight,raw,bits=bits)
    second,_,_=core.quantize_projection(weight,raw,bits=bits)
    assert all(torch.equal(first[key],second[key]) for key in first)
    assert torch.equal(raw["H"],original) and raw["finalized"] is False
    assert evidence["stored_shape"]==[640,6144] and evidence["output_padding"]==64
    dense=core.reconstruct_projection(first,weight.shape,device="cuda:0")
    assert dense.shape==weight.shape and dense.dtype==torch.bfloat16
    torch.cuda.synchronize()


@pytest.mark.parametrize("bits",[4,6])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("batch",[1,33])
def test_native_mcg_wrapper_matches_serialized_bf16_linear(bits,dtype,batch):
    generator=torch.Generator(device="cuda").manual_seed(41)
    weight=(torch.randn(576,128,device="cuda",generator=generator)*.015).bfloat16()
    packed,_,_=core.quantize_projection(weight,{"H": torch.eye(128,device="cuda"),"count": 4096,"finalized": False},bits=bits)
    module=ExllamaV3Linear.from_tensors(in_features=128,out_features=576,name="glm.kv_a",tensors=packed).cuda()
    dense=core.reconstruct_projection(packed,weight.shape,device="cuda:0",dtype=torch.float16)
    x=(torch.randn(batch,128,device="cuda",generator=generator)*.02).to(dtype)
    actual=module(x)
    oracle=torch.nn.functional.linear(x.half(),dense).to(dtype)
    torch.cuda.synchronize()
    assert actual.dtype==dtype and actual.shape==(batch,576)
    torch.testing.assert_close(actual,oracle,rtol=2e-3,atol=2e-3)
