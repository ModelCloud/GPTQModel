# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 matrix oracles, reduction boundaries and storage ties."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('exl3_hessian_under_test',Path(__file__).parents[1]/'gptqmodel/utils/exl3_hessian.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Accumulator=module.BoundedHessianAccumulator


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_all_rows_are_bounded_fp32_products_with_fp32_owned_storage(monkeypatch,dtype):
    x=torch.linspace(-2,2,301*16).reshape(301,16).to(dtype)
    actual=[];mm=torch.mm
    def observed(left,right,*a,**kw):
        actual.append(right.clone());assert left.dtype==right.dtype==torch.float32
        assert torch.equal(left,right.T)
        return mm(left,right,*a,**kw)
    monkeypatch.setattr(torch,'mm',observed)
    state=Accumulator(16,device='cpu');state.add(x[:200]);state.update(x[200:])
    assert [len(p) for p in actual]==[128,72,101]
    assert torch.equal(torch.cat(actual),x.float()) and state.count==301
    result=state.snapshot();oracle=x.double().T@x.double()
    assert result.dtype==torch.float32 and bool(torch.isfinite(result).all())
    assert float(torch.linalg.norm(result.double()-oracle)/torch.linalg.norm(oracle))<1e-6
    with torch.inference_mode():
        result.zero_()
    assert torch.count_nonzero(state.snapshot())>0


def test_outer_sum_preserves_small_contributions_and_cancellation():
    state=Accumulator(8,device='cpu');large=torch.full((1,8),10000.)
    small=torch.full((128,8),1/16);small[:,1::2].neg_()
    state.add(large);legacy=large.T@large
    for _ in range(2048):state.add(small);legacy.addmm_(small.T,small)
    oracle=large.double().T@large.double()+2048*(small.double().T@small.double())
    result=state.snapshot()
    assert state.count==262145 and torch.equal(result,oracle.float())
    assert float(torch.linalg.norm(legacy.double()-oracle)/torch.linalg.norm(oracle))>1e-6


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('direction',[-1,0,1])
def test_one_input_step_around_fp32_storage_half_ulp(dtype,direction):
    value=torch.tensor(2**-12,dtype=dtype)
    if direction:value=torch.nextafter(value,torch.tensor(float('inf')*direction,dtype=dtype))
    rows=torch.stack((torch.ones((),dtype=dtype),value)).reshape(2,1)
    state=Accumulator(1,device='cpu');state.add(rows)
    oracle=(rows.double().T@rows.double()).float()
    assert torch.equal(state.snapshot().view(torch.int32),oracle.view(torch.int32))
    expected=torch.nextafter(torch.tensor(1.),torch.tensor(float('inf'))) if direction>0 else torch.tensor(1.)
    assert state.snapshot().item()==expected.item()


def test_signed_zero_and_invalid_rows_do_not_change_count():
    state=Accumulator(4,device='cpu')
    for bad in (torch.empty(0,4),torch.ones(2,3),torch.ones(2,4,dtype=torch.int64),torch.full((2,4),float('nan'))):
        with pytest.raises(ValueError):state.add(bad)
        assert state.count==0 and torch.count_nonzero(state.snapshot())==0
    rows=torch.tensor([[0.,-0.,1.,-1.],[-0.,0.,-1.,1.]])
    state.add(rows)
    assert torch.equal(state.snapshot().view(torch.int32),(rows.double().T@rows.double()).float().view(torch.int32))


def test_failed_update_poison_is_explicit():
    state=Accumulator(2,device='cpu')
    with pytest.raises(ValueError,match='nonfinite FP32'):state.add(torch.full((2,2),1e30))
    assert state.count==0
    with pytest.raises(RuntimeError,match='failed partial'):state.snapshot()
    with pytest.raises(RuntimeError,match='failed partial'):state.add(torch.ones(2,2))


@pytest.mark.parametrize('width,max_rows',[(0,128),(True,128),(8,0),(8,129),(8,True)])
def test_invalid_geometry_and_product_bounds(width,max_rows):
    with pytest.raises(ValueError):Accumulator(width,device='cpu',max_rows=max_rows)
