# SPDX-License-Identifier: Apache-2.0
"""Ownership, packed-QKV observation, native boundaries and MTP alignment."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.utils import step5_core as core


def source_slice(tmp_path):
    values = torch.arange(3*4*8).reshape(3,4,8).bfloat16()
    name = "model.layers.3.moe.gate_proj.weight"
    save_file({name:values},tmp_path/"weights.safetensors")
    source = core.Step5BF16Source.__new__(core.Step5BF16Source)
    source.snapshot = tmp_path
    source.config = {"moe_layer_list": [3],"moe_num_experts": 3}
    source.weight_map={name:"weights.safetensors"}
    source._reader=source._reader_context=source._reader_shard=None
    return source,values


def test_fused_expert_slice_owns_only_one_matrix_and_survives_mapping_close(tmp_path):
    source,values=source_slice(tmp_path)
    tensor=source.expert_tensor(3,1,"gate_proj")
    source.close()
    assert tensor.shape==(4,8) and torch.equal(tensor,values[1])
    tensor.zero_()
    assert torch.equal(source.expert_tensor(3,1,"gate_proj"),values[1])
    assert tensor.untyped_storage().nbytes()==tensor.numel()*tensor.element_size()
    source.close()


@pytest.mark.parametrize("args",[(True,1,"gate_proj"),(3,True,"gate_proj"),(3,3,"gate_proj"),
                                 (4,1,"gate_proj"),(3,1,"invalid")])
def test_expert_slice_rejects_outside_reviewed_geometry(tmp_path,args):
    source,_=source_slice(tmp_path)
    with pytest.raises(ValueError):
        source.expert_tensor(*args)
    source.close()


class PackedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj=nn.Linear(8,8,bias=False)
        self.k_proj=nn.Linear(8,4,bias=False)
        self.v_proj=nn.Linear(8,4,bias=False)
        self.o_proj=nn.Linear(8,8,bias=False)
        self.register_buffer("_qkv_weight",torch.empty(0),persistent=False)
        self.calls=0

    def _project_packed_qkv(self,x):
        self.calls+=1
        if not self._qkv_weight.numel():
            self._qkv_weight=torch.cat([self.q_proj.weight,self.k_proj.weight,self.v_proj.weight])
        return torch.nn.functional.linear(x,self._qkv_weight)


def block():
    b=nn.Module()
    b.layer_idx=4
    b.self_attn=PackedAttention()
    return b


def test_capture_packed_qkv_observes_same_operand_without_extra_gemm():
    b=block()
    x=torch.randn(5,8)
    observed={}
    names=["model.layers.4.self_attn."+part+"_proj" for part in "qkv"]
    with core.capture_inputs(b,names,lambda name,value:observed.setdefault(name,value)):
        actual=b.self_attn._project_packed_qkv(x)
    assert b.self_attn.calls==1
    assert all(observed[name] is x for name in names)
    assert torch.equal(actual,torch.nn.functional.linear(x,b.self_attn._qkv_weight))
    assert not b.self_attn.q_proj._forward_pre_hooks


def test_capture_restores_method_and_hooks_on_failure():
    b=block()
    original=b.self_attn._project_packed_qkv
    with pytest.raises(RuntimeError):
        with core.capture_inputs(b,["self_attn.q_proj","self_attn.o_proj"],lambda *_:None):
            raise RuntimeError("abort")
    assert b.self_attn._project_packed_qkv==original
    assert not b.self_attn.o_proj._forward_pre_hooks


def test_replacing_projection_retires_stale_packed_storage():
    b=block()
    x=torch.randn(5,8)
    old=b.self_attn._project_packed_qkv(x)
    replacement=torch.zeros_like(b.self_attn.q_proj.weight)
    core.replace_projection(b,"self_attn.q_proj",replacement)
    assert b.self_attn._qkv_weight.numel()==0
    actual=b.self_attn._project_packed_qkv(x)
    assert torch.count_nonzero(actual[:,:8])==0
    assert torch.equal(actual[:,8:],old[:,8:])


class Norm(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.weight=nn.Parameter(torch.linspace(-.1,.2,width))
    def forward(self,x):
        value=x.float()
        return (value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)
                *(1+self.weight)).to(x.dtype)


def test_three_mtp_teacher_shifts_preserve_raw_residual_and_positions():
    embedding=torch.randn(16,8).bfloat16()
    previous=torch.randn(7,8)
    ids=torch.arange(7)
    positions=torch.arange(7)
    for stage in range(3):
        aligned=core.aligned_inputs(previous,ids,embedding,position_ids=positions)
        assert torch.equal(aligned["target_hidden"],previous[:-1])
        assert torch.equal(aligned["inputs_embeds"],embedding[ids[1:]])
        assert torch.equal(aligned["position_ids"],torch.arange(stage+1,7))
        previous=aligned["target_hidden"]+10.0
        ids=aligned["input_ids"]
        positions=aligned["position_ids"]


def test_mtp_envelope_narrows_before_each_zero_centered_norm():
    b=SimpleNamespace(enorm=Norm(8),hnorm=Norm(8))
    embedding=torch.randn(5,8).bfloat16()
    residual=torch.randn(5,8)*100.1
    actual=core.mtp_envelope(b,embedding,residual)
    def oracle(x,w):
        x=x.bfloat16().float()
        return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*(1+w)).bfloat16()
    expected=torch.cat((oracle(embedding,b.enorm.weight),oracle(residual,b.hnorm.weight)),-1)
    assert torch.equal(actual,expected)


def test_rope_growth_retires_old_tables_before_native_builder():
    class Attention:
        def __init__(self):
            self.rope_cos=torch.ones(4,2)
            self.rope_sin=torch.ones(4,2)
        def build_rope(self,*,max_position,device,shared_cache):
            assert self.rope_cos.numel()==0 and self.rope_sin.numel()==0
            self.rope_cos=torch.ones(max_position,2)
            self.rope_sin=torch.ones(max_position,2)
    b=SimpleNamespace(self_attn=Attention(),_gptqmodel_step5_source=SimpleNamespace(_rope={}),
                      _gptqmodel_step5_extent=4)
    context=core.prepare_context(torch.ones(130,8),position_ids=torch.arange(130))
    core._prepare_rope(b,context)
    assert b.self_attn.rope_cos.shape[0]==256


@pytest.mark.parametrize("positions",[torch.tensor([0,2,3]),torch.tensor([-1,0,1]),torch.tensor([0.,1.,2.])])
def test_context_rejects_gaps_negative_or_float_positions(positions):
    with pytest.raises(ValueError):
        core.prepare_context(torch.ones(3,8),position_ids=positions)


def test_router_preserves_native_topk_and_recovers_only_unselected_experts():
    class Router:
        top_k=2
        gate=nn.Linear(4,6,bias=False)
        router_bias=torch.zeros(6)
        def route(self,x):
            ids=torch.tensor([[5,3],[2,0]])
            return torch.tensor([[.8,.2],[.6,.4]]),ids
    router=Router()
    with torch.no_grad():router.gate.weight.zero_()
    selected,weights,recovery=core.router_choices(router,torch.ones(2,4),rank_max=4)
    assert torch.equal(selected,router.route(None)[1])
    assert torch.equal(weights,router.route(None)[0])
    assert torch.equal(recovery,torch.tensor([[0,1],[1,3]]))


@pytest.mark.parametrize("bits",[4,5,6])
def test_all_pilot_bits_use_calibrated_contract_without_mutating_raw_hessian(monkeypatch,bits):
    original=torch.eye(128,dtype=torch.float32)
    hessian={"H": original,"count": 17,"finalized": False}
    class Quantizer:
        def quantize_exl3(self,weight,raw,args,return_weight_q):
            assert raw["H"].data_ptr()!=original.data_ptr()
            assert args["K"]==bits and args["mcg"] is True and return_weight_q is False
            raw["H"].zero_()
            args["error_metrics"]={
                "quantizer_path":"hessian_ldlq","hessian_metric_status":"ok",
                "hessian_numerical_contract":"signed-block-hadamard-congruence-fp64-v1",
                "hessian_transform_compute_dtype":"torch.float64","hessian_storage_dtype":"torch.float32",
                "hessian_sample_count":17,"reported_metric_value":.001,
                "hessian_weighted_error_numerator":1.,"hessian_weighted_reference_denominator":1000.,
                "hessian_weighted_relative_error":.001}
            return None,None,{"trellis": torch.zeros(8,8,16*bits,dtype=torch.int16),
                "suh": torch.ones(128,dtype=torch.float16),"svh": torch.ones(128,dtype=torch.float16),
                "mcg": torch.tensor(-877912083,dtype=torch.int32)}
    monkeypatch.setattr(core.base,"_quantizer",lambda:Quantizer())
    packed,metric,evidence=core.quantize_projection(torch.ones(128,128).bfloat16(),hessian,bits=bits,device="cpu")
    assert torch.equal(original,torch.eye(128))
    assert evidence["bits"]==bits and evidence["hessian_rows"]==17 and evidence["identity_rows"]==0


def test_native_config_constructor_cannot_mutate_reviewed_source_document(tmp_path,monkeypatch):
    document={"model_type": "step5_vl","text_config": {"model_type": "step5",
        "num_hidden_layers": 92,"num_nextn_predict_layers": 3,"norm_dtype": "float32",
        "fp32_residual_connection": True,"rope_scaling": {"factor": 16.0,"nested": {"values": [1,2]}}}}
    (tmp_path/"config.json").write_text(json.dumps(document))
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    class MutatingConfig:
        def __init__(self,**config):
            config["rope_scaling"]["rope_theta"]=[10000.0]*95
            config["rope_scaling"]["nested"]["values"].append(3)
    release=SimpleNamespace(Step5Config=MutatingConfig)
    reference=SimpleNamespace(verify_source=lambda _:{"revision": core.REVISION},
        load_release=lambda _:(release,None),decoder_config=lambda value:dict(value["text_config"]),
        validate_config=lambda _:None)
    monkeypatch.setattr(core,"_load_reference",lambda _:reference)
    import sys
    monkeypatch.setitem(sys.modules,"shape_invariant",SimpleNamespace(install=lambda:None))
    source=core.Step5BF16Source(tmp_path,reference_root=tmp_path)
    assert source.document==document
    assert source.config==document["text_config"]
    assert json.loads((tmp_path/"config.json").read_text())==document


def test_functional_indexer_weight_capture_observes_native_operand_without_extra_gemm():
    b=block()
    b.self_attn.sparse_indexer_q=nn.Linear(8,4,bias=False,dtype=torch.bfloat16)
    b.self_attn.sparse_indexer_w=nn.Linear(8,2,bias=False,dtype=torch.float32)
    class Indexer:
        def __init__(self,owner):
            self.owner=owner
            self.calls=0
        def project(self,hidden,positions):
            self.calls+=1
            query=self.owner.sparse_indexer_q(hidden)
            # Deliberately bypass the W module, exactly as the pinned release.
            weight=torch.nn.functional.linear(hidden,self.owner.sparse_indexer_w.weight.to(hidden.dtype)).float()
            return query,weight
    b.self_attn.indexer=Indexer(b.self_attn)
    indexer=b.self_attn.indexer
    original=indexer.project
    x=torch.randn(5,8).bfloat16()
    positions=torch.arange(5)
    expected=indexer.project(x,positions)
    before=indexer.calls
    observations=[]
    names=["self_attn.sparse_indexer_q","self_attn.sparse_indexer_w"]
    with core.capture_inputs(b,names,lambda name,value:observations.append((name,value))):
        actual=indexer.project(x,positions)
    assert indexer.calls==before+1
    assert all(torch.equal(a,e) for a,e in zip(actual,expected))
    assert sorted(name for name,_ in observations)==["model.layers.4."+name for name in names]
    assert all(value is x for _,value in observations)
    assert indexer.project==original
    assert not b.self_attn.sparse_indexer_q._forward_pre_hooks
    assert not b.self_attn.sparse_indexer_w._forward_pre_hooks


def test_functional_indexer_observer_restores_method_on_callback_failure():
    b=block()
    class Indexer:
        def project(self,hidden,positions):
            raise AssertionError("callback must abort before the native project")
    b.self_attn.indexer=Indexer()
    original=b.self_attn.indexer.project
    def abort(*_):raise RuntimeError("capture failed")
    with pytest.raises(RuntimeError,match="capture failed"):
        with core.capture_inputs(b,["self_attn.sparse_indexer_w"],abort):
            b.self_attn.indexer.project(torch.ones(2,8),torch.arange(2))
    assert b.self_attn.indexer.project==original


def test_hessian_products_bound_k_and_preserve_all_actual_rows_without_padding(monkeypatch):
    generator=torch.Generator().manual_seed(51)
    inputs=torch.randn(301,8,generator=generator).bfloat16()
    H=torch.zeros(8,8,dtype=torch.float32)
    actual=[]
    addmm=torch.Tensor.addmm_
    def observed(self,left,right,*args,**kwargs):
        if self is H:
            actual.append(right.clone())
            assert right.shape[0]<=128
            assert left.dtype==right.dtype==self.dtype==torch.float32
            assert torch.equal(left,right.T)
        return addmm(self,left,right,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'addmm_',observed)
    assert core.accumulate_hessian_fp32(H,inputs)==301
    assert [len(value) for value in actual]==[128,128,45]
    assert torch.equal(torch.cat(actual),inputs.float())
    expected=(inputs.double().T@inputs.double()).float()
    torch.testing.assert_close(H,expected,rtol=2e-6,atol=1e-5)
    assert float(torch.linalg.norm(H-expected)/torch.linalg.norm(expected))<2e-6
    assert H.dtype==torch.float32


def test_dense_hessian_add_and_update_keep_exact_actual_count_no_identity_rows(monkeypatch):
    inputs=torch.linspace(-2,2,317*8).reshape(317,8).bfloat16()
    capture=core.DenseRawHessian(8,device='cpu')
    lengths=[]
    addmm=torch.Tensor.addmm_
    def observed(self,left,right,*args,**kwargs):
        if self is capture.H:lengths.append(len(right))
        return addmm(self,left,right,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'addmm_',observed)
    capture.add(inputs[:200].reshape(2,100,8))
    capture.update(inputs[200:].T.contiguous().T)
    raw=capture.raw_dict(minimum_rows=317)
    assert capture.count==raw['count']==317 and raw['finalized'] is False
    assert raw.get('identity_rows',0)==0
    assert lengths==[128,72,117]
    assert raw['H'].dtype==torch.float32
    torch.testing.assert_close(raw['H'],(inputs.double().T@inputs.double()).float(),rtol=1e-6,atol=1e-4)


def test_bounded_hessian_accumulation_is_deterministic_and_honors_smaller_chunks():
    inputs=torch.randn(519,7,generator=torch.Generator().manual_seed(9))
    first=torch.zeros(7,7)
    second=torch.zeros(7,7)
    core.accumulate_hessian_fp32(first,inputs,max_rows=64)
    core.accumulate_hessian_fp32(second,inputs,max_rows=64)
    assert torch.equal(first,second)
    capture=core.DenseRawHessian(7,device='cpu',chunk_rows=64)
    capture.add(inputs)
    assert torch.equal(capture.H,first)


@pytest.mark.parametrize('max_rows',[0,-1,129,2048,True,1.5])
def test_hessian_rejects_unsafe_or_noninteger_product_bounds(max_rows):
    H=torch.zeros(4,4)
    with pytest.raises(ValueError):core.accumulate_hessian_fp32(H,torch.ones(2,4),max_rows=max_rows)
    assert torch.count_nonzero(H)==0


@pytest.mark.parametrize('inputs',[torch.empty(0,4),torch.full((2,4),float('nan')),
                                  torch.ones(2,3),torch.ones(2,4,dtype=torch.int64)])
def test_invalid_hessian_observations_do_not_increment_dense_count(inputs):
    capture=core.DenseRawHessian(4,device='cpu')
    with pytest.raises(ValueError):capture.add(inputs)
    assert capture.count==0 and torch.count_nonzero(capture.H)==0


def test_hessian_requires_fp32_storage():
    with pytest.raises(ValueError):
        core.accumulate_hessian_fp32(torch.zeros(4,4,dtype=torch.float64),torch.ones(2,4))
