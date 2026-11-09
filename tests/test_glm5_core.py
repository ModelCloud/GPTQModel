# SPDX-License-Identifier: Apache-2.0
"""CPU ownership, native replay and independent accumulated-matrix oracles."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.utils import glm5_core as core


def packed(outputs=576, inputs=128, bits=6):
    stored=(outputs+127)//128*128
    scale=torch.ones(stored,dtype=torch.float16)
    scale[outputs:]=0
    return {"trellis": torch.zeros(inputs//16,stored//16,bits*16,dtype=torch.int16),
                "suh": torch.ones(inputs,dtype=torch.float16),"svh": scale,
                "mcg": torch.tensor(0xCBAC1FED,dtype=torch.uint32).view(torch.int32)}


def metrics(count):
    return {"quantizer_path": "hessian_ldlq","hessian_metric_status": "ok",
        "hessian_numerical_contract": core._HESSIAN_CONTRACT,
        "hessian_transform_compute_dtype": "torch.float64","hessian_storage_dtype": "torch.float32",
        "hessian_sample_count": count,"reported_metric_value": .001,"hessian_weighted_error_numerator": 1.,
        "hessian_weighted_reference_denominator": 1000.,"hessian_weighted_relative_error": .001}


@pytest.mark.parametrize("bits",[4,6])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_quantizer_owns_raw_inputs_and_preserves_standard_padded_storage(monkeypatch,bits,dtype):
    w=(torch.arange(576*128).reshape(576,128)/8192).to(dtype)
    original=w.clone()
    raw={"H": torch.eye(128),"count": 1024,"finalized": False,"identity_rows": 7}
    called=[]
    def quantize(weight,h,args,*,return_weight_q):
        assert args=={"K": bits,"seed": 787,"sigma_reg": .025,"devices": [torch.device("cpu")],"mcg": True,"apply_out_scales": None}
        assert not return_weight_q and weight.shape==(128,640)
        torch.testing.assert_close(weight[:,:576],original.T.float(),rtol=0,atol=0)
        assert torch.count_nonzero(weight[:,576:])==0
        assert h["H"].data_ptr()!=raw["H"].data_ptr()
        called.append(True)
        h["H"].zero_()
        h["finalized"]=True
        weight.zero_()
        args["error_metrics"]=metrics(h["count"])
        return None,.001,packed(bits=bits)
    monkeypatch.setattr(core,"_quantizer",lambda:SimpleNamespace(quantize_exl3=quantize))
    actual,_,evidence=core.quantize_projection(w,raw,bits=bits,device="cpu")
    assert called and evidence["bits"]==bits and evidence["identity_rows"]==7
    assert evidence["output_padding"]==64 and evidence["hessian_rows"]==1024
    assert all(x.device.type=="cpu" for x in actual.values())
    assert torch.equal(w,original) and torch.equal(raw["H"],torch.eye(128))
    assert raw["finalized"] is False


@pytest.mark.parametrize("key,value",[("quantizer_path","fallback"),("hessian_metric_status","oom"),
    ("hessian_numerical_contract","old"),("hessian_transform_compute_dtype","torch.float32"),
    ("hessian_storage_dtype","torch.float64"),("hessian_sample_count",1),
    ("reported_metric_value",float("nan")),("hessian_weighted_reference_denominator",0.),
    ("hessian_weighted_relative_error",-1.)])
def test_quantizer_rejects_fallback_or_invalid_metric(monkeypatch,key,value):
    def quantize(weight,h,args,**kwargs):
        args["error_metrics"]=metrics(h["count"])
        args["error_metrics"][key]=value
        return None,0.,packed(bits=4)
    monkeypatch.setattr(core,"_quantizer",lambda:SimpleNamespace(quantize_exl3=quantize))
    with pytest.raises(ValueError):
        core.quantize_projection(torch.ones(576,128),{"H": torch.eye(128),"count": 3,"finalized": False},bits=4,device="cpu")


@pytest.mark.parametrize("options",[{"bits": 2},{"bits": True},{"bits": 4.},{"seed": False},
    {"sigma": 0},{"sigma": float("inf")}])
def test_quantizer_rejects_unreviewed_parameters(options):
    with pytest.raises(ValueError):
        core.quantize_projection(torch.ones(576,128),{"H": torch.eye(128),"count": 4,"finalized": False},device="cpu",**options)


def test_raw_hessian_matches_fp64_and_existing_normalized_2i_floor():
    gen=torch.Generator().manual_seed(787)
    x=torch.randn(71,128,generator=gen).bfloat16()
    capture=core.DenseRawHessian(128,device="cpu",chunk_rows=13)
    capture.add(x[:37])
    capture.add(x[37:])
    raw=capture.raw_dict()
    expected=x.double().T@x.double()
    relative=(raw["H"].double()-expected).norm()/expected.norm()
    assert relative < 1e-6 and raw["count"]==71 and raw["finalized"] is False
    # Existing GPTQ captures raw partials, adds missing*I, materializes2/count,
    # then EXL3 prepare restores rawXTX with count/2. Derive that convention
    # independently rather than calling the implementation under test.
    missing=1024-len(x)
    old_normalized=(2./1024)*(expected+missing*torch.eye(128,dtype=torch.float64))
    restored=old_normalized*(1024/2.)
    fresh=raw["H"].double()+missing*torch.eye(128,dtype=torch.float64)
    assert (fresh-restored).norm()/restored.norm() < 1e-6


def source_files(tmp_path,tensors,config):
    save_file(tensors,tmp_path/"model.safetensors")
    (tmp_path/"config.json").write_text(json.dumps(config))
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")}))


def toy_source(tmp_path,layer=3):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
    config=GlmMoeDsaConfig(hidden_size=128,intermediate_size=256,moe_intermediate_size=128,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=128,kv_lora_rank=128,
        qk_nope_head_dim=64,qk_rope_head_dim=64,v_head_dim=128,n_routed_experts=16,num_experts_per_tok=8,
        n_group=1,topk_group=1,first_k_dense_replace=3,index_n_heads=2,index_head_dim=128,index_topk=8,
        indexer_types=["full"]*4,rope_parameters={"rope_theta": 8000000.,"rope_type": "default"})
    config._attn_implementation="sdpa"
    torch.manual_seed(787)
    native=GlmMoeDsaDecoderLayer(config,layer).to(torch.bfloat16).eval()
    for parameter in native.parameters():
        parameter.data.normal_(0,.1)
    if layer>=3:
        native.mlp.gate.e_score_correction_bias=native.mlp.gate.e_score_correction_bias.float()
    native.self_attn.indexer.weights_proj=native.self_attn.indexer.weights_proj.float()
    tensors={}
    prefix=f"model.layers.{layer}."
    for name,value in native.state_dict().items():
        if name=="mlp.experts.gate_up_proj":
            for i in range(16):
                for part,t in zip(("gate_proj","up_proj"),value[i].chunk(2,0)):
                    tensors[f"{prefix}mlp.experts.{i}.{part}.weight"]=t.contiguous()
        elif name=="mlp.experts.down_proj":
            for i in range(16):tensors[f"{prefix}mlp.experts.{i}.down_proj.weight"]=value[i].contiguous()
        else:
            tensors[prefix+name]=value.bfloat16() if name=="self_attn.indexer.weights_proj.weight" else value
    source_files(tmp_path,tensors,config.to_dict())
    return core.GlmBF16Source(tmp_path),native


@pytest.mark.parametrize("layer",[1,3])
def test_source_materializes_native_weights_and_defers_experts(tmp_path,layer,monkeypatch):
    source,native=toy_source(tmp_path,layer)
    calls=[]
    original=source.tensor
    def tensor(name,device="cpu"):
        calls.append(name)
        return original(name,device)
    monkeypatch.setattr(source,"tensor",tensor)
    actual=source.load_block(layer,"cpu")
    assert actual.self_attn.indexer.weights_proj.weight.dtype==torch.float32
    assert source._reader is None
    if layer==3:
        assert all(p.is_meta for p in actual.mlp.experts.parameters())
        assert not any(".experts." in n for n in calls)
        source.materialize_routed_experts(actual,device="cpu")
        assert actual.mlp.gate.e_score_correction_bias.dtype==torch.float32
        count=len(calls)
        source.materialize_routed_experts(actual,device="cpu")
        assert len(calls)==count
    for name,value in native.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name],value,rtol=0,atol=0)
    # Materializing experts never reloads changed core predecessors.
    actual.self_attn.q_a_proj.weight.data.zero_()
    source.materialize_block(actual,layer,"cpu",include_experts=layer==3)
    assert torch.count_nonzero(actual.self_attn.q_a_proj.weight)==0


def test_source_reader_is_single_shard_and_tensors_are_owned(tmp_path,monkeypatch):
    import safetensors
    tensors={"one.weight":torch.arange(8,dtype=torch.bfloat16).reshape(2,4)}
    source_files(tmp_path,tensors,{"model_type": "glm_moe_dsa"})
    save_file({"two.weight":torch.ones(2,4,dtype=torch.bfloat16)},tmp_path/"second.safetensors")
    index=json.loads((tmp_path/"model.safetensors.index.json").read_text())
    index["weight_map"]["two.weight"]="second.safetensors"
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps(index))
    original=safetensors.safe_open
    active=set()
    opened=[]
    class Reader:
        def __init__(self,path,**kwargs):
            self.reader=original(path,**kwargs)
            self.path=path
        def __enter__(self):
            assert not active
            active.add(self.path)
            opened.append(self.path)
            self.reader.__enter__()
            return self
        def __exit__(self,*args):
            active.remove(self.path)
            return self.reader.__exit__(*args)
        def get_tensor(self,name):return self.reader.get_tensor(name)
        def get_slice(self,name):return self.reader.get_slice(name)
    monkeypatch.setattr(safetensors,"safe_open",Reader)
    source=core.GlmBF16Source(tmp_path)
    owned=source.tensor("one.weight")
    owned.zero_()
    assert source.tensor("one.weight")[1,1]==5
    assert source.tensor_header("one.weight")==("BF16",[2,4])
    assert len(opened)==1
    source.tensor("two.weight")
    assert len(opened)==2 and len(active)==1
    source.close()
    assert not active


def test_failed_expert_materialization_restarts_without_trusting_partial_storage(tmp_path,monkeypatch):
    source,native=toy_source(tmp_path,3)
    block=source.load_block(3,"cpu")
    original=source.tensor
    calls=[]
    def fail(name,device="cpu"):
        calls.append(name)
        if len(calls)==7:
            raise OSError("injected source interruption")
        return original(name,device)
    monkeypatch.setattr(source,"tensor",fail)
    with pytest.raises(OSError,match="injected"):
        source.materialize_routed_experts(block,device="cpu")
    assert not block._gptqmodel_glm5_experts_complete and source._reader is None
    monkeypatch.setattr(source,"tensor",original)
    source.materialize_routed_experts(block,device="cpu")
    assert block._gptqmodel_glm5_experts_complete
    assert torch.equal(block.mlp.experts.gate_up_proj,native.mlp.experts.gate_up_proj)
    assert torch.equal(block.mlp.experts.down_proj,native.mlp.experts.down_proj)


def test_native_attention_and_moe_stages_match_reference_block(tmp_path):
    source,native=toy_source(tmp_path,3)
    block=source.load_block(3,"cpu",include_experts=True)
    hidden=torch.randn(1,11,128).bfloat16()
    shifted=torch.arange(1,12).unsqueeze(0)
    context=core.prepare_context(hidden,source.rotary("cpu"),position_ids=shifted)
    assert torch.equal(context["position_ids"],shifted)
    with torch.inference_mode():
        expected,topk=native(hidden,**context)
        entry=core.attention_inputs(block,hidden)
        attention,actual_topk=core.attention_forward(block,entry,context)
        residual,normalized=core.attention_finish(block,hidden,attention)
        actual=core.complete_moe(residual,normalized,block.mlp)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(actual_topk,topk)
        independent_down=torch.nn.functional.silu(torch.nn.functional.linear(normalized,block.mlp.shared_experts.gate_proj.weight))*torch.nn.functional.linear(normalized,block.mlp.shared_experts.up_proj.weight)
        assert torch.equal(core.shared_down_input(block.mlp.shared_experts,normalized),independent_down)


def test_router_adjacent_candidates_preserve_exact_selected_tie_boundary(tmp_path):
    source,_=toy_source(tmp_path,3)
    block=source.load_block(3,"cpu")
    x=torch.randn(1,7,128).bfloat16()
    for tied in (False,True):
        if tied:
            block.mlp.gate.weight.data.zero_()
            block.mlp.gate.e_score_correction_bias.zero_()
        selected,weights,candidates=core.router_choices(block.mlp,x)
        logits,expected_weights,expected_selected=block.mlp.gate(x)
        score=torch.sigmoid(logits.float())+block.mlp.gate.e_score_correction_bias
        score.scatter_(1,expected_selected,float("-inf"))
        expected=score.topk(8,dim=-1,sorted=True).indices
        assert torch.equal(selected,expected_selected) and torch.equal(weights,expected_weights)
        assert torch.equal(candidates,expected)
        assert not (selected[:,:,None]==candidates[:,None,:]).any()


@pytest.mark.parametrize("bad",[{"position_ids": torch.zeros(1,2,dtype=torch.int32)},
    {"position_ids": torch.tensor([[-1,0]])},{"attention_mask": torch.ones(2,2)},
    {"prev_topk": torch.ones(1,2,2)}])
def test_context_rejects_malformed_positions_masks_or_shared_state(bad):
    with pytest.raises(ValueError):
        core.prepare_context(torch.ones(1,2,128),lambda x,p:(x,x),**bad)
