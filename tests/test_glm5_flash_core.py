# SPDX-License-Identifier: Apache-2.0
"""Independent native decoder and checkpoint-ownership contracts for Flash."""
import inspect
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.utils import glm5_flash_core as core
from gptqmodel.utils import glm5_flash_mtp as mtp


def config():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    cfg = Glm5NextTextConfig(hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=2, num_nextn_predict_layers=1, num_attention_heads=2, num_key_value_heads=2,
        q_lora_rank=128, kv_lora_rank=128, qk_rope_head_dim=0, qk_nope_head_dim=128,
        v_head_dim=128, index_n_heads=2, index_head_dim=128, index_topk=4, index_kpool=4,
        linear_num_heads=2, linear_head_dim=128, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "deepseek_sparse_attention"], mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"], n_routed_experts=16, num_experts_per_tok=8,
        hc_mult=4, hc_sinkhorn_iters=3, vocab_size=256, pad_token_id=0)
    cfg._attn_implementation="sdpa"
    cfg._experts_implementation="eager"
    return cfg


def initialize(block):
    generator=torch.Generator().manual_seed(787)
    with torch.no_grad():
        for key, parameter in (*block.named_parameters(), *block.named_buffers()):
            values=torch.randn(parameter.shape,generator=generator)*.02
            if "norm" in key and key.endswith("weight"):
                values.fill_(1.)
            if key.endswith("scale"):
                values.fill_(.1)
            if "conv1d" in key:
                values=values.bfloat16().float()
            parameter.data=values.to(torch.float32 if any(s in key for s in
                ("conv1d","A_log","dt_bias","e_score_correction_bias","hc.base","hc.scale")) else torch.bfloat16)
    return block.eval()


@pytest.mark.parametrize("layer",[0,1])
def test_staged_mhc_replay_is_byte_exact_to_native_decoder(layer,monkeypatch):
    from transformers.models.glm5_next import modeling_glm5_next as native
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    function=getattr(native.chunk_kimi_delta_attention,"_gptqmodel_kda_original",native.chunk_kimi_delta_attention)
    monkeypatch.setattr(native,"chunk_kimi_delta_attention",core.bounded_kda(inspect.unwrap(function)))
    core.install_bounded_kda()
    block=initialize(Glm5NextTextDecoderLayer(config(),layer))
    hidden=torch.randn(1,9,4,128,generator=torch.Generator().manual_seed(1)).bfloat16()
    state={"hidden": hidden}
    with torch.inference_mode():
        expected=core.native_forward(block,state)
        actual=core.staged_forward(block,state)
    assert set(actual)==set(expected)
    assert all(torch.equal(actual[k],expected[k]) for k in actual)


def test_kda_head_slicing_matches_whole_native_recurrence():
    from transformers.models.glm5_next import modeling_glm5_next as native
    function=native.chunk_kimi_delta_attention
    # Keep a direct callable independent of any previously installed wrapper.
    function=inspect.unwrap(getattr(function,"_gptqmodel_kda_original",function))
    generator=torch.Generator().manual_seed(32)
    q=torch.randn(1,7,4,8,generator=generator).bfloat16()
    k=torch.randn(q.shape,generator=generator).bfloat16()
    v=torch.randn(q.shape,generator=generator).bfloat16()
    g=-torch.rand(q.shape,generator=generator)
    beta=torch.rand(1,7,4,generator=generator).bfloat16()
    expected,_=function(q,k,v,g,beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
    actual,_=core.bounded_kda(function,heads=2)(q,k,v,g,beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
    assert torch.equal(actual,expected)


@pytest.mark.parametrize("layer",[0,1])
def test_source_remaps_native_names_and_owns_checkpoint_buffers(tmp_path,layer):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    cfg=config()
    original=initialize(Glm5NextTextDecoderLayer(cfg,layer))
    prefix=f"{core.PREFIX}layers.{layer}."
    tensors={}
    for key,value in original.state_dict().items():
        if key.startswith("mlp.experts."):
            continue
        names=core.checkpoint_names(key)
        chunks=value.chunk(3,dim=0) if len(names)==3 else [value]
        for name,chunk in zip(names,chunks,strict=True):
            tensors[prefix+name]=chunk.to(torch.float32 if any(x in name for x in
                ("_base","_scale","A_log","dt_bias","e_score_correction_bias")) else torch.bfloat16).contiguous()
    if hasattr(original.mlp,"experts"):
        for expert in range(cfg.n_routed_experts):
            gate,up=original.mlp.experts.gate_up_proj[expert].chunk(2,dim=0)
            for part,value in (("gate_proj",gate),("up_proj",up),("down_proj",original.mlp.experts.down_proj[expert])):
                tensors[prefix+f"mlp.experts.{expert}.{part}.weight"]=value.contiguous()
    save_file(tensors,tmp_path/"weights.safetensors")
    (tmp_path/"config.json").write_text(json.dumps({"model_type": "glm5_next","text_config": cfg.to_dict()}))
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")}))
    source=core.GlmFlashBF16Source(tmp_path)
    block=source.load_block(layer,"cpu",include_experts=True)
    assert source._reader is None
    for key,value in original.state_dict().items():
        assert torch.equal(value,block.state_dict()[key]),key
    name=next(iter(tensors))
    tensor=source.tensor(name)
    tensor.zero_()
    assert not torch.equal(tensor,source.tensor(name))
    source.close()


def test_shared_down_hessian_input_uses_native_swiglu_clamp():
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP
    mlp=initialize(Glm5NextTextMLP(config()))
    hidden=(torch.randn(1,5,128)*100).bfloat16()
    with torch.inference_mode():
        capture=core.capture_linear_input(mlp.down_proj,mlp,hidden)
        assert torch.equal(core.shared_down_input(mlp,hidden),capture)


def test_flash_mtp_pairs_stream_mean_and_shifted_positions_without_zeroing_first_embedding():
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
    norm=Glm5NextTextRMSNorm(8).bfloat16()
    frontier=torch.randn(1,5,4,8).bfloat16()
    ids=torch.tensor([[1,2,3,4,5]])
    embedding=torch.randn(16,8).bfloat16()
    aligned=mtp.aligned_inputs(frontier,ids,embedding,norm)
    assert torch.equal(aligned["target_hidden"],norm(frontier.mean(dim=2))[:,:-1])
    assert torch.equal(aligned["inputs_embeds"],embedding[ids[:,1:]])
    assert torch.equal(aligned["position_ids"],torch.tensor([[1,2,3,4]]))
    assert aligned["attention_mask"].shape==(1,4)


def test_mtp_rejects_misaligned_input():
    with pytest.raises(ValueError):
        mtp.aligned_inputs(torch.ones(1,4,4,8),torch.ones(1,3,dtype=torch.long),torch.ones(2,8),nn.Identity())
