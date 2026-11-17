# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real CPU acceptance for the package-owned MiMo-V2 family."""

import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from gptqmodel import GPTQModel


def tiny_config(**overrides):
    values = {
        "model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"], "vocab_size": 64,
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "v_head_dim": 4,
        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 1, "swa_head_dim": 8, "swa_v_head_dim": 4,
        "partial_rotary_factor": 0.5, "rope_theta": 10000.0, "swa_rope_theta": 500.0,
        "max_position_embeddings": 64, "sliding_window": 4, "hybrid_layer_pattern": [0, 1],
        "attention_projection_layout": "fused_qkv", "n_routed_experts": 4,
        "moe_intermediate_size": 32, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
        "moe_layer_freq": [0, 1], "routed_scaling_factor": 1.3,
        "add_swa_attention_sink_bias": True, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
    }
    values.update(overrides)
    return values


def make_model(dtype=torch.float32, implementation="eager", **overrides):
    from gptqmodel.models.ports.mimo_v2.configuration_mimo_v2 import MiMoV2Config
    from gptqmodel.models.ports.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM

    config = MiMoV2Config(**tiny_config(**overrides))
    torch.manual_seed(73)
    model = MiMoV2ForCausalLM._from_config(config, dtype=dtype, attn_implementation=implementation).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("e_score_correction_bias"):
                parameter.copy_(torch.linspace(-0.13, 0.19, parameter.numel()))
            elif name.endswith("attention_sink_bias"):
                parameter.copy_(torch.linspace(-0.6, 0.8, parameter.numel()))
    return model


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def write_tokenizer(path):
    tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                                            **{str(i): i for i in range(4, 64)}}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>",
                           eos_token="<eos>", unk_token="<unk>").save_pretrained(path)


def hostile_maps(path):
    marker = path / "EXECUTED"
    for name in ("configuration_evil", "modeling_evil", "tokenization_evil", "processing_evil"):
        (path / (name + ".py")).write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n")
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "configuration_evil.Evil", "AutoModelForCausalLM": "modeling_evil.Evil"}
    (path / "config.json").write_text(json.dumps(payload))
    tokenizer = json.loads((path / "tokenizer_config.json").read_text())
    tokenizer["auto_map"] = {"AutoTokenizer": ["tokenization_evil.Evil", "tokenization_evil.EvilFast"]}
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    for asset in ("processor_config.json", "preprocessor_config.json"):
        asset_path = path / asset
        value = json.loads(asset_path.read_text()) if asset_path.exists() else {}
        value["auto_map"] = {"AutoProcessor": "processing_evil.Evil", "AutoImageProcessor": "processing_evil.Evil"}
        asset_path.write_text(json.dumps(value))
    return marker


def media_config():
    return {
        "vision_config": {"depth": 2, "fullatt_block_indexes": [1], "hidden_act": "silu", "hidden_size": 32,
                           "in_chans": 3, "intermediate_size": 64, "num_heads": 4, "num_key_value_heads": 2,
                           "qk_channels": 8, "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                           "spatial_merge_size": 2, "use_sink": True, "visual_token_window_size": 3,
                           "vit_window_attn_types": [1, 0]},
        "audio_config": {"audio_channels": 2, "group_size": 2, "input_full_attention": True,
                          "input_local_dim": 16, "input_local_layers": 1, "input_local_attn_heads": 4,
                          "input_local_intermediate_size": 32, "input_local_hidden_dropout": 0.0,
                          "out_hidden_size": 32, "add_post_norm": True, "projection_layers": 2,
                          "speech_vocab_size": "16", "speech_zeroemb_idx": "0", "rope_theta": 640000.0},
        "image_token_id": 59, "video_token_id": 60,
        "processor_config": {"audio_token_id": 56},
    }


def write_media_checkpoint(path, model):
    from transformers import Qwen2VLImageProcessor

    model.save_pretrained(path)
    specials = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<|audio|>": 56,
                "<|vision_start|>": 57, "<|vision_end|>": 58, "<|image_pad|>": 59, "<|video_pad|>": 60}
    tokenizer = Tokenizer(models.WordLevel({**specials, **{str(i): i for i in range(4, 64)
                                                                    if i not in specials.values()}}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(list(specials))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
                           unk_token="<unk>", additional_special_tokens=list(specials)[4:]).save_pretrained(path)
    Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                         min_pixels=64, max_pixels=64).save_pretrained(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_image_video_audio_encoder_and_decoder_roundtrip(tmp_path, dtype):
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    from transformers import Qwen2Model

    from gptqmodel.models.ports.mimo_v2.processing_mimo_v2 import MiMoV2Processor

    model = make_model(dtype=dtype, **media_config())
    write_media_checkpoint(tmp_path, model)
    processor = MiMoV2Processor.from_pretrained(tmp_path, config=model.config)
    image = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    video = [image, Image.fromarray(255 - np.asarray(image))]
    batch = processor(text=["1 <|image_pad|> <|video_pad|> <|audio|> <|audio|> 5"], images=[image],
                      videos=[video], return_tensors="pt", videos_kwargs={"do_sample_frames": False})
    batch["audio_codes"] = torch.tensor([[1, 2], [3, 4], [5, 6]])
    for name in ("pixel_values", "video_pixel_values"):
        batch[name] = batch[name].to(dtype)
    with torch.inference_mode():
        visual = model.visual(batch["pixel_values"], batch["image_grid_thw"])
        video_features = model.visual(batch["video_pixel_values"], batch["video_grid_thw"])
        # Independent grouped-code pad/sum and native Qwen2 local-transformer oracle.
        codes = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [5, 6]]])
        local_embeds = sum(model.speech_embeddings[i](codes[:, :, i]) for i in range(2))
        local = model.audio_encoder.input_local_transformer
        native = Qwen2Model._from_config(local.config, dtype=dtype).eval()
        native.load_state_dict(local.state_dict())
        local_hidden = native(inputs_embeds=local_embeds, use_cache=False, is_causal=False).last_hidden_state
        audio_expected = F.linear(F.gelu(F.linear(local_hidden.flatten(1), model.audio_encoder.projection.mlp[0].weight)),
                                  model.audio_encoder.projection.mlp[2].weight)
        torch.testing.assert_close(model.audio_encoder(model.speech_embeddings, audio_codes=batch["audio_codes"]),
                                   audio_expected, atol=0, rtol=0)
        embeds = model.get_input_embeddings()(batch["input_ids"]).clone()
        for token, features in ((59, visual), (60, video_features), (56, audio_expected)):
            embeds[batch["input_ids"] == token] = features
        expected = model(inputs_embeds=embeds, attention_mask=batch["attention_mask"], use_cache=False).logits
        actual = model(**batch, use_cache=True)
        torch.testing.assert_close(actual.logits, expected, atol=0, rtol=0)
        changed = dict(batch, pixel_values=batch["pixel_values"] * -2 + 0.1)
        assert not torch.allclose(model(**changed, use_cache=False).logits, expected)
        next_id = expected[:, -1:].argmax(-1)
        resumed = model(next_id, past_key_values=actual.past_key_values,
                        attention_mask=torch.cat([batch["attention_mask"], torch.ones_like(next_id)], -1))
        full = model(**dict(batch, input_ids=torch.cat([batch["input_ids"], next_id], -1),
                            attention_mask=torch.cat([batch["attention_mask"], torch.ones_like(next_id)], -1)))
        tol = 0.016 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 2e-6
        torch.testing.assert_close(resumed.logits, full.logits[:, -1:], atol=tol, rtol=tol)
        generated = model.generate(**batch, max_new_tokens=2, eos_token_id=None, do_sample=False)
        assert torch.equal(generated[:, batch["input_ids"].shape[1]:][:, :1], next_id)
        assert generated.shape[1] == batch["input_ids"].shape[1] + 2
    from gptqmodel.models.ports.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM
    loaded = MiMoV2ForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor, atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**batch, use_cache=False).logits, expected, atol=0, rtol=0)
    processor.save_pretrained(tmp_path / "processor")
    p2 = MiMoV2Processor.from_pretrained(tmp_path / "processor", config=model.config)
    again = p2(text=["1 <|image_pad|> 5"], images=[image], return_tensors="pt")
    first = processor(text=["1 <|image_pad|> 5"], images=[image], return_tensors="pt")
    for key in first:
        torch.testing.assert_close(again[key], first[key], atol=0, rtol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_owned_media_processor_ignores_hostile_assets(tmp_path, monkeypatch, trust, bypass):
    model = make_model(**media_config())
    write_media_checkpoint(tmp_path, model)
    marker = hostile_maps(tmp_path)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                            attn_implementation="eager")
    assert type(wrapper.processor).__module__ == "gptqmodel.models.ports.mimo_v2.processing_mimo_v2"
    assert not marker.exists()
    ids = torch.tensor([[1, 56, 56, 4]])
    with torch.inference_mode():
        expected = model(ids, audio_codes=torch.tensor([[1, 2], [3, 4], [5, 6]])).logits
        actual = wrapper.model(ids, audio_codes=torch.tensor([[1, 2], [3, 4], [5, 6]])).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("kv", [1, 2, 4])
def test_vision_attention_against_independent_float64_segment_equations(full, kv):
    import torch.nn.functional as F

    from gptqmodel.models.ports.mimo_v2.modeling_mimo_v2 import MiMoVisionAttention

    torch.manual_seed(41)
    attention = MiMoVisionAttention(dim=16, num_heads=4, num_kv_heads=kv, head_dim=4,
                                    use_sinks=True, window_size=1).eval()
    attention.sinks.data.copy_(torch.tensor([-1.1, -0.3, 0.4, 1.3]))
    x = torch.randn(7, 16)
    angles = torch.arange(28).reshape(7, 4).float() / 7
    cos, sin = angles.cos(), angles.sin()
    q, k, v = F.linear(x.double(), attention.qkv.weight.double(), attention.qkv.bias.double()).split([16, kv*4, kv*4], -1)
    q, k, v = q.reshape(7, 4, 4), k.reshape(7, kv, 4), v.reshape(7, kv, 4)
    def rotate(t):
        return torch.cat([-t[..., 2:], t[..., :2]], -1)
    q = q * cos[:, None].double() + rotate(q) * sin[:, None].double()
    k = k * cos[:, None].double() + rotate(k) * sin[:, None].double()
    pieces = []
    for start, end in ((0, 3), (3, 7)):
        heads = []
        for h in range(4):
            kh = h // (4 // kv)
            scores = q[start:end, h] @ k[start:end, kh].T / 2
            if not full:
                idx = torch.arange(end-start)
                scores = scores.masked_fill((idx[:, None] - idx[None, :]).abs() > 1, -torch.inf)
            scores[:, 0] += attention.sinks[h].double()
            heads.append(scores.softmax(-1) @ v[start:end, kh])
        pieces.append(torch.cat(heads, -1))
    expected = F.linear(torch.cat(pieces), attention.proj.weight.double(), attention.proj.bias.double())
    with torch.inference_mode():
        actual = attention(x, torch.tensor([0, 3, 7], dtype=torch.int32), (cos, sin), full_attn=full)
    torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)


def test_audio_codec_mel_codes_nearest_codebook_and_safetensors(tmp_path):
    from gptqmodel.models.ports.mimo_v2.modeling_mimo_v2 import (
        EuclideanCodebook,
        MiMoAudioTokenizer,
        MiMoAudioTokenizerConfig,
    )

    torch.manual_seed(31)
    cfg = MiMoAudioTokenizerConfig(d_model=32, encoder_layers=1, encoder_attention_heads=4,
                                  encoder_ffn_dim=64, n_mels=8, num_quantizers=2,
                                  codebook_size=[16, 16], avg_pooler=2, max_audio_seconds=2)
    model = MiMoAudioTokenizer(cfg).eval()
    mels = torch.randn(28, 8)
    lengths = torch.tensor([16, 12])
    with torch.inference_mode():
        first = model.encode(mels, lengths)
        model.save_pretrained(tmp_path)
        loaded = MiMoAudioTokenizer.from_pretrained(tmp_path).eval()
        second = loaded.encode(mels, lengths)
        for left, right in zip(first, second):
            if left is not None:
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        book = EuclideanCodebook(dim=4, codebook_size=5)
        vectors = torch.randn(2, 3, 4)
        expected = ((vectors[..., None, :] - book.embed[None, None])**2).sum(-1).argmin(-1)
        assert torch.equal(book.encode(vectors), expected)


def test_media_count_and_processor_failure_repair_retry(tmp_path):
    from gptqmodel.models.ports.mimo_v2.processing_mimo_v2 import MiMoV2Processor

    model = make_model(**media_config())
    write_media_checkpoint(tmp_path, model)
    ids = torch.tensor([[1, 56, 56, 4]])
    with torch.inference_mode():
        with pytest.raises(ValueError, match="count mismatch"):
            model(ids, audio_embeds=torch.randn(1, 32))
        assert torch.isfinite(model(ids, audio_embeds=torch.randn(2, 32)).logits).all()
    filename = tmp_path / "preprocessor_config.json"
    original = filename.read_bytes()
    filename.write_text("{invalid-json")
    with pytest.raises(OSError, match="JSON"):
        MiMoV2Processor.from_pretrained(tmp_path, config=model.config)
    filename.write_bytes(original)
    processor = MiMoV2Processor.from_pretrained(tmp_path, config=model.config)
    assert torch.equal(processor(text=["<bos> 4"], return_tensors="pt")["input_ids"], torch.tensor([[1, 4]]))


@pytest.mark.parametrize("first_sliding", [False, True])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_owned_layer_replay_after_rotary_offload(first_sliding, implementation):
    from torch import nn

    from gptqmodel.models.definitions.mimo_v2 import MimoV2QModel

    model = make_model(implementation=implementation, hybrid_layer_pattern=[1, 0] if first_sliding else [0, 1])
    wrapper = MimoV2QModel.__new__(MimoV2QModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    ids = torch.tensor([[1, 5, 6, 7, 8, 9], [1, 11, 12, 13, 14, 15]])
    mask = torch.ones_like(ids)
    mask[0, :2] = 0
    positions = torch.arange(7, 13).expand_as(ids)
    captured, expected = {}, []

    def capture_first(module, args, kwargs):
        captured["hidden"] = args[0]
        captured["kwargs"] = wrapper.capture_first_layer_input_kwargs(args, kwargs, torch.device("cpu"),
                                                                    {"position_ids": kwargs["position_ids"]})

    handles = [model.model.layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)]
    handles.extend(layer.register_forward_hook(lambda module, args, output: expected.append(output))
                   for layer in model.model.layers)
    with torch.inference_mode():
        wrapper.run_input_capture({"input_ids": ids, "attention_mask": mask, "position_ids": positions},
                                  use_cache=False, data_device=torch.device("cpu"))
    for handle in handles:
        handle.remove()
    model.model.rotary_emb.to("meta")
    model.model.swa_rotary_emb.to("meta")
    hidden = captured["hidden"]
    for layer, output in zip(model.model.layers, expected):
        kwargs = wrapper.prepare_layer_replay_kwargs(layer, [hidden], captured["kwargs"], torch.device("cpu"))
        with torch.inference_mode():
            hidden = layer(hidden, **kwargs)
        torch.testing.assert_close(hidden, output, atol=0, rtol=0)


def independent_logits(model, ids, positions, mask):
    """Float64 equations transcribed from the pinned source, without port helpers.

    Compute full/SWA attention per query head and route per token/expert rather
    than reusing the production head-repeat and one-hot/index-add algorithms.
    """
    import torch.nn.functional as F

    c = model.config
    w = {k: v.double() for k, v in model.state_dict().items()}
    h = F.embedding(ids, w["model.embed_tokens.weight"])

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.layernorm_epsilon) * w[name + ".weight"]

    def linear(x, name):
        return F.linear(x, w[name + ".weight"], w.get(name + ".bias"))

    def mlp(x, name):
        return linear(F.silu(linear(x, name + ".gate_proj")) * linear(x, name + ".up_proj"), name + ".down_proj")

    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}"
        swa = bool(c.hybrid_layer_pattern[i])
        prefix = "swa_" if swa else ""
        heads, kv = getattr(c, prefix + "num_attention_heads"), getattr(c, prefix + "num_key_value_heads")
        hd, vd = getattr(c, prefix + "head_dim"), getattr(c, prefix + "v_head_dim")
        x = norm(h, p + ".input_layernorm")
        if c.attention_projection_layout == "fused_qkv":
            q, k, v = linear(x, p + ".self_attn.qkv_proj").split([heads * hd, kv * hd, kv * vd], -1)
        else:
            q, k, v = [linear(x, p + ".self_attn." + n + "_proj") for n in ("q", "k", "v")]
        q, k, v = q.unflatten(-1, (heads, hd)), k.unflatten(-1, (kv, hd)), v.unflatten(-1, (kv, vd))
        if c.attention_value_scale is not None:
            v = v * c.attention_value_scale
        rd = int(hd * c.partial_rotary_factor)
        theta = c.swa_rope_theta if swa else c.rope_theta
        angles = positions.double()[..., None] * (theta ** (-torch.arange(0, rd, 2, dtype=torch.float64) / rd))
        angles = torch.cat([angles, angles], -1).unsqueeze(-2)

        def rope(y):
            part = y[..., :rd]
            rotated = torch.cat([-part[..., rd // 2:], part[..., :rd // 2]], -1)
            return torch.cat([part * angles.cos() + rotated * angles.sin(), y[..., rd:]], -1)

        q, k = rope(q), rope(k)
        outputs = []
        for head in range(heads):
            key_head = head // (heads // kv)
            scores = q[:, :, head] @ k[:, :, key_head].transpose(-1, -2) / hd**0.5
            index = torch.arange(ids.shape[1])
            allowed = index[:, None] >= index[None, :]
            if swa:
                allowed &= index[:, None] - index[None, :] < c.sliding_window
            allowed = allowed[None] & mask[:, None].bool()
            scores = scores.masked_fill(~allowed, torch.finfo(torch.float32).min)
            sink = w.get(p + ".self_attn.attention_sink_bias")
            if sink is not None:
                scores = torch.cat([scores, sink[head].expand(*scores.shape[:-1], 1)], -1)
            probs = scores.softmax(-1)
            if sink is not None:
                probs = probs[..., :-1]
            outputs.append(probs @ v[:, :, key_head])
        h = h + linear(torch.cat(outputs, -1), p + ".self_attn.o_proj")
        x = norm(h, p + ".post_attention_layernorm")
        if c.n_routed_experts and c.moe_layer_freq[i]:
            flat = x.flatten(0, 1)
            logits = linear(flat, p + ".mlp.gate")
            scores = logits.sigmoid()
            choice = scores + w[p + ".mlp.gate.e_score_correction_bias"]
            groups = choice.unflatten(-1, (c.n_group, -1))
            selected_groups = groups.topk(2, -1).values.sum(-1).topk(c.topk_group, -1).indices
            group_mask = torch.zeros_like(groups[..., 0], dtype=torch.bool).scatter_(1, selected_groups, True)
            choice = choice.masked_fill(~group_mask.repeat_interleave(c.n_routed_experts // c.n_group, -1), -torch.inf)
            selected = choice.topk(c.num_experts_per_tok, -1).indices
            weights = scores.gather(1, selected)
            if c.num_experts_per_tok > 1 and c.norm_topk_prob:
                weights /= weights.sum(-1, keepdim=True) + 1e-20
            weights *= c.routed_scaling_factor or 1.0
            rows = []
            for token in range(flat.shape[0]):
                rows.append(sum(mlp(flat[token], p + f".mlp.experts.{int(expert)}") * weights[token, rank]
                                for rank, expert in enumerate(selected[token])))
            h = h + torch.stack(rows).view_as(h)
        else:
            h = h + mlp(x, p + ".mlp")
    return linear(norm(h, "model.norm"), "lm_head")


@pytest.mark.parametrize("layout", ["split", "fused_qkv"])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_independent_decoder_equations_and_causality(layout, kv, bias):
    model = make_model(attention_projection_layout=layout, num_key_value_heads=kv, swa_num_key_value_heads=kv,
                       attention_bias=bias, attention_value_scale=0.707)
    ids = torch.tensor([[1, 8, 5, 6, 9, 12], [1, 4, 11, 7, 14, 8]])
    positions = torch.arange(7, 13).expand_as(ids)
    mask = torch.ones_like(ids)
    mask[0, 0] = 0
    with torch.inference_mode():
        actual = model(ids, position_ids=positions, attention_mask=mask, use_cache=False).logits
        expected = independent_logits(model, ids, positions, mask)
        # The entirely padded first query has implementation-specific masking;
        # every valid query is independently checked.
        torch.testing.assert_close(actual[:, 1:].double(), expected[:, 1:], atol=3e-7, rtol=2e-5)
        changed = ids.clone()
        changed[:, -2:] = torch.tensor([21, 22])
        torch.testing.assert_close(model(changed, position_ids=positions, attention_mask=mask).logits[:, 1:-2],
                                   actual[:, 1:-2], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("layout", ["split", "fused_qkv"])
def test_cache_generation_and_safetensors_roundtrip(tmp_path, dtype, implementation, layout):
    from gptqmodel.models.ports.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM

    model = make_model(dtype=dtype, implementation=implementation, attention_projection_layout=layout)
    ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
    tol = 2e-6 if dtype == torch.float32 else (0.002 if dtype == torch.float16 else 0.016)
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        for step in (1, 4):
            prefix = model(ids[:, :-step], use_cache=True)
            resumed = model(ids[:, -step:], past_key_values=prefix.past_key_values, use_cache=True)
            torch.testing.assert_close(resumed.logits, full[:, -step:], atol=tol, rtol=tol)
        expected = ids[:, :3]
        for _ in range(3):
            expected = torch.cat([expected, model(expected, use_cache=False).logits[:, -1:].argmax(-1)], -1)
        generated = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None)
        assert torch.equal(generated, expected)
        beam = model.generate(ids[:, :3], max_new_tokens=2, num_beams=2, eos_token_id=None)
        assert beam.shape == (1, 5)
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    reloaded = MiMoV2ForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation=implementation).eval()
    assert set(reloaded.state_dict()) == set(model.state_dict())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, reloaded.state_dict()[key], atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, full, atol=0, rtol=0)


@pytest.mark.parametrize("sink", ["full", "none"])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_full_sink_and_native_swa_without_sink_against_oracle(sink, implementation):
    model = make_model(implementation=implementation, add_full_attention_sink_bias=sink == "full",
                       add_swa_attention_sink_bias=False)
    ids = torch.tensor([[1, 4, 6, 8, 10, 12]])
    with torch.inference_mode():
        expected = independent_logits(model, ids, torch.arange(6)[None], torch.ones_like(ids))
        actual = model(ids, use_cache=False).logits
        torch.testing.assert_close(actual.double(), expected, atol=3e-7, rtol=2e-5)
        prefix = model(ids[:, :2], use_cache=True)
        torch.testing.assert_close(model(ids[:, 2:], past_key_values=prefix.past_key_values).logits,
                                   actual[:, 2:], atol=3e-7, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_public_requested_dtype_subfolder_and_exact_assignment(tmp_path, dtype):
    path = tmp_path / "nested"
    model = make_model(dtype=dtype, **media_config())
    write_media_checkpoint(path, model)
    marker = hostile_maps(path)
    loaded = GPTQModel.load(str(tmp_path), subfolder="nested", dtype=dtype, device="cpu", attn_implementation="eager")
    assert not marker.exists()
    assert set(model.state_dict()) == set(loaded.model.state_dict())
    for name, value in model.state_dict().items():
        assert loaded.model.state_dict()[name].dtype == value.dtype
        torch.testing.assert_close(loaded.model.state_dict()[name], value, atol=0, rtol=0)
    ids = torch.tensor([[1, 56, 56, 4]])
    with torch.inference_mode():
        kwargs = {"audio_codes": torch.tensor([[1, 2], [3, 4], [5, 6]])}
        torch.testing.assert_close(loaded.model(ids, **kwargs).logits, model(ids, **kwargs).logits, atol=0, rtol=0)


def test_corrupt_safetensors_public_failure_repair_retry(tmp_path):
    model = make_model()
    model.save_pretrained(tmp_path)
    write_tokenizer(tmp_path)
    marker = hostile_maps(tmp_path)
    weights = tmp_path / "model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original[:12])
    with pytest.raises(Exception, match="header|Header|buffer"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    assert not marker.exists()
    weights.write_bytes(original)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        ids = torch.tensor([[1, 5, 9]])
        torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, atol=0, rtol=0)
    assert not marker.exists()


@pytest.mark.parametrize("layout", ["split", "fused_qkv"])
@pytest.mark.parametrize("moe", [[0, 0], [0, 1], [1, 1]])
def test_public_plain_awq_packs_every_decoder_linear_and_two_roundtrips(tmp_path, layout, moe):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig

    media = moe == [0, 1]
    source = make_model(dtype=torch.float16, attention_projection_layout=layout, moe_layer_freq=moe,
                        swa_num_key_value_heads=2, num_experts_per_tok=4, topk_group=2,
                        **(media_config() if media else {}))
    if media:
        write_media_checkpoint(tmp_path, source)
    else:
        source.save_pretrained(tmp_path)
        write_tokenizer(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), dtype=torch.float16, backend=BACKEND.TORCH,
                            quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None))
    before = {name: value.clone() for name, value in source.state_dict().items()
              if not name.startswith("model.layers.") or any(key in name for key in ("norm", "mlp.gate.", "sink_bias"))}
    expected_names = {name for name, module in source.named_modules()
                      if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear)}
    for layer in wrapper.model.model.layers:
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    if media:
        calibration = [{"input_ids": torch.tensor([[1, 59, 59, 59, 59, 60, 60, 60, 60, 56, 56, 4, 5, 6]]),
                        "attention_mask": torch.ones(1, 14, dtype=torch.long),
                        "pixel_values": torch.linspace(-2, 3, 384, dtype=torch.float16).reshape(16, 24),
                        "image_grid_thw": torch.tensor([[1, 4, 4]]),
                        "video_pixel_values": torch.linspace(2, -1, 384, dtype=torch.float16).reshape(16, 24),
                        "video_grid_thw": torch.tensor([[1, 4, 4]]),
                        "audio_codes": torch.tensor([[1, 2], [3, 4], [5, 6]])} for _ in range(2)]
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    packed = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == expected_names
    for name, value in before.items():
        torch.testing.assert_close(wrapper.model.state_dict()[name], value, atol=0, rtol=0)
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    for index in range(2):
        saved = tmp_path / ("quantized" + str(index))
        wrapper.save_quantized(str(saved))
        assert list(saved.glob("*.safetensors")) and not list(saved.rglob("*.bin"))
        reloaded = GPTQModel.load(str(saved), device="cpu", dtype=torch.float16, backend=BACKEND.TORCH)
        assert {name for name, module in reloaded.model.named_modules() if isinstance(module, AwqTorchLinear)} == packed
        for name, tensor in wrapper.model.state_dict().items():
            torch.testing.assert_close(reloaded.model.state_dict()[name], tensor, atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, expected,
                                       atol=2e-4, rtol=2e-3)
        wrapper = reloaded


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_load_owns_config_model_and_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    # Build an independent one-layer checkpoint: no imports from the new port
    # are needed to demonstrate the original public loading failure.
    config = tiny_config(num_hidden_layers=1, hybrid_layer_pattern=[0], moe_layer_freq=[0],
                         n_routed_experts=None, attention_projection_layout="split")
    (tmp_path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(73)
    shapes = {"model.embed_tokens.weight": (64, 32), "model.norm.weight": (32,), "lm_head.weight": (64, 32)}
    for name, shape in {"input_layernorm": (32,), "post_attention_layernorm": (32,),
                        "self_attn.q_proj": (32, 32), "self_attn.k_proj": (16, 32),
                        "self_attn.v_proj": (8, 32), "self_attn.o_proj": (32, 16),
                        "mlp.gate_proj": (64, 32), "mlp.up_proj": (64, 32), "mlp.down_proj": (32, 64)}.items():
        shapes[f"model.layers.0.{name}.weight"] = shape
    save_file({key: torch.ones(shape) if len(shape) == 1 else torch.randn(shape, generator=generator) * 0.1
               for key, shape in shapes.items()}, tmp_path / "model.safetensors")
    write_tokenizer(tmp_path)
    marker = hostile_maps(tmp_path)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                           attn_implementation="eager")
    assert type(loaded.model).__module__ == "gptqmodel.models.ports.mimo_v2.modeling_mimo_v2"
    assert type(loaded.model.config).__module__ == "gptqmodel.models.ports.mimo_v2.configuration_mimo_v2"
    assert not loaded.trust_remote_code
    assert not marker.exists()
    with torch.inference_mode():
        assert torch.isfinite(loaded.model(torch.tensor([[1, 5, 7]])).logits).all()
