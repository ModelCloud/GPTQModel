# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.deepseek_v41.attention import V41AttentionState
from gptqmodel.models.definitions.deepseek_v41_flash import DeepSeekV41FlashQModel


@pytest.fixture
def v41_checkpoint(tmp_path):
    # Layers 0, 1, 2, 3, 4, 5 have the roles of published layers 1, 14, 15, 20, 21, 24.
    text = {
        "model_type": "deepseek_v41_text", "vocab_size": 32, "hidden_size": 32,
        "moe_intermediate_size": 32, "num_hidden_layers": 6,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "qk_rope_head_dim": 8,
        "q_lora_rank": 32, "o_lora_rank": 32, "o_groups": 2, "rms_norm_eps": 1e-20,
        "n_routed_experts": 3, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0, "sliding_window": 4, "compress_ratios": [0, 2, 2, 1, 1, 1, 0, 0],
        "kv_source_layer_ids": [1, 3], "index_source_layer_ids": [1, 3, 5],
        "compress_rope_theta": 160000, "index_n_heads": 2, "index_head_dim": 32, "index_topk": 3,
        "candidate_source_layer_id": 3, "candidate_topk_blocks": 1, "candidate_block_size": 2,
        "hc_mult": 2, "hc_sinkhorn_iters": 4, "hc_eps": 1e-6,
        "engram_layer_ids": [0, 1], "engram_num_embeddings": [384006168, 384016682],
        "num_nextn_predict_layers": 2, "dspark_target_layer_ids": [4, 5],
    }
    config = {
        "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 31,
        "text_config": text,
        "vision_config": {
            "model_type": "deepseek_v41_vision", "num_hidden_layers": 1, "hidden_size": 8,
            "num_attention_heads": 2, "intermediate_size": 16, "patch_size": 2,
            "downsample_ratio": 2, "rope_theta": 10000,
        },
        "quantization_config": {
            "quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
            "activation_scheme": "dynamic", "expert_dtype": "fp4",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    generator = torch.Generator().manual_seed(723)
    tensors = {}

    def dense(name, shape, *, dtype=torch.bfloat16, offset=0):
        tensors[name] = (torch.randn(shape, generator=generator) * 0.07 + offset).to(dtype)

    def packed(name, rows, cols, *, fp4=False):
        if fp4:
            weight = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator).view(torch.int8)
        else:
            weight = (torch.randn((rows, cols), generator=generator) * 2).to(torch.float8_e4m3fn)
        scales = torch.randint(-7, -4, (rows if fp4 else rows // 32, cols // 32), generator=generator)
        tensors[name + ".weight"] = weight
        tensors[name + ".scale"] = scales.exp2().to(torch.float8_e8m0fnu)

    dense("embed.weight", (32, 32))
    dense("head.weight", (32, 32))
    dense("norm.weight", (32,), offset=1)
    for layer in range(6):
        prefix = f"layers.{layer}."
        for name in ("attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"):
            dense(prefix + name + ".weight", (32,), offset=1)
        for branch in ("attn", "ffn"):
            dense(prefix + f"hc_{branch}_fn", (8, 64), dtype=torch.float32)
            dense(prefix + f"hc_{branch}_base", (8,), dtype=torch.float32)
            dense(prefix + f"hc_{branch}_scale", (3,), dtype=torch.float32, offset=1)
        dense(prefix + "attn.attn_sink", (2,), dtype=torch.float32)
        for name, rows, cols in (("wq_a", 32, 32), ("wq_b", 64, 32), ("wkv", 32, 32),
                                 ("wo_a", 64, 32), ("wo_b", 32, 64)):
            packed(prefix + "attn." + name, rows, cols)
        if layer in (1, 3, 5):
            packed(prefix + "attn.indexer.wq_b", 64, 32)
            dense(prefix + "attn.indexer.weights_proj.weight", (2, 32))
        if layer in (1, 3):
            for name in ("compressor.norm", "indexer.k_norm"):
                dense(prefix + f"attn.{name}.weight", (32,), offset=1)
            for name in ("compressor.wkv", "indexer.wk"):
                dense(prefix + f"attn.{name}.weight", (32, 32))
        if layer == 1:
            dense(prefix + "attn.compressor.wgate.weight", (32, 32))

        dense(prefix + "ffn.gate.weight", (3, 32))
        tensors[prefix + "ffn.gate.bias"] = torch.tensor([1.0, 0.0, -1.0])
        tensors[prefix + "ffn.gate.bias_vl"] = torch.tensor([-1.0, 0.0, 1.0])
        for projection in ("w1", "w2", "w3"):
            packed(prefix + "ffn.shared_experts." + projection, 32, 32)
            for expert in range(3):
                packed(prefix + f"ffn.experts.{expert}.{projection}", 32, 32, fp4=True)

    for name in ("image_start", "image_end", "image_newline"):
        dense(name, (32,))
    for name, rows, cols in (
        ("vision.patch_embed.proj", 8, 12), ("vision.blocks.0.attn.wqkv", 24, 8),
        ("vision.blocks.0.attn.wo", 8, 8), ("aligner.w1", 32, 32), ("aligner.w2", 32, 32),
    ):
        dense(name + ".weight", (rows, cols))
        dense(name + ".bias", (rows,))
    for name, rows, cols in (("w1", 32, 8), ("w2", 8, 16)):
        dense(f"vision.blocks.0.mlp.{name}.weight", (rows, cols))
    for name in ("norm", "blocks.0.norm1", "blocks.0.norm2"):
        dense("vision." + name + ".weight", (8,), offset=1)

    shard = "main.safetensors"
    weight_map = dict.fromkeys(tensors, shard)
    # Missing auxiliary-only shards must not be requested. A mixed shard may
    # also contain auxiliary keys; even their tensor payload is skipped.
    weight_map.update({
        "mtp.0.attn.wq_a.weight": "draft.safetensors",
        "layers.0.engram.embed.weight": "embeddings.safetensors",
        "layers.1.engram.q_weight": shard,
    })
    tensors["layers.1.engram.q_weight"] = torch.full((1,), float("nan"))
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8",
    )

    tokenizer = Tokenizer(WordLevel({"<s>": 0, "</s>": 1, "<pad>": 2, "<unk>": 3}, unk_token="<unk>"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
    ).save_pretrained(tmp_path)
    return tmp_path


@pytest.fixture
def model(v41_checkpoint):
    return GPTQModel.load(v41_checkpoint, device="cpu")


def test_v41_module_plan_accounts_for_loaded_decoder_weights(model):
    assert isinstance(model, DeepSeekV41FlashQModel)
    plan = model.module_plan()
    assert model.extract_layers_node() == ["layers"]
    assert not hasattr(model.model, "mtp")
    assert model.model.config.text_config.num_nextn_predict_layers == 0
    assert model.model.config.text_config.engram_layer_ids == []
    for layer in model.model.layers:
        for key, value in layer.state_dict().items():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None and "!" in spec.flags, key
            assert not value.is_meta
        assert not hasattr(layer, "engram")

    assert plan.expert_root("ffn.experts.2.w1") == "ffn.experts.2"
    assert plan.shared_expert_containers() == ("ffn.shared_experts",)
    assert plan.find("ffn.experts.2.w1.undeclared") is None
    assert plan.find("attn.compressor_extra.wkv") is None
    assert model.model.layers[0].ffn.experts[0].w1.weight.dtype == torch.int8
    assert model.model.layers[0].attn.wq_a.weight.dtype == torch.float8_e4m3fn
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.quantize(None)


def test_v41_attention_variants_and_chunked_cache(model):
    layers = model.model.layers
    assert [(attn.ratio, attn.compressor is not None, attn.indexer is not None)
            for attn in (layer.attn for layer in layers)] == [
        (0, False, False), (2, True, True), (2, False, False),
        (1, True, True), (1, False, False), (1, False, True),
    ]
    assert layers[1].attn.compressor.wgate is not None
    assert layers[3].attn.compressor.wgate is None

    tokens = torch.tensor([[4, 7, 9, 13, 16, 5, 6], [6, 5, 16, 13, 9, 7, 4]])
    full = model(tokens, use_cache=False).logits
    cache = model(tokens[:, :1]).past_key_values
    assert cache.layers[1].compressed.length == 0
    assert cache.layers[1].pool_tail.size(1) == 1
    second = model(tokens[:, 1:5], past_key_values=cache)
    assert second.past_key_values is cache
    assert cache.get_seq_length() == 5
    assert cache.layers[1].compressed.length == 2
    assert cache.layers[3].index_keys.length == 5
    assert cache.layers[5].compressed.length == 0
    assert all(layer.window.size(1) == 3 for layer in cache.layers)
    estimate = model.kv_cache_estimate(dtype=torch.bfloat16)
    assert estimate.kv_bytes_per_token == 192
    # Six windows, two compressed banks, and a one-token FP32 pooling tail.
    assert estimate.kv_bytes_for(5) == 6 * 3 * 32 * 2 + (2 + 5) * 64 * 2 + 2 * 32 * 4
    final = model(tokens[:, 5:], past_key_values=cache)
    assert final.past_key_values.get_seq_length() == 7
    assert torch.isfinite(full).all()
    torch.testing.assert_close(second.logits, full[:, 1:5], atol=0.012, rtol=0.015)
    torch.testing.assert_close(final.logits, full[:, 5:], atol=0.012, rtol=0.015)
    # A different request gets independent storage and cannot expose future inputs.
    prefix = model(tokens[:, :4]).logits
    torch.testing.assert_close(prefix, full[:, :4], atol=0.012, rtol=0.015)


def test_v41_sparse_attention_uses_only_the_window_and_selected_values(model):
    attention = model.model.layers[2].attn
    query = torch.arange(2 * 2 * 32).reshape(1, 2, 2, 32).float() / 100
    window = torch.arange(5 * 32).reshape(1, 5, 32).float() / 100
    compressed = -window[:, :3].clone()
    state = V41AttentionState(kv=compressed, selected=torch.tensor([[[1, -1], [0, 2]]]))
    actual = attention._attend(query, window, 3, state)

    expected = []
    for token, chosen in enumerate(([1], [0, 2])):
        values = torch.cat((window[0, token:token + 4], compressed[0, chosen]))
        scores = query[0, token] @ values.T / 32**0.5
        scores = torch.cat((scores, attention.attn_sink[:, None]), dim=-1)
        expected.append(scores.softmax(-1)[:, :-1] @ values)
    torch.testing.assert_close(actual[0], torch.stack(expected))


def test_v41_vision_uses_published_alignment_and_routing(model):
    model = model.model
    patches = torch.linspace(-1, 1, 9 * 12).reshape(9, 12)
    grid = torch.tensor([[1, 3, 3]])
    features = model.get_image_features(patches, grid)[0]
    assert features.shape == (8, 32)  # Two rows of two patches and a newline, plus the delimiters.
    torch.testing.assert_close(features[0], model.image_start)
    torch.testing.assert_close(features[3], model.image_newline)
    torch.testing.assert_close(features[6], model.image_newline)
    torch.testing.assert_close(features[7], model.image_end)
    gate = model.layers[0].ffn.gate
    indices, _ = gate(torch.zeros(2, 32, dtype=torch.bfloat16), torch.tensor([[False, True]]))
    assert indices.tolist() == [[0, 1], [2, 1]]
    logits = model(
        input_ids=torch.tensor([[4, *([31] * 8), 6]]),
        pixel_values=patches, image_grid_thw=grid,
    ).logits
    assert logits.shape == (1, 10, 32) and torch.isfinite(logits).all()


def test_v41_load_reports_missing_source_projection(v41_checkpoint):
    path = v41_checkpoint / "model.safetensors.index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    del index["weight_map"]["layers.3.attn.indexer.wq_b.scale"]
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Missing DeepSeek V4.1 checkpoint tensor layers\.3\.attn\.indexer\.wq_b\.scale"):
        GPTQModel.load(str(v41_checkpoint), device="cpu")


def test_v41_partial_forward_does_not_advance_cache(model):
    tokens = torch.tensor([[4, 7, 9, 13, 16, 5]])
    cache = model(tokens[:, :3]).past_key_values

    def fail(*args):
        raise RuntimeError("after compression")

    handle = model.model.layers[4].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="after compression"):
            model(tokens[:, 3:], past_key_values=cache)
    finally:
        handle.remove()
    assert cache.get_seq_length() == 3
    assert cache.layers[1].compressed.length == 1
    assert cache.layers[1].pool_tail.size(1) == 1
    assert cache.layers[3].compressed.length == 3

    retried = model(tokens[:, 3:], past_key_values=cache).logits
    fresh_cache = model(tokens[:, :3]).past_key_values
    expected = model(tokens[:, 3:], past_key_values=fresh_cache).logits
    torch.testing.assert_close(retried, expected, rtol=0, atol=0)


def test_v41_generation_uses_its_cache(model):
    tokens = torch.tensor([[4, 7, 9, 13]])
    next_token = model(tokens, use_cache=False).logits[:, -1].argmax(dim=-1)
    generated = model.generate(tokens, max_new_tokens=2, eos_token_id=None, do_sample=False)
    assert generated.shape == (1, 6)
    torch.testing.assert_close(generated[:, :4], tokens)
    torch.testing.assert_close(generated[:, 4], next_token)


def test_v41_packed_projection_matches_independent_weights_and_retains_packed_storage(model):
    layer = model.model.layers[0]
    x = torch.tensor([-2.0, -1.0, 0.0, 0.5] * 8, dtype=torch.bfloat16).unsqueeze(0)
    for projection in (layer.ffn.experts[0].w1, layer.attn.wq_a):
        raw = projection.weight
        if raw.dtype == torch.int8:
            levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
            rows = [[levels[code] for byte in row for code in (byte % 16, byte // 16)]
                    for row in raw.view(torch.uint8).tolist()]
            weight = torch.tensor(rows) * projection.scale.float()
        else:
            weight = raw.float() * projection.scale.float().item()
        expected = (x.float() @ weight.T).bfloat16()
        for _ in range(2):
            torch.testing.assert_close(projection(x), expected, atol=0, rtol=0)
            assert projection.weight is raw
            assert tuple(dict(projection.named_buffers())) == ("scale", "weight")


def test_v41_candidate_order_does_not_depend_on_masked_tail():
    from gptqmodel.models.deepseek_v41.attention import _candidate_blocks, _rank_entries

    scores = torch.tensor([[[3.0, 3.0, 3.0, 1.0, 1.0, -torch.inf, -torch.inf, -torch.inf]]])
    torch.testing.assert_close(_rank_entries(scores, 2), torch.tensor([[[0, 1]]], dtype=torch.int32))
    short = _rank_entries(scores[..., :5], 2)
    torch.testing.assert_close(_rank_entries(scores, 2), short)
    visible = torch.tensor([[5]])
    blocks = _candidate_blocks(scores, visible, block_size=2, topk=2)
    torch.testing.assert_close(blocks, torch.tensor([[[2, 0]]], dtype=torch.int32))


def test_v41_public_load_rejects_quantization(v41_checkpoint):
    with pytest.raises(NotImplementedError, match="inference-only"):
        GPTQModel.load(v41_checkpoint, device="cpu", quantize_config={"bits": 4, "group_size": 32})


def test_v41_hyper_connection_mixes_input_streams_into_each_output():
    from gptqmodel.models.deepseek_v41.layers import expand_streams

    residual = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    hidden = torch.tensor([[[10.0, 20.0]]])
    post_mix = torch.tensor([[[0.5, 1.5]]])
    residual_mix = torch.tensor([[[[0.25, 0.75], [0.5, 0.5]]]])
    expected = torch.tensor([[[[6.75, 12.5], [17.25, 33.5]]]])
    torch.testing.assert_close(expand_streams(hidden, residual, post_mix, residual_mix), expected)
