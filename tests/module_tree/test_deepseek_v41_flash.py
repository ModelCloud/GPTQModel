# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.deepseek_v41.attention import V41AttentionState
from gptqmodel.models.definitions.deepseek_v41_flash import DeepSeekV41FlashQModel


@pytest.fixture
def v41_checkpoint(tmp_path, request):
    width = getattr(request, "param", 32)
    # Layers 0, 1, 2, 3, 4, 5 have the roles of published layers 1, 14, 15, 20, 21, 24.
    text = {
        "model_type": "deepseek_v41_text", "vocab_size": 32, "hidden_size": width,
        "moe_intermediate_size": width, "num_hidden_layers": 6,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": width, "qk_rope_head_dim": 8,
        "q_lora_rank": width, "o_lora_rank": width, "o_groups": 2, "rms_norm_eps": 1e-20,
        "n_routed_experts": 3, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0, "sliding_window": 4, "compress_ratios": [0, 2, 2, 1, 1, 1, 0, 0],
        "kv_source_layer_ids": [1, 3], "index_source_layer_ids": [1, 3, 5],
        "compress_rope_theta": 160000, "index_n_heads": 2, "index_head_dim": 32, "index_topk": 3,
        "candidate_source_layer_id": 3, "candidate_topk_blocks": 1, "candidate_block_size": 2,
        "hc_mult": 2, "hc_sinkhorn_iters": 4, "hc_eps": 1e-6,
        "engram_layer_ids": [0, 1], "engram_num_embeddings": [48, 102],
        "engram_max_ngram_size": 3, "engram_vocab_size": 7, "engram_n_heads": 2, "engram_head_dim": 32,
        "engram_pad_token_id": 2, "engram_compressed_vocab_size": 32,
        "num_nextn_predict_layers": 2, "dspark_target_layer_ids": [4, 5],
        "dspark_block_size": 3, "dspark_noise_token_id": 30, "dspark_markov_rank": 8,
        "dspark_n_routed_experts": 2, "dspark_num_experts_per_tok": 1,
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

    dense("embed.weight", (32, width))
    dense("head.weight", (32, width))
    dense("norm.weight", (width,), offset=1)
    def decoder(prefix, experts):
        for name in ("attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"):
            dense(prefix + name + ".weight", (width,), offset=1)
        for branch in ("attn", "ffn"):
            dense(prefix + f"hc_{branch}_fn", (8, 2 * width), dtype=torch.float32)
            dense(prefix + f"hc_{branch}_base", (8,), dtype=torch.float32)
            dense(prefix + f"hc_{branch}_scale", (3,), dtype=torch.float32, offset=1)
        dense(prefix + "attn.attn_sink", (2,), dtype=torch.float32)
        for name, rows, cols in (("wq_a", width, width), ("wq_b", 2 * width, width), ("wkv", width, width),
                                 ("wo_a", 2 * width, width), ("wo_b", width, 2 * width)):
            packed(prefix + "attn." + name, rows, cols)
        dense(prefix + "ffn.gate.weight", (experts, width))
        tensors[prefix + "ffn.gate.bias"] = torch.tensor([1.0, 0.0, -1.0])[:experts].clone()
        tensors[prefix + "ffn.gate.bias_vl"] = torch.tensor([-1.0, 0.0, 1.0])[:experts].clone()
        for projection in ("w1", "w2", "w3"):
            packed(prefix + "ffn.shared_experts." + projection, width, width)
            for expert in range(experts):
                packed(prefix + f"ffn.experts.{expert}.{projection}", width, width, fp4=True)

    for layer in range(6):
        prefix = f"layers.{layer}."
        decoder(prefix, 3)
        if layer in (1, 3, 5):
            packed(prefix + "attn.indexer.wq_b", 64, width)
            dense(prefix + "attn.indexer.weights_proj.weight", (2, width))
        if layer in (1, 3):
            dense(prefix + "attn.compressor.norm.weight", (width,), offset=1)
            dense(prefix + "attn.indexer.k_norm.weight", (32,), offset=1)
            dense(prefix + "attn.compressor.wkv.weight", (width, width))
            dense(prefix + "attn.indexer.wk.weight", (32, width))
        if layer == 1:
            dense(prefix + "attn.compressor.wgate.weight", (width, width))

    for layer, rows in enumerate(text["engram_num_embeddings"]):
        prefix = f"layers.{layer}.engram."
        table = torch.randn((rows, 32), generator=generator).to(torch.float8_e4m3fn)
        tensors[prefix + "embed.weight"] = table
        tensors[prefix + "embed.scale"] = torch.full((rows, 1), 0.0625).to(torch.float8_e8m0fnu)
        for name in ("q_weight", "k_weight"):
            dense(prefix + name, (2, width), offset=1)
        packed(prefix + "wkv", 3 * width, 128)

    for layer in range(2):
        decoder(f"mtp.{layer}.", 2)
    packed("mtp.0.main_proj", width, width * 2)
    dense("mtp.0.main_norm.weight", (width,), offset=1)
    dense("mtp.1.norm.weight", (width,), offset=1)
    dense("mtp.1.markov_head.embed.weight", (32, 8))
    dense("mtp.1.markov_head.head.weight", (32, 8))
    dense("mtp.1.confidence_head.proj.weight", (1, width + 8))

    for name in ("image_start", "image_end", "image_newline"):
        dense(name, (width,))
    for name, rows, cols in (
        ("vision.patch_embed.proj", 8, 12), ("vision.blocks.0.attn.wqkv", 24, 8),
        ("vision.blocks.0.attn.wo", 8, 8), ("aligner.w1", width, 32), ("aligner.w2", width, width),
    ):
        dense(name + ".weight", (rows, cols))
        dense(name + ".bias", (rows,))
    for name, rows, cols in (("w1", 32, 8), ("w2", 8, 16)):
        dense(f"vision.blocks.0.mlp.{name}.weight", (rows, cols))
    for name in ("norm", "blocks.0.norm1", "blocks.0.norm2"):
        dense("vision." + name + ".weight", (8,), offset=1)

    shard = "main.safetensors"
    weight_map = dict.fromkeys(tensors, shard)
    for filename, names in (
        ("draft.safetensors", [name for name in tensors if name.startswith("mtp.") and "mtp.1.norm" not in name]),
        ("embeddings.safetensors", ["layers.0.engram.embed.weight", "layers.1.engram.embed.weight"]),
    ):
        save_file({name: tensors.pop(name) for name in names}, tmp_path / filename)
        weight_map.update(dict.fromkeys(names, filename))
    # Exercise repacking when only one operand is available from the first shard.
    if width != 32:
        separate = "additional.safetensors"
        projections = ("layers.1.attn.wq_a.scale", "layers.1.ffn.experts.0.w1.weight")
        save_file({name: tensors.pop(name) for name in projections}, tmp_path / separate)
        weight_map.update(dict.fromkeys(projections, separate))
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8",
    )

    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "<unk>": 3}
    vocab.update({f"t{i}": i for i in range(4, 32)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
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
    assert len(model.model.mtp) == model.model.config.text_config.num_nextn_predict_layers == 2
    assert model.model.config.text_config.engram_layer_ids == [0, 1]
    for layer in model.model.layers:
        for key, value in layer.state_dict().items():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert not value.is_meta
    for path in ("engram.embed", "engram.wkv", "engram"):
        assert "!" in plan.find(path).flags
    assert all(not value.is_meta for value in model.model.mtp.state_dict().values())

    assert plan.expert_root("ffn.experts.2.w1") == "ffn.experts.2"
    assert "!" not in plan.find("attn.wq_a").flags
    assert plan.shared_expert_containers() == ("ffn.shared_experts",)
    assert plan.find("ffn.experts.2.w1.undeclared") is None
    assert plan.find("attn.compressor_extra.wkv") is None
    assert model.model.layers[0].ffn.experts[0].w1.weight.dtype == torch.int8
    assert model.model.layers[0].attn.wq_a.weight.dtype == torch.float8_e4m3fn


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


def test_v41_engram_hash_uses_request_history_and_stops_at_images(model):
    net = model.model
    ids = torch.tensor([[4, 7, 31, 9, 13, 16]])
    image_mask = ids == 31
    digest = net.engram_hash
    context, _ = digest.prepare(ids, None, image_mask)
    row = ids.flatten().tolist()
    for layer, primes in ((0, (7, 11, 13, 17)), (1, (19, 23, 29, 31))):
        limit = (torch.iinfo(torch.int64).max // 32) // 2
        factors = (np.random.default_rng(layer * 10007).integers(limit, size=3) * 2 + 1).tolist()
        expected = []
        for index in range(len(row)):
            history = []
            for previous in reversed(row[:index + 1]):
                if previous == 31 or len(history) == 3:
                    break
                history.append(previous)
            history += [2] * (3 - len(history))
            one, two, three = [token_id * factor for token_id, factor in zip(history, factors)]
            offsets = [0, *np.cumsum(primes[:-1]).tolist()]
            expected.append([
                value % prime + offset
                for value, prime, offset in zip((one ^ two,) * 2 + (one ^ two ^ three,) * 2, primes, offsets)
            ])
        assert digest(context, layer).tolist() == [expected]
        prefix, tail = digest.prepare(ids[:, :4], None, image_mask[:, :4])
        suffix, _ = digest.prepare(ids[:, 4:], tail, None)
        torch.testing.assert_close(torch.cat((digest(prefix, layer), digest(suffix, layer)), dim=1), digest(context, layer))

    hidden = net.embed(ids).unsqueeze(-2).expand(-1, -1, 2, -1)
    engram = net.layers[0].engram
    output = engram(hidden, digest(context, 0), image_mask)
    torch.testing.assert_close(output[image_mask], hidden[image_mask], atol=0, rtol=0)
    assert not torch.equal(output[~image_mask], hidden[~image_mask])

    tokens = torch.tensor([[4, 7, 9]])
    result = net(tokens)
    tail = result.past_key_values.engram_tail
    assert tail.tolist() == [[7, 9]]
    assert not list(engram.embed.parameters()) and not list(engram.embed.buffers())
    torch.testing.assert_close(
        net(inputs_embeds=net.embed(tokens), engram_input_ids=tokens).logits, result.logits, atol=0, rtol=0,
    )


def test_v41_mtp_forward_is_explicit_and_disabled_during_calibration(model):
    net = model.model
    prefix = torch.tensor([[4, 7, 9]])
    next_token = torch.tensor([[11]])
    plain = net(prefix)
    assert plain.mtp_features is None
    assert plain.past_key_values.draft_windows == ()
    prompt = net(prefix, output_mtp_features=True)
    cache = prompt.past_key_values
    assert prompt.mtp_features.shape == (1, 3, 64)
    assert net.forward_spec(next_token[:, 0], prompt.mtp_features, cache=cache) is None
    assert cache.draft_length == 3 and len(cache.draft_windows) == len(net.mtp)
    continuation = net(next_token, past_key_values=cache, output_mtp_features=True)
    with net.calibration():
        assert net(prefix, output_mtp_features=True).mtp_features is None
        with pytest.raises(RuntimeError, match="draft execution is disabled"):
            net.forward_spec(next_token[:, 0], continuation.mtp_features, start_pos=3, cache=cache)
    assert cache.draft_length == 3

    tokens, logits, confidence = net.forward_spec(
        next_token[:, 0], continuation.mtp_features, start_pos=3, cache=cache,
    )
    assert cache.draft_length == 4 and all(window.size(1) == 4 for window in cache.draft_windows)
    assert tokens.shape == (1, 4) and logits.shape == (1, 3, 32) and confidence.shape == (1, 3)
    assert torch.isfinite(logits).all() and torch.isfinite(confidence).all()
    torch.testing.assert_close(tokens[:, :1], next_token)
    torch.testing.assert_close(tokens[:, 1:], logits.argmax(dim=-1))


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
    tail = cache.engram_tail.clone()

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
    torch.testing.assert_close(cache.engram_tail, tail)

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


@pytest.mark.cuda
@pytest.mark.parametrize("v41_checkpoint", [128], indirect=True)
def test_v41_cuda_load_runs_packed_projections(v41_checkpoint, monkeypatch):
    from gptqmodel.models._const import DEVICE, normalize_device
    from gptqmodel.models.deepseek_v41 import layers
    from gptqmodel.utils.macaq_triton import TRITON_AVAILABLE
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error

    device = torch.device("cuda", torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device("cpu")
    if normalize_device(device) != DEVICE.CUDA or torch.cuda.get_device_capability(device) < (8, 0):
        pytest.skip("NVIDIA CUDA compute capability 8.0 or later is required")
    native_mx = torch.cuda.get_device_capability(device)[0] in (10, 11, 12)
    if native_mx and not TRITON_AVAILABLE:
        pytest.skip("Triton is required for native MX")
    if not native_mx and not marlin_runtime_available(torch.bfloat16):
        pytest.skip(marlin_runtime_error(torch.bfloat16))

    source = GPTQModel.load(v41_checkpoint, device="cpu").model
    loaded = GPTQModel.load(v41_checkpoint, device_map="auto", max_memory={device.index: "1GiB"}).model
    reference = GPTQModel.load(v41_checkpoint, device=device, backend="torch").model
    assert isinstance(reference.layers[1].attn.wq_a, layers.V41PackedLinear)
    assert torch.equal(
        reference.layers[1].attn.wq_a.weight.cpu().view(torch.uint8),
        source.layers[1].attn.wq_a.weight.view(torch.uint8),
    )

    def dense_decode(*args, **kwargs):
        pytest.fail("A CUDA projection decoded the entire packed weight")

    monkeypatch.setattr(layers, "dequantize_fp8", dense_decode)
    monkeypatch.setattr(layers, "dequantize_deepseek_v4_fp4_expert", dense_decode)
    names = ("attn.wq_a", "attn.wo_a", "attn.indexer.wq_b", "ffn.experts.0.w1")
    generator = torch.Generator().manual_seed(13)
    fp4_levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]).double()
    for name in names:
        raw = source.layers[1].get_submodule(name)
        projection = loaded.layers[1].get_submodule(name)
        exponents = raw.scale.view(torch.uint8).double() - 127
        scales = exponents.exp2().repeat_interleave(32, dim=1)
        if raw.fp4:
            unsigned = raw.weight.view(torch.uint8).long()
            codes = torch.stack((unsigned & 15, unsigned >> 4), dim=-1).flatten(1)
            weight = fp4_levels[codes] * scales
        else:
            weight = raw.weight.double() * scales.repeat_interleave(32, dim=0)

        buffers = tuple(projection.buffers())
        assert not list(projection.parameters())
        projection.to(dtype=torch.bfloat16)
        if native_mx:
            assert isinstance(projection, layers.V41NativeMXLinear)
            assert torch.equal(projection.weight.cpu().view(torch.uint8), raw.weight.view(torch.uint8))
            assert torch.equal(projection.scale.cpu(), raw.scale.view(torch.uint8))
        else:
            assert isinstance(projection, layers.V41MarlinLinear)
            assert projection.qweight.dtype == torch.int32 and projection.qweight.nbytes == raw.weight.nbytes
            assert projection.scales.dtype == torch.uint8
            assert projection.scales.nbytes == raw.scale.nbytes * (1 if raw.fp4 else 32)
        for tokens in (1, 17):
            shape = (2, tokens, raw.groups, weight.size(1)) if raw.groups > 1 else (tokens, weight.size(1))
            x = torch.randn(shape, generator=generator).bfloat16()
            blocks = x.double().unflatten(-1, (-1, 32))
            scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
            quantized = ((blocks / scale).to(torch.float8_e4m3fn).double() * scale).flatten(-2)
            if raw.groups > 1:
                expected = torch.einsum("bsgd,grd->bsgr", quantized, weight.unflatten(0, (raw.groups, -1)))
            else:
                expected = quantized @ weight.T
            actual = projection(x.to(device)).cpu()
            torch.testing.assert_close(actual, expected.bfloat16(), atol=2e-3, rtol=2e-3)
            assert all(a is b for a, b in zip(projection.buffers(), buffers, strict=True))
            assert all(t.device == device for t in buffers)


def test_v41_candidate_order_does_not_depend_on_masked_tail():
    from gptqmodel.models.deepseek_v41.attention import _candidate_blocks, _rank_entries

    scores = torch.tensor([[[3.0, 3.0, 3.0, 1.0, 1.0, -torch.inf, -torch.inf, -torch.inf]]])
    torch.testing.assert_close(_rank_entries(scores, 2), torch.tensor([[[0, 1]]], dtype=torch.int32))
    short = _rank_entries(scores[..., :5], 2)
    torch.testing.assert_close(_rank_entries(scores, 2), short)
    visible = torch.tensor([[5]])
    blocks = _candidate_blocks(scores, visible, block_size=2, topk=2)
    torch.testing.assert_close(blocks, torch.tensor([[[2, 0]]], dtype=torch.int32))


def test_v41_gptq_export_preserves_engram_and_mtp(v41_checkpoint, tmp_path):
    from gptqmodel.quantization.config import CalibrationConfig, QuantizeConfig

    wrapped = GPTQModel.load(
        v41_checkpoint, device="cpu",
        quantize_config=QuantizeConfig(bits=4, group_size=32, desc_act=False, act_group_aware=False),
    )
    calibration = CalibrationConfig(dataset=[[4, 7, 8, 9, 11, 12], [21, 20, 16, 15, 10, 13]], min_length=0)
    for projection in ("layers.0.engram.wkv", "mtp.0.attn.wq_a"):
        with pytest.raises(ValueError, match="preserves the published weights"):
            wrapped.quantize(calibration, projections=[projection])

    seen = []
    def observe_engram(*_):
        seen.append(True)
    def fail_mtp(*_):
        pytest.fail("The draft path ran during calibration")
    handles = [wrapped.model.layers[0].engram.register_forward_pre_hook(observe_engram)]
    handles.extend(block.register_forward_pre_hook(fail_mtp) for block in wrapped.model.mtp)
    try:
        report = wrapped.quantize(calibration, projections=["layers.0.attn.wq_a", "layers.1.attn.wo_a"])
    finally:
        for handle in handles:
            handle.remove()
    assert seen
    assert wrapped.quantized and len(report["gptq"]) == 3

    output = tmp_path / "gptq"
    wrapped.save(output, max_shard_size=11000)
    source_map = json.loads((v41_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    saved_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    for name in source_map:
        if not (name.startswith("mtp.") or ".engram." in name):
            continue
        with safe_open(v41_checkpoint / source_map[name], framework="pt") as src, safe_open(
            output / saved_map[name], framework="pt",
        ) as dst:
            a, b = src.get_tensor(name), dst.get_tensor(name)
            assert (a.dtype, a.shape) == (b.dtype, b.shape)
            assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)), name
    assert "layers.0.attn.wq_a.weight" not in saved_map
    assert "layers.0.attn.wq_a.projections.0.qweight" in saved_map
    reloaded = GPTQModel.load(output, device="cpu")
    tokens = torch.tensor([[4, 11, 9]])
    assert reloaded.quantized and len(reloaded.model.mtp) == 2
    assert reloaded.model.config.text_config.engram_layer_ids == [0, 1]
    torch.testing.assert_close(reloaded(tokens).logits, wrapped(tokens).logits, atol=0, rtol=0)


def test_v41_marlin_backend_requires_cuda(v41_checkpoint):
    with pytest.raises(ValueError, match="Marlin requires NVIDIA CUDA"):
        GPTQModel.load(v41_checkpoint, device="cpu", backend="marlin")


def test_v41_auto_prefers_native_mx_on_blackwell(monkeypatch):
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.deepseek_v41 import checkpoint
    from gptqmodel.utils.backend import BACKEND

    ampere = torch.device("cuda:1")
    blackwell = torch.device("cuda:3")
    monkeypatch.setattr(checkpoint, "normalize_device", lambda device: DEVICE.CUDA)
    capability = {ampere: (8, 0), blackwell: (12, 0)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability.__getitem__)
    assert checkpoint._device_projection_backend(ampere, BACKEND.AUTO) == BACKEND.MARLIN
    assert checkpoint._device_projection_backend(blackwell, BACKEND.AUTO) == BACKEND.TRITON
    assert checkpoint._device_projection_backend(blackwell, BACKEND.MARLIN) == BACKEND.MARLIN
    assert checkpoint._device_projection_backend(blackwell, BACKEND.TORCH) == BACKEND.TORCH
    monkeypatch.setattr(checkpoint, "TRITON_AVAILABLE", True)
    monkeypatch.setattr(checkpoint, "marlin_runtime_available", lambda _: pytest.fail("Marlin was loaded on Blackwell"))
    with torch.device("meta"):
        model = torch.nn.Sequential(checkpoint.V41PackedLinear(128, 128))
    classes = checkpoint._configure_projection_backends(model, {"0.weight": blackwell}, BACKEND.AUTO)
    assert classes == {"0": checkpoint.V41NativeMXLinear}
    monkeypatch.setattr(checkpoint, "normalize_device", lambda device: DEVICE.ROCM)
    assert checkpoint._device_projection_backend(blackwell, BACKEND.AUTO) == BACKEND.TORCH


def test_v41_hyper_connection_normalizes_before_rounding():
    from gptqmodel.models.deepseek_v41.layers import V41HyperConnection

    config = SimpleNamespace(hc_mult=2, hc_eps=1e-6, rms_norm_eps=1e-20, hc_sinkhorn_iters=4, hidden_size=32)
    generator = torch.Generator().manual_seed(609)
    connection = V41HyperConnection(config)
    with torch.no_grad():
        connection.fn.copy_(torch.randn(connection.fn.shape, generator=generator) / 8)
        connection.base.zero_()
        connection.scale.fill_(1)
    residual = torch.randn((2, 13, 2, config.hidden_size), generator=generator).bfloat16()
    rows = residual.double().flatten(-2)
    dot = (rows.unsqueeze(-2) * connection.fn[:config.hc_mult].double()).sum(-1)
    rms = (rows.square().mean(-1, keepdim=True) + config.rms_norm_eps).sqrt()
    expected = (dot / rms).float().sigmoid() + config.hc_eps
    torch.testing.assert_close(connection(residual)[0], expected, atol=0, rtol=0)


def test_v41_hyper_connection_mixes_input_streams_into_each_output():
    from gptqmodel.models.deepseek_v41.layers import expand_streams

    residual = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    hidden = torch.tensor([[[10.0, 20.0]]])
    post_mix = torch.tensor([[[0.5, 1.5]]])
    residual_mix = torch.tensor([[[[0.25, 0.75], [0.5, 0.5]]]])
    expected = torch.tensor([[[[6.75, 12.5], [17.25, 33.5]]]])
    torch.testing.assert_close(expand_streams(hidden, residual, post_mix, residual_mix), expected)
