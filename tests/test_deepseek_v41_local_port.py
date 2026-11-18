# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 port: numerical oracle against DeepSeek's reference and checkpoint inventory.

The oracle (``tests/fixtures/ports/deepseek_v41/reference_model.py``) is DeepSeek's
``inference/model.py`` with dense FP32 storage and its TileLang kernels re-expressed in
PyTorch. Weights are mapped from the port into the oracle through
``gptqmodel.utils.v41_source.source_model_key``, so every comparison also checks the
runtime-to-checkpoint key mapping.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41QModel
from gptqmodel.models.ports.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config
from gptqmodel.models.ports.deepseek_v41.hashing_deepseek_v41 import DeepseekV41NgramHash, EngramLayout
from gptqmodel.models.ports.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41Cache,
    DeepseekV41DSpark,
    DeepseekV41ForCausalLM,
    DeepseekV41RotaryEmbedding,
)
from gptqmodel.utils.v41_replay import V41ReplayBatch
from gptqmodel.utils.v41_source import V41Source, source_layer_key, source_model_key


FIXTURES = Path(__file__).parent / "fixtures" / "ports" / "deepseek_v41"

# deepseek-ai/DeepSeek-V4.1-Flash @ 314113f8a7dadb388215396ba5fbee2c610a04d6
# model.safetensors.index.json: 96,085 keys. The text model and dSpark cover every key
# except the vision tower, the aligner, the image-span parameters and the FP8/FP4
# block scales of dense-decoded linears (the Engram tables keep theirs).
TEXT_INVENTORY_COUNT = 48232
TEXT_INVENTORY_SHA256 = "55c35d48c5424b3618add7cde8987a651f5fab6ba151a03869a6a9ee1bfcc52c"


def _reference():
    spec = importlib.util.spec_from_file_location("deepseek_v41_reference", FIXTURES / "reference_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _reference()

TINY_TEXT = {
    "vocab_size": 96,
    "hidden_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "head_dim": 32,
    "qk_rope_head_dim": 8,
    "q_lora_rank": 24,
    "o_lora_rank": 16,
    "o_groups": 2,
    "swiglu_limit": 0.75,
    "rms_norm_eps": 1e-20,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                  "original_max_position_embeddings": 64},
    "n_routed_experts": 6,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 1.5,
    "sliding_window": 4,
    "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1, 0, 0],
    "kv_source_layer_ids": [2, 5],
    "index_source_layer_ids": [2, 5, 7],
    # Enough indexer heads that ReLU-zeroed scores rarely tie: the reference top-k
    # resolves exact ties by width, so its prefill would otherwise depend on length.
    "index_n_heads": 8,
    "index_head_dim": 16,
    "index_topk": 3,
    "candidate_source_layer_id": 5,
    "candidate_topk_blocks": 2,
    "candidate_block_size": 2,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [1, 4],
    "engram_max_ngram_size": 3,
    "engram_vocab_size": 40,
    "engram_n_heads": 2,
    "engram_head_dim": 32,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 50,
    "num_nextn_predict_layers": 2,
    "dspark_block_size": 3,
    "dspark_noise_token_id": 95,
    "dspark_target_layer_ids": [5, 6, 7],
    "dspark_markov_rank": 16,
    "dspark_n_routed_experts": 4,
    "dspark_num_experts_per_tok": 2,
}


def _engram_rows(text):
    args = REF.ModelArgs(engram_layer_ids=tuple(text["engram_layer_ids"]), engram_num_embeddings=(0, 0),
                         engram_max_ngram_size=text["engram_max_ngram_size"],
                         engram_vocab_size=text["engram_vocab_size"], engram_n_heads=text["engram_n_heads"],
                         engram_head_dim=text["engram_head_dim"])
    return [sum(prime for ngram in layer for prime in ngram) for layer in REF.EngramLayout.from_args(args).primes]


def _tiny_config():
    text = {**TINY_TEXT, "engram_num_embeddings": _engram_rows(TINY_TEXT)}
    return DeepseekV41Config(text_config=text, vision_config={"num_hidden_layers": 0})


def _reference_args(text, max_seq_len=64):
    scaling = text.rope_parameters["compress"]
    return REF.ModelArgs(
        max_batch_size=2, max_seq_len=max_seq_len, temperature=0, expert_dtype=None,
        vocab_size=text.vocab_size, dim=text.hidden_size, moe_inter_dim=text.moe_intermediate_size,
        n_layers=text.num_hidden_layers, n_mtp_layers=text.num_nextn_predict_layers, n_heads=text.num_attention_heads,
        n_routed_experts=text.n_routed_experts, n_activated_experts=text.num_experts_per_tok,
        score_func=text.scoring_func, route_scale=text.routed_scaling_factor, swiglu_limit=text.swiglu_limit,
        q_lora_rank=text.q_lora_rank, head_dim=text.head_dim, rope_head_dim=text.qk_rope_head_dim,
        norm_eps=text.rms_norm_eps, o_groups=text.o_groups, o_lora_rank=text.o_lora_rank,
        window_size=text.sliding_window, compress_ratios=tuple(text.compress_ratios),
        kv_source_layers=tuple(text.kv_source_layer_ids), index_source_layers=tuple(text.index_source_layer_ids),
        compress_rope_theta=text.compress_rope_theta, original_seq_len=scaling["original_max_position_embeddings"],
        rope_theta=text.rope_theta, rope_factor=scaling["factor"], beta_fast=scaling["beta_fast"],
        beta_slow=scaling["beta_slow"], index_n_heads=text.index_n_heads, index_head_dim=text.index_head_dim,
        index_topk=text.index_topk, candidate_source_layer=text.candidate_source_layer_id,
        candidate_topk_blocks=text.candidate_topk_blocks, candidate_block_size=text.candidate_block_size,
        hc_mult=text.hc_mult, hc_sinkhorn_iters=text.hc_sinkhorn_iters, hc_eps=text.hc_eps,
        engram_layer_ids=tuple(text.engram_layer_ids), engram_num_embeddings=tuple(text.engram_num_embeddings),
        engram_max_ngram_size=text.engram_max_ngram_size, engram_vocab_size=text.engram_vocab_size,
        engram_n_heads=text.engram_n_heads, engram_head_dim=text.engram_head_dim,
        engram_pad_id=text.engram_pad_token_id, engram_compressed_vocab_size=text.engram_compressed_vocab_size,
        dspark_block_size=text.dspark_block_size, dspark_noise_token_id=text.dspark_noise_token_id,
        dspark_target_layer_ids=tuple(text.dspark_target_layer_ids), dspark_markov_rank=text.dspark_markov_rank,
        dspark_n_routed_experts=text.dspark_n_routed_experts,
        dspark_n_activated_experts=text.dspark_num_experts_per_tok,
    )


@torch.no_grad()
def _randomize(module, generator):
    for name, param in module.named_parameters():
        if param.dtype == torch.float8_e4m3fn:
            param.copy_((torch.randn(param.shape, generator=generator) * 2).to(torch.float8_e4m3fn))
        elif param.dtype == torch.float8_e8m0fnu:
            param.copy_(torch.exp2(torch.randint(-3, 1, param.shape, generator=generator).float()).to(param.dtype))
        elif name.endswith(("norm.weight", "layernorm.weight", "q_weight", "k_weight")):
            param.copy_(1 + 0.2 * torch.randn(param.shape, generator=generator))
        elif name.endswith((".scale", ".base", "sinks")):
            param.copy_(0.5 + torch.rand(param.shape, generator=generator) if name.endswith(".scale")
                        else torch.randn(param.shape, generator=generator))
        else:
            param.copy_(torch.randn(param.shape, generator=generator) * param.shape[-1] ** -0.5)
    for name, buffer in module.named_buffers():
        if "e_score_correction_bias" in name:
            buffer.copy_(0.3 * torch.randn(buffer.shape, generator=generator))


def _expand_experts(key, value):
    prefix, _, leaf = key.rpartition(".")
    if leaf == "gate_up_proj":
        gate, up = value.chunk(2, dim=1)
        for expert in range(value.shape[0]):
            yield f"{prefix}.{expert}.gate_proj.weight", gate[expert]
            yield f"{prefix}.{expert}.up_proj.weight", up[expert]
    elif leaf == "down_proj" and prefix.endswith(".experts"):
        for expert in range(value.shape[0]):
            yield f"{prefix}.{expert}.down_proj.weight", value[expert]
    else:
        yield key, value


def _source_state(*modules):
    """Map port tensors to checkpoint keys (``prefix`` is ``""`` or ``"mtp."``)."""

    state = {}
    for prefix, module in modules:
        for key, value in module.state_dict().items():
            for runtime_key, tensor in _expand_experts(prefix + key, value):
                state[source_model_key(runtime_key)] = tensor
    return state


def _build(seed=0):
    config = _tiny_config()
    text = config.text_config
    generator = torch.Generator().manual_seed(seed)
    model = DeepseekV41ForCausalLM(config).eval()
    _randomize(model, generator)
    token_map = torch.randint(0, text.engram_compressed_vocab_size, (text.vocab_size,), generator=generator)
    model.model.bind_engram_hash(DeepseekV41NgramHash(text, token_map))
    dspark = DeepseekV41DSpark(text).eval()
    _randomize(dspark, generator)

    reference = REF.Transformer(_reference_args(text), tokenizer=token_map.tolist()).eval()
    state = _source_state(("", model), ("mtp.", dspark))
    missing, unexpected = reference.load_state_dict(state, strict=False)
    assert not unexpected
    # dSpark stages share the backbone embedding and head modules.
    assert sorted(missing) == sorted(f"mtp.{stage}.{name}.weight" for stage in range(2) for name in ("embed", "head"))
    return model, dspark, reference, token_map


@pytest.fixture(scope="module")
def built():
    return _build()


def _tokens(length, seed=1, batch=2):
    return torch.randint(0, TINY_TEXT["vocab_size"], (batch, length), generator=torch.Generator().manual_seed(seed))


def _close(actual, expected, atol=2e-5, rtol=2e-5):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def test_rotary_and_engram_layout_match_reference():
    config = DeepseekV41Config.from_dict(json.loads((FIXTURES / "upstream_config.json").read_text()))
    text = config.text_config
    rotary = DeepseekV41RotaryEmbedding(text)
    positions = torch.arange(0, 70000, 997)[None]
    for family, (original, theta) in {"main": (0, text.rope_theta), "compress": (65536, text.compress_rope_theta)}.items():
        expected = REF.precompute_freqs_cis(text.qk_rope_head_dim, 70000, original, theta, 16, 32, 1)[positions[0]]
        cos, sin = rotary(torch.zeros(1), positions, family)
        torch.testing.assert_close(torch.complex(cos, sin)[0], expected, rtol=0, atol=0)
    # The prime bucket layout reproduces the released table row counts.
    layout = EngramLayout.from_config(text)
    assert [sum(p for ngram in layer for p in ngram) for layer in layout.primes] == text.engram_num_embeddings


def test_ngram_hash_matches_reference_prefill_and_decode(built):
    model, _, reference, _ = built
    hasher = model.model.engram_hash
    tokens = _tokens(9)
    tokens[0, 3] = TINY_TEXT["engram_pad_token_id"]
    expected = reference.engram_hash(tokens[:, :6], 0)
    torch.testing.assert_close(hasher(tokens[:, :6]), expected, rtol=0, atol=0)
    history = hasher.next_history(None, tokens[:, :6])
    for position in range(6, 9):
        step = tokens[:, position : position + 1]
        torch.testing.assert_close(hasher(step, history), reference.engram_hash(step, position), rtol=0, atol=0)
        history = hasher.next_history(history, step)


def _capture(layers):
    outputs = []
    handles = [layer.register_forward_hook(lambda module, args, output: outputs.append(output)) for layer in layers]
    return outputs, handles


@torch.no_grad()
def test_prefill_matches_reference_layer_by_layer(built):
    model, _, reference, _ = built
    tokens = _tokens(13)
    port_outputs, handles = _capture(model.model.layers)
    ref_outputs, ref_handles = _capture(reference.layers)
    try:
        port = model(input_ids=tokens, use_cache=False, output_dspark_hidden=True)
        _, ref_logits, ref_main_hidden = reference(tokens, 0)
    finally:
        for handle in handles + ref_handles:
            handle.remove()
    assert len(port_outputs) == len(ref_outputs) == TINY_TEXT["num_hidden_layers"]
    for (hidden, pre_mix), (ref_hidden, ref_pre_mix) in zip(port_outputs, ref_outputs):
        _close(hidden, ref_hidden)
        _close(pre_mix, ref_pre_mix)
    _close(port.logits[:, -1], ref_logits, atol=5e-5)
    _close(port.dspark_hidden, ref_main_hidden)


@torch.no_grad()
def test_cached_decode_matches_reference_and_full_forward(built):
    model, _, reference, _ = built
    tokens = _tokens(15, seed=2)
    prompt = 7
    cache = DeepseekV41Cache(model.config.text_config)
    port = model(input_ids=tokens[:, :prompt], past_key_values=cache, use_cache=True)
    _, ref_logits, _ = reference(tokens[:, :prompt], 0)
    _close(port.logits[:, -1], ref_logits, atol=5e-5)
    stepped = [port.logits]
    for position in range(prompt, tokens.shape[1]):
        port = model(input_ids=tokens[:, position : position + 1], past_key_values=cache, use_cache=True)
        _, ref_logits, _ = reference(tokens[:, position : position + 1], position)
        _close(port.logits[:, -1], ref_logits, atol=5e-5)
        stepped.append(port.logits)
    full = model(input_ids=tokens, use_cache=False).logits
    _close(torch.cat(stepped, dim=1), full, atol=5e-5)
    # A multi-token continuation keeps per-query causal visibility.
    chunked = DeepseekV41Cache(model.config.text_config)
    first = model(input_ids=tokens[:, :5], past_key_values=chunked).logits
    rest = model(input_ids=tokens[:, 5:], past_key_values=chunked).logits
    _close(torch.cat([first, rest], dim=1), full, atol=5e-5)


@torch.no_grad()
def test_dspark_matches_reference_forward_spec(built):
    model, dspark, reference, _ = built
    tokens = _tokens(12, seed=3)
    prompt = 8
    output_ids, _, main_hidden = reference(tokens[:, :prompt], 0)
    reference.forward_spec(output_ids, main_hidden, 0)
    history = [main_hidden]
    for position in range(prompt, tokens.shape[1]):
        output_ids, _, main_hidden = reference(tokens[:, position : position + 1], position)
        history.append(main_hidden)
        ref_ids, ref_logits, ref_confidence = reference.forward_spec(output_ids, main_hidden, position)
        port = dspark(torch.cat(history, dim=1), output_ids, model.model.embed_tokens, model.lm_head)
        _close(port.logits, ref_logits, atol=5e-5)
        _close(port.confidence, ref_confidence, atol=5e-5)
        assert torch.equal(port.output_ids, ref_ids)


@torch.no_grad()
def test_joint_dspark_anchors_equal_independent_histories(built):
    attention = built[1].layers[0].self_attn
    generator = torch.Generator().manual_seed(10)
    main_x = torch.randn(1, 9, TINY_TEXT["hidden_size"], generator=generator)
    drafts = torch.randn(3, TINY_TEXT["dspark_block_size"], TINY_TEXT["hidden_size"], generator=generator)
    anchors = torch.tensor([1, 4, 8])
    joint, _ = attention(drafts, {}, main_x=main_x, anchor_positions=anchors)
    for row, anchor in enumerate(anchors.tolist()):
        single, _ = attention(drafts[row : row + 1], {}, main_x=main_x[:, : anchor + 1])
        _close(joint[row : row + 1], single, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_image_routing_bias_and_engram_mask_match_reference(built):
    model, _, reference, _ = built
    layer, ref_layer = model.model.layers[4], reference.layers[4]
    hidden = torch.randn(2, 5, TINY_TEXT["hidden_size"], generator=torch.Generator().manual_seed(4))
    image = torch.tensor([[True, False, True, False, False], [False] * 5])
    logits, weights, indices = layer.mlp.gate(hidden.flatten(0, 1), image.flatten())
    ref_weights, ref_indices = ref_layer.ffn.gate(hidden.flatten(0, 1), image.flatten())
    assert torch.equal(indices, ref_indices)
    _close(weights, ref_weights)
    _close(layer.mlp(hidden, image), ref_layer.ffn(hidden, image))
    streams = torch.randn(2, 5, TINY_TEXT["hc_mult"], TINY_TEXT["hidden_size"], generator=torch.Generator().manual_seed(5))
    hashes = model.model.engram_hash(_tokens(5))[:, :, 1]
    rows = model.model.engram_tables["4"](hashes)
    _close(layer.engram(streams, rows, ~image), ref_layer.engram(streams, hashes, ~image))


@torch.no_grad()
def test_generate_with_cache_equals_uncached_greedy(built):
    model = built[0]
    prompt = _tokens(6, seed=6, batch=1)
    config = copy.deepcopy(model.generation_config)
    config.eos_token_id = -1  # never stop early: compare all six steps
    generated = model.generate(prompt, generation_config=config, max_new_tokens=6, do_sample=False,
                               attention_mask=torch.ones_like(prompt))
    tokens = prompt
    for _ in range(6):
        next_token = model(input_ids=tokens, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    assert torch.equal(generated, tokens)
    with pytest.raises(ValueError, match="padded"):
        model(input_ids=prompt, attention_mask=torch.zeros_like(prompt).index_fill_(1, torch.tensor([1]), 1))


@torch.no_grad()
def test_unfused_definition_experts_and_replay_frontiers_keep_outputs(built):
    model = built[0]
    tokens = _tokens(10, seed=7)
    expected = model(input_ids=tokens, use_cache=False).logits
    clone = DeepseekV41ForCausalLM(model.config).eval()
    clone.load_state_dict(model.state_dict())
    clone.model.bind_engram_hash(model.model.engram_hash)
    assert DeepSeekV41QModel.convert_model_structure(clone)
    _close(clone(input_ids=tokens, use_cache=False).logits, expected, atol=1e-6, rtol=1e-6)

    # Explicit calibration frontiers replay the same decoder, one block at a time.
    state = V41ReplayBatch.prepare(clone, tokens)
    for layer in clone.model.layers:
        state = state.advance(layer, "cpu")
    collapsed = clone.model.layers[-1].hc_collapse(state.hidden, state.pre_mix)
    logits = torch.nn.functional.linear(clone.model.norm(collapsed).float(), clone.lm_head.weight.float())
    _close(logits, expected, atol=1e-6, rtol=1e-6)
    hidden = model(input_ids=tokens, use_cache=False, output_dspark_hidden=True).dspark_hidden
    _close(torch.cat([state.target_features[i] for i in TINY_TEXT["dspark_target_layer_ids"]], -1), hidden)


@torch.no_grad()
def test_v41_source_loads_blocks_matching_the_port(built, tmp_path):
    model, dspark, _, _ = built
    config = model.config.to_dict()
    config["model_type"] = "deepseek_v41"
    state = {key: value.contiguous() for key, value in _source_state(("", model), ("mtp.", dspark)).items()}
    state["embed.weight"] = state["embed.weight"].bfloat16()  # released dtype, required by load_main_input
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(state, "model.safetensors")}))
    source = V41Source(tmp_path)
    tokens = _tokens(9, seed=8)
    frontier = V41ReplayBatch.prepare(model, tokens)
    for index in range(TINY_TEXT["num_hidden_layers"]):
        block = source.load_decoded_block(index, device="cpu")
        expected = frontier.advance(model.model.layers[index], "cpu")
        actual = frontier.advance(block, "cpu")
        _close(actual.hidden, expected.hidden, atol=0, rtol=0)
        _close(actual.pre_mix, expected.pre_mix, atol=0, rtol=0)
        frontier = expected

    # Main-input preparation reads the Engram rows through memory-mapped tables.
    main = source.load_main_input(model.model.engram_hash, device="cpu")
    try:
        prepared = main.prepare(tokens)
    finally:
        main.close()
    hashes = model.model.engram_hash(tokens)
    for index, layer in enumerate(TINY_TEXT["engram_layer_ids"]):
        torch.testing.assert_close(prepared.engram_rows[layer], model.model.engram_tables[str(layer)](hashes[:, :, index]),
                                   rtol=0, atol=0)
    positions = torch.arange(tokens.shape[1])[None].expand(tokens.shape[0], -1)
    for kind in ("main", "compress"):
        expected_cos, expected_sin = model.model.rotary_emb(tokens.float(), positions, kind)
        torch.testing.assert_close(prepared.kwargs["position_embeddings"][kind][0], expected_cos, rtol=0, atol=0)
        torch.testing.assert_close(prepared.kwargs["position_embeddings"][kind][1], expected_sin, rtol=0, atol=0)


def _expected_text_inventory(index_keys):
    return sorted(
        key for key in index_keys
        if not key.startswith(("vision.", "aligner.", "image_"))
        and not (key.endswith(".scale") and ".engram.embed." not in key)
    )


def test_full_size_meta_model_maps_onto_the_released_text_inventory():
    config = DeepseekV41Config.from_dict(json.loads((FIXTURES / "upstream_config.json").read_text()))
    text = config.text_config
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config)
        dspark = DeepseekV41DSpark(text)
    mapped = []
    for prefix, module in (("", model), ("mtp.", dspark)):
        for key, value in module.state_dict().items():
            mapped.extend(source_model_key(runtime) for runtime, _ in _expand_experts(prefix + key, value))
    assert len(mapped) == len(set(mapped))
    mapped = sorted(mapped)
    digest = hashlib.sha256("\n".join(mapped).encode()).hexdigest()
    assert (len(mapped), digest) == (TEXT_INVENTORY_COUNT, TEXT_INVENTORY_SHA256)
    # Spot shapes against the released geometry.
    layers = model.model.layers
    assert layers[2].self_attn.o_a_proj.weight.shape == (8 * 1024, 64 * 512 // 8)
    assert layers[2].self_attn.compressor.gate_proj is not None and layers[20].self_attn.compressor.gate_proj is None
    assert layers[2].self_attn.indexer.k_proj.weight.shape == (128, 512)
    assert layers[24].self_attn.indexer is not None and not hasattr(layers[24].self_attn.indexer, "k_proj")
    assert layers[1].engram.kv_proj.weight.shape == (5120 * 5, 3 * 8 * 256)
    assert model.model.engram_tables["14"].weight.shape == (384016682, 256)
    assert dspark.layers[0].main_proj.weight.shape == (5120, 3 * 5120)
    assert dspark.layers[0].mlp.experts.gate_up_proj.shape == (128, 2 * 2304, 5120)
    assert source_layer_key(3, "attn_hc.fn") == "layers.3.hc_attn_fn"
