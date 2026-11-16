# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Device-resident Llama prefill/decode over out-axis block-float checkpoints.

CPU tests cover the checkpoint reader, the bf8 embedding row gather, the KV
block pool and the geometry gate.  The hardware test (``GPTQMODEL_TEST_TTNN=1``)
runs a 2-layer Llama with Llama-3.2-1B geometry on one chip.
"""

from __future__ import annotations

import os

import pytest
import torch

from gptqmodel.utils import tt_blockfloat_quant as bq


transformers = pytest.importorskip("transformers")

_BITS = {"*": 4, "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8, "down_proj": 8}


def _config(**overrides):
    base = {
        "vocab_size": 512, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "tie_word_embeddings": True,
        "max_position_embeddings": 8192,
    }
    base.update(overrides)
    return transformers.LlamaConfig(**base)


def _checkpoint(tmp_path, config, *, block_axis="out", seed=0):
    torch.manual_seed(seed)
    model = transformers.LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and ".layers." in f".{name}":
                width = _BITS.get(name.rsplit(".", 1)[-1], 4)
                w = module.weight.detach()
                if block_axis == "out":
                    module.weight.copy_(bq.quantize_blockfloat(w.T.contiguous(), width)[0].T)
                else:
                    module.weight.copy_(bq.quantize_blockfloat(w, width)[0])
        e = model.model.embed_tokens.weight.detach()
        if block_axis == "out":
            model.model.embed_tokens.weight.copy_(bq.quantize_blockfloat(e.T.contiguous(), 8)[0].T)
        else:
            model.model.embed_tokens.weight.copy_(bq.quantize_blockfloat(e, 8)[0])
    path = tmp_path / f"ckpt-{block_axis}"
    bq.save_blockfloat_model(model, str(path), bits=_BITS, embed_bits=8, block_axis=block_axis)
    return str(path), model


def test_reader_rejects_in_axis_checkpoint(tmp_path):
    from gptqmodel.utils.tt_llama import load_tt_llama_weights

    path, _ = _checkpoint(tmp_path, _config(), block_axis="in")
    with pytest.raises(ValueError, match="block_axis='out'"):
        load_tt_llama_weights(path)


def test_reader_keeps_codes_and_expands_exactly(tmp_path):
    from gptqmodel.utils.tt_llama import PackedBlockfloat, load_tt_llama_weights

    path, model = _checkpoint(tmp_path, _config())
    _cfg, weights = load_tt_llama_weights(path)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".layers." in f".{name}":
            packed = weights[name]
            # The reader holds packed storage only; nothing is expanded at load.
            assert isinstance(packed, PackedBlockfloat) and packed.codes.dtype == torch.uint8
            assert packed.bits == (4 if name.endswith(("gate_proj", "up_proj")) else 8)
            assert torch.equal(packed.expand(), module.weight.detach().T), name
    assert torch.equal(weights["embed"].expand(), model.model.embed_tokens.weight.detach().T)
    assert torch.equal(weights["model.norm.weight"], model.model.norm.weight.detach())


def test_embedding_row_gather_matches_independent_oracle(tmp_path):
    from gptqmodel.utils.tt_llama import load_tt_llama_weights

    path, model = _checkpoint(tmp_path, _config())
    _cfg, weights = load_tt_llama_weights(path)
    table = model.model.embed_tokens.weight.detach()
    ids = torch.tensor([0, 1, 15, 16, 17, 255, 511, 3, 3])
    assert torch.equal(weights["embed"].expand_columns(ids), table[ids])
    gather = weights["embed"].column_major()
    assert gather.codes.dtype == torch.uint8 and gather.exponents.dtype == torch.int8
    assert torch.equal(gather(ids), table[ids])


def test_kv_block_pool_reserve_release():
    from gptqmodel.utils.tt_llama import KVBlockPool

    pool = KVBlockPool(num_blocks=6, block_size=64, users=4, max_blocks_per_user=3)
    pool.reserve(0, 100)          # 2 blocks
    pool.reserve(1, 64)           # 1 block
    assert pool.table[0, :2].tolist() == [1, 2] and pool.table[1, 0].item() == 3
    assert 0 not in pool.owned[0] + pool.owned[1]  # block 0 is scratch
    pool.reserve(0, 120)          # already covered, no new block
    assert len(pool.owned[0]) == 2
    with pytest.raises(ValueError, match="per-user context"):
        pool.reserve(2, 64 * 3 + 1)
    pool.reserve(2, 128)
    with pytest.raises(MemoryError):
        pool.reserve(3, 64)
    pool.release(0)
    assert pool.table[0].tolist() == [0, 0, 0]
    pool.reserve(3, 64)
    assert pool.owned[3]


@pytest.mark.parametrize(
    "max_context, kv_tokens, expect",
    [
        (100, 356, (8, 9)),        # short generate: pool grown to the 8-wide table + scratch
        (4096, 8192, (64, 129)),   # already large enough
        (65, 64, (8, 9)),          # 2 blocks of context still rounds the row to 8
        (16384 + 128, 32 * 16512, (264, 8257)),
    ],
)
def test_kv_pool_geometry_satisfies_device_ops(max_context, kv_tokens, expect):
    from gptqmodel.utils.tt_llama import kv_pool_geometry

    per_user, blocks = kv_pool_geometry(max_context, kv_tokens, 64)
    assert (per_user, blocks) == expect
    assert per_user % 8 == 0                 # chunked SDPA: 32-byte page-table rows
    assert per_user <= blocks                # paged_update_cache: table width <= pool
    assert per_user * 64 >= max_context and (blocks - 1) * 64 >= kv_tokens


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"tie_word_embeddings": False}, "tied"),
        ({"attention_bias": True}, "bias"),
        ({"hidden_size": 96, "num_attention_heads": 2}, "head_dim"),
        ({"num_attention_heads": 64, "hidden_size": 4096, "num_key_value_heads": 8}, "32"),
    ],
)
def test_geometry_gate(overrides, match):
    from gptqmodel.utils.tt_llama import validate_llama_config

    with pytest.raises(NotImplementedError, match=match):
        validate_llama_config(_config(**overrides))


def test_open_mesh_forwards_trace_region_and_device_id(monkeypatch):
    from gptqmodel.utils import ttnn_runtime

    calls = {}

    class _Fake:
        class FabricConfig:
            DISABLED = "DISABLED"

        @staticmethod
        def open_device(**kwargs):
            calls.update(kwargs)
            return object()

        @staticmethod
        def close_device(_device):
            calls["closed"] = True

        @staticmethod
        def set_fabric_config(_cfg):
            return None

    monkeypatch.setattr(ttnn_runtime, "import_ttnn", lambda: _Fake)
    monkeypatch.setattr(ttnn_runtime, "_reject_pjrt_owner", lambda: None)
    monkeypatch.setattr(ttnn_runtime, "before_device_open", lambda **_: None)
    monkeypatch.setattr(ttnn_runtime, "after_device_close", lambda **_: None)
    monkeypatch.setattr(ttnn_runtime, "park_idle_clock", lambda: None)
    monkeypatch.setattr(ttnn_runtime, "_MESH", None)
    ttnn_runtime.open_mesh(trace_region_size=1234, device_id=2, owner="t")
    try:
        assert calls == {"device_id": 2, "trace_region_size": 1234}
    finally:
        ttnn_runtime.close_mesh(owner="t")
    with pytest.raises(ValueError, match="device_id"):
        ttnn_runtime.open_mesh(device_count=2, device_id=1, owner="t")


def _no_dense_weight_retained(model, config):
    """No host tensor of weight size other than packed codes/exponents survives."""

    import ttnn

    smallest_weight = min(config.hidden_size * config.intermediate_size, config.hidden_size * config.vocab_size)
    for name, value in vars(model).items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert value.numel() < smallest_weight, f"dense host tensor {name} {tuple(value.shape)}"
    embed = model.embed
    assert embed.codes.dtype == torch.uint8 and embed.exponents.dtype == torch.int8
    assert embed.codes.numel() == config.vocab_size * config.hidden_size  # one byte per bf8 value
    blockfloat = {ttnn.bfloat4_b, ttnn.bfloat8_b}
    assert model.head.dtype == ttnn.bfloat8_b
    for layer in model.layers:
        for key in ("qkv", "o", "gate", "up", "down"):
            assert layer[key].dtype in blockfloat, key


@pytest.mark.skipif(os.environ.get("GPTQMODEL_TEST_TTNN") != "1", reason="needs a Tenstorrent chip")
def test_prefill_and_decode_match_cpu_on_device(tmp_path):
    from gptqmodel.utils.tt_llama import TTLlama, load_tt_llama_weights
    from gptqmodel.utils.ttnn_runtime import close_mesh, open_mesh

    # Llama-3.2-1B geometry, two layers, reduced vocab.
    config = _config(vocab_size=4096, hidden_size=2048, intermediate_size=8192, num_attention_heads=32,
                     num_key_value_heads=8, rope_theta=500000.0,
                     rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                                   "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
    path, _ = _checkpoint(tmp_path, config)
    reference, _, _ = bq.build_blockfloat_model(path, dtype=torch.float32)
    torch.manual_seed(1)
    # 2300 tokens: two prefill chunks of 2048, the second padded.
    prompts = [torch.randint(0, config.vocab_size, (n,)).tolist() for n in (2300, 77, 300)]
    new_tokens = 6
    with torch.no_grad():
        expected = [reference.generate(torch.tensor([p]), max_new_tokens=new_tokens, do_sample=False,
                                       output_logits=True, return_dict_in_generate=True) for p in prompts]
    cfg, weights = load_tt_llama_weights(path)
    mesh = open_mesh(device_count=1, trace_region_size=100_000_000, owner="test_tt_llama")
    try:
        model = TTLlama(mesh.device, cfg, weights, max_context=4096, kv_tokens=8192)
        del weights
        assert model.drift and max(model.drift.values()) == 0.0
        for user, (prompt, ref) in enumerate(zip(prompts, expected)):
            logits = model.prefill(user, prompt)
            cos = float(torch.nn.functional.cosine_similarity(logits, ref.logits[0][0].float(), dim=0))
            assert cos >= 0.998, (len(prompt), cos)
        # Batched decode (3 active rows) vs CPU greedy per prompt; tolerate one late near-tie.
        traced = model.generate(prompts, new_tokens, use_trace=True)
        for got, ref, prompt in zip(traced, expected, prompts):
            want = ref.sequences[0, len(prompt):].tolist()
            agree = sum(a == b for a, b in zip(got, want))
            assert agree >= new_tokens - 1 and got[0] == want[0], (got, want)
        # Eager equals traced; attending over 4 active rows equals all 32 rows for those slots.
        assert model.generate(prompts, new_tokens, use_trace=False) == traced
        _no_dense_weight_retained(model, cfg)
    finally:
        close_mesh(owner="test_tt_llama")
