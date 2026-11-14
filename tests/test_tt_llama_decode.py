# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Device-resident Llama decode over out-axis block-float checkpoints.

The CPU tests cover the checkpoint reader and geometry gate.  The hardware test
(``GPTQMODEL_TEST_TTNN=1``) runs a 2-layer Llama with Llama-3.2-1B geometry on
one chip and checks zero upload drift plus teacher-forced agreement with the CPU
model built from the same checkpoint.
"""

from __future__ import annotations

import os

import pytest
import torch

from gptqmodel.utils import tt_blockfloat_quant as bq


transformers = pytest.importorskip("transformers")


def _config(**overrides):
    base = {
        "vocab_size": 512, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "tie_word_embeddings": True,
        "max_position_embeddings": 1024,
    }
    base.update(overrides)
    return transformers.LlamaConfig(**base)


def _checkpoint(tmp_path, config, *, block_axis="out", seed=0):
    torch.manual_seed(seed)
    model = transformers.LlamaForCausalLM(config).eval()
    bits = {"*": 4, "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8, "down_proj": 8}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and ".layers." in f".{name}":
                width = bits.get(name.rsplit(".", 1)[-1], 4)
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
    bq.save_blockfloat_model(model, str(path), bits=bits, embed_bits=8, block_axis=block_axis)
    return str(path), model


def test_reader_rejects_in_axis_checkpoint(tmp_path):
    from gptqmodel.utils.tt_llama import load_tt_llama_weights

    path, _ = _checkpoint(tmp_path, _config(), block_axis="in")
    with pytest.raises(ValueError, match="block_axis='out'"):
        load_tt_llama_weights(path)


def test_reader_returns_exact_transposed_operands(tmp_path):
    from gptqmodel.utils.tt_llama import load_tt_llama_weights

    config = _config()
    path, model = _checkpoint(tmp_path, config)
    _cfg, weights = load_tt_llama_weights(path)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".layers." in f".{name}":
            stored, bits = weights[name]
            assert bits == (4 if name.endswith(("gate_proj", "up_proj")) else 8)
            assert torch.equal(stored, module.weight.detach().T), name
            assert bq.quantize_blockfloat(stored, bits)[0].equal(stored), name
    head, head_bits = weights["head"]
    embed, _ = weights["embed"]
    assert head_bits == 8
    assert torch.equal(head, model.model.embed_tokens.weight.detach().T)
    assert torch.equal(embed, model.model.embed_tokens.weight.detach())
    assert torch.equal(weights["model.norm.weight"][0], model.model.norm.weight.detach())


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


@pytest.mark.skipif(os.environ.get("GPTQMODEL_TEST_TTNN") != "1", reason="needs a Tenstorrent chip")
def test_decode_matches_cpu_on_device(tmp_path):
    from gptqmodel.utils.tt_llama import TTLlamaDecoder, load_tt_llama_weights
    from gptqmodel.utils.ttnn_runtime import close_mesh, open_mesh

    # Llama-3.2-1B geometry, two layers, reduced vocab.
    config = _config(vocab_size=4096, hidden_size=2048, intermediate_size=8192, num_attention_heads=32,
                     num_key_value_heads=8, rope_theta=500000.0,
                     rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                                   "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
    path, _ = _checkpoint(tmp_path, config)
    reference, _, _ = bq.build_blockfloat_model(path, dtype=torch.float32)
    steps = 8
    torch.manual_seed(1)
    seqs = torch.randint(0, config.vocab_size, (32, steps))
    with torch.no_grad():
        expected = reference(seqs).logits.float()
    cfg, weights = load_tt_llama_weights(path)
    mesh = open_mesh(device_count=1, trace_region_size=50_000_000, owner="test_tt_llama")
    try:
        decoder = TTLlamaDecoder(mesh.device, cfg, weights, max_seq=256)
        assert decoder.drift and max(decoder.drift.values()) == 0.0
        got = []
        for pos in range(steps):
            got.append(decoder.read_logits(decoder.step(*decoder.inputs(seqs[:, pos], [pos] * 32))))
        got = torch.stack(got, 1)
        top1 = float((got.argmax(-1) == expected.argmax(-1)).float().mean())
        cos = float(torch.nn.functional.cosine_similarity(got, expected, dim=-1).mean())
        assert top1 >= 0.95 and cos >= 0.998, (top1, cos)
        # Traced greedy generation agrees with eager step-by-step argmax.
        prompts = [seqs[i, :3].tolist() for i in range(4)]
        traced = decoder.generate(prompts, 4, use_trace=True)
        eager = decoder.generate(prompts, 4, use_trace=False)
        assert traced == eager
    finally:
        close_mesh(owner="test_tt_llama")
