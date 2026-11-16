# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``GPTQModel.export(..., "hf")`` must emit a loadable dense model.

The export routes through the checkpoint-level dequantizer so AWQ layouts,
quantized embeddings, and legacy ``quantize_config.json``-only checkpoints are
all covered. The synthetic cases below build the packed tensors from explicit
codes, zero points, and scales, then compare the exported dense weights against
an independent fp64 oracle.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from gptqmodel import GPTQModel
from gptqmodel.utils.model_dequant import pack_cols


_LOCAL_AWQ = "/monster/data/model/AWQ-Llama-3.2-1B-g128-gemm"


def _pack_rows(values, bits):
    """Pack logical rows into GPTQ ``qweight`` words (32-bit int32)."""

    values = values.to(torch.int32)
    rows, cols = values.shape
    if bits == 3:
        if rows % 32 != 0:
            raise ValueError("3-bit rows must be divisible by 32")
        blocks = rows // 32
        values_i64 = (values.to(torch.int64) & 0x7).reshape(blocks, 32, cols)
        packed = torch.zeros((blocks * 3, cols), dtype=torch.int32)
        for block in range(blocks):
            word0 = torch.zeros((cols,), dtype=torch.int64)
            for idx in range(10):
                word0 |= values_i64[block, idx, :] << (3 * idx)
            word0 |= values_i64[block, 10, :] << 30
            word1 = (values_i64[block, 10, :] >> 2) & 0x1
            for idx in range(10):
                word1 |= values_i64[block, 11 + idx, :] << (1 + 3 * idx)
            word1 |= values_i64[block, 21, :] << 31
            word2 = (values_i64[block, 21, :] >> 1) & 0x3
            for idx in range(10):
                word2 |= values_i64[block, 22 + idx, :] << (2 + 3 * idx)
            packed[block * 3 + 0, :] = word0.to(torch.int32)
            packed[block * 3 + 1, :] = word1.to(torch.int32)
            packed[block * 3 + 2, :] = word2.to(torch.int32)
        return packed

    pack_factor = 32 // bits
    if rows % pack_factor:
        raise ValueError("rows must be divisible by the pack factor")
    words = (values.to(torch.int64) & ((1 << bits) - 1)).reshape(rows // pack_factor, pack_factor, cols)
    shifts = (torch.arange(pack_factor, dtype=torch.int64) * bits).view(1, pack_factor, 1)
    return (words << shifts).sum(dim=1).to(torch.int32)


def _oracle_weight(codes, zeros, scales, group_size):
    in_features = codes.shape[0]
    groups = 1 if group_size <= 0 else in_features // group_size
    g_idx = torch.arange(in_features, dtype=torch.long) // (in_features if groups == 1 else group_size)
    return ((codes.to(torch.float64) - zeros.to(torch.float64)[g_idx]) * scales.to(torch.float64)[g_idx]).T


def _packed_module(prefix, bits, in_features, out_features, group_size, seed):
    generator = torch.Generator().manual_seed(seed)
    high = (1 << bits) - 1
    groups = 1 if group_size <= 0 else in_features // group_size
    codes = torch.randint(0, high + 1, (in_features, out_features), generator=generator, dtype=torch.int32)
    zeros = torch.randint(0, high + 1, (groups, out_features), generator=generator, dtype=torch.int32)
    scales = torch.rand((groups, out_features), generator=generator, dtype=torch.float32) + 0.01
    g_idx = (torch.arange(in_features, dtype=torch.long) // (in_features if groups == 1 else group_size)).to(torch.int32)
    tensors = {
        f"{prefix}.qweight": _pack_rows(codes, bits),
        f"{prefix}.qzeros": pack_cols(zeros, bits, pack_dtype=torch.int32),
        f"{prefix}.scales": scales,
        f"{prefix}.g_idx": g_idx,
    }
    return tensors, _oracle_weight(codes, zeros, scales, group_size)


def _tiny_llama_checkpoint(path: Path, *, bits: int, lm_head_bits: int, legacy_config: bool):
    hidden = 32
    vocab = 64
    intermediate = 64
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": hidden // 2,
        "vocab_size": vocab,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-5,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "dtype": "float32",
    }
    quantize_config = {
        "bits": bits,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "quant_method": "gptq",
        "checkpoint_format": "gptq_v2",
        "dynamic": {"lm_head": {"bits": lm_head_bits}},
    }
    if legacy_config:
        (path / "quantize_config.json").write_text(json.dumps(quantize_config), encoding="utf-8")
    else:
        config["quantization_config"] = quantize_config

    embed, embed_expected = _packed_module("model.embed_tokens", bits, hidden, vocab, 32, 1)
    q_proj, q_proj_expected = _packed_module(
        "model.layers.0.self_attn.q_proj", bits, hidden, hidden, 32, 2
    )
    lm_head, lm_head_expected = _packed_module("lm_head", lm_head_bits, hidden, vocab, 32, 3)

    generator = torch.Generator().manual_seed(9)
    def dense(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02

    tensors = {
        **embed,
        **q_proj,
        **lm_head,
        "model.layers.0.input_layernorm.weight": torch.ones(hidden),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden),
        "model.layers.0.self_attn.k_proj.weight": dense((hidden, hidden)),
        "model.layers.0.self_attn.v_proj.weight": dense((hidden, hidden)),
        "model.layers.0.self_attn.o_proj.weight": dense((hidden, hidden)),
        "model.layers.0.mlp.gate_proj.weight": dense((intermediate, hidden)),
        "model.layers.0.mlp.up_proj.weight": dense((intermediate, hidden)),
        "model.layers.0.mlp.down_proj.weight": dense((hidden, intermediate)),
        "model.norm.weight": torch.ones(hidden),
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path / "model.safetensors"))
    return {
        "model.embed_tokens.weight": embed_expected,
        "model.layers.0.self_attn.q_proj.weight": q_proj_expected,
        "lm_head.weight": lm_head_expected,
    }


@pytest.mark.parametrize("bits,lm_head_bits", [(3, 4), (4, 8)])
def test_export_hf_quantized_embeddings_and_dynamic_bits(tmp_path, bits, lm_head_bits):
    source = tmp_path / "source"
    source.mkdir()
    expected = _tiny_llama_checkpoint(
        source, bits=bits, lm_head_bits=lm_head_bits, legacy_config=False
    )

    out = tmp_path / "out"
    GPTQModel.export(str(source), str(out), "hf")

    saved_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in saved_config

    model = AutoModelForCausalLM.from_pretrained(str(out))
    logits = model(torch.tensor([[1, 2, 3, 4]])).logits
    assert torch.isfinite(logits).all()

    from safetensors import safe_open

    with safe_open(str(out / "model.safetensors"), framework="pt") as reader:
        for key, oracle in expected.items():
            # Export writes the checkpoint dtype (float32 for this fixture);
            # keep the tolerance loose enough for half-precision checkpoints.
            torch.testing.assert_close(
                reader.get_tensor(key).to(torch.float64), oracle, rtol=1e-2, atol=2e-2
            )


def test_export_hf_accepts_legacy_quantize_config_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    expected = _tiny_llama_checkpoint(
        source, bits=4, lm_head_bits=4, legacy_config=True
    )

    out = tmp_path / "out"
    GPTQModel.export(str(source), str(out), "hf")

    assert (out / "model.safetensors").is_file()
    model = AutoModelForCausalLM.from_pretrained(str(out))
    assert torch.isfinite(model(torch.tensor([[1, 2, 3]])).logits).all()

    from safetensors import safe_open

    with safe_open(str(out / "model.safetensors"), framework="pt") as reader:
        for key, oracle in expected.items():
            torch.testing.assert_close(
                reader.get_tensor(key).to(torch.float64), oracle, rtol=1e-2, atol=2e-2
            )


@pytest.mark.skipif(not Path(_LOCAL_AWQ).is_dir(), reason="local AWQ checkpoint not present")
def test_export_hf_awq_checkpoint_reloads(tmp_path):
    out = tmp_path / "out"
    GPTQModel.export(_LOCAL_AWQ, str(out), "hf")

    saved_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in saved_config

    model = AutoModelForCausalLM.from_pretrained(str(out))
    logits = model(torch.tensor([[1, 2, 3]])).logits
    assert logits.shape[-1] == model.config.vocab_size
    assert torch.isfinite(logits).all()
