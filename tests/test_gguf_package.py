# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Packing a GPT-QModel GGUF checkpoint into one llama.cpp file.

The writer is a container change, not a conversion: every packed row must come
out byte-identical to its source row, so these tests compare bytes instead of
error statistics. Refusals are part of the contract too: a layout the writer
cannot prove byte-exact must fail before a file is published.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from gptqmodel.packaging import (
    CheckpointFormatError,
    GGUFPackageConfig,
    PackageConfig,
    UnsafeTransformError,
    UnsupportedArchitectureError,
    UnsupportedTensorError,
    package,
)
from gptqmodel.packaging.arch import resolve_spec
from gptqmodel.packaging.errors import MissingDependencyError
from gptqmodel.packaging.tokenizer import (
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    TOKEN_TYPE_UNUSED,
    TOKEN_TYPE_USER_DEFINED,
    build_tokenizer_metadata,
)
from gptqmodel.quantization.config import GGUFConfig


gguf = pytest.importorskip("gguf")

HIDDEN = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
INTERMEDIATE = 512
LAYERS = 1
Q_DIM = HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM
#: ``a``..``h`` plus the result of the single merge below.
VOCAB = 9

REAL_TOKENIZER_DIR = Path("/monster/data/model/Qwen3-0.6B")
ARCH_ENUM = {
    "qwen2": gguf.MODEL_ARCH.QWEN2,
    "qwen3": gguf.MODEL_ARCH.QWEN3,
    "llama": gguf.MODEL_ARCH.LLAMA,
}


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {chr(ord("a") + index): index for index in range(VOCAB - 1)}
    vocab["ab"] = VOCAB - 1
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")]))
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="a", eos_token="b")


def _packed(shape, qtype):
    from gptqmodel.nn_modules.qlinear.gguf import _quantize_gguf_tensor_numpy

    seed = (int(np.prod(shape)) + sum(ord(char) for char in qtype)) % (2**32)
    weight = (np.random.default_rng(seed).standard_normal(shape) * 0.02).astype(np.float32)
    return torch.from_numpy(_quantize_gguf_tensor_numpy(weight, qtype))


def _layer(index, leaf):
    return f"model.layers.{index}.{leaf}"


def _build_checkpoint(
    root,
    *,
    model_type="qwen3",
    vocab_size=VOCAB,
    dynamic=None,
    tied=True,
    include_lm_head=False,
    extra=None,
    embed_dtype=torch.float16,
    quantize_config_only=False,
    config_overrides=None,
    attention_bias=False,
):
    from safetensors.torch import save_file

    qcfg = GGUFConfig(bits="q4_k_m", dynamic=dynamic or {})
    tensors = {}

    def put(name, tensor):
        tensors[name] = tensor.contiguous()

    down_qtype = "Q6_K" if dynamic else "Q4_K"
    for leaf, shape in (
        ("self_attn.q_proj", (Q_DIM, HIDDEN)),
        ("self_attn.k_proj", (KV_DIM, HIDDEN)),
        ("self_attn.v_proj", (KV_DIM, HIDDEN)),
        ("self_attn.o_proj", (HIDDEN, Q_DIM)),
        ("mlp.gate_proj", (INTERMEDIATE, HIDDEN)),
        ("mlp.up_proj", (INTERMEDIATE, HIDDEN)),
    ):
        put(_layer(0, leaf + ".qweight"), _packed(shape, "Q4_K"))
    put(_layer(0, "mlp.down_proj.qweight"), _packed((HIDDEN, INTERMEDIATE), down_qtype))

    embed = torch.randn(vocab_size, HIDDEN) * 0.02
    put("model.embed_tokens.weight", embed.to(embed_dtype))
    put("model.norm.weight", torch.ones(HIDDEN))
    put(_layer(0, "input_layernorm.weight"), torch.ones(HIDDEN))
    put(_layer(0, "post_attention_layernorm.weight"), torch.ones(HIDDEN))
    if model_type != "llama":
        put(_layer(0, "self_attn.q_norm.weight"), torch.ones(HEAD_DIM))
        put(_layer(0, "self_attn.k_norm.weight"), torch.ones(HEAD_DIM))
    if include_lm_head:
        put("lm_head.qweight", _packed((vocab_size, HIDDEN), "Q4_K"))
    if attention_bias:
        put(_layer(0, "self_attn.q_proj.bias"), torch.randn(Q_DIM) * 0.01)
        put(_layer(0, "self_attn.k_proj.bias"), torch.randn(KV_DIM) * 0.01)
        put(_layer(0, "self_attn.v_proj.bias"), torch.randn(KV_DIM) * 0.01)
    for name, tensor in (extra or {}).items():
        put(name, tensor)

    config = {
        "architectures": [f"{model_type.capitalize()}ForCausalLM"],
        "model_type": model_type,
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": vocab_size,
        "tie_word_embeddings": tied,
    }
    config.update(config_overrides or {})
    if not quantize_config_only:
        config["quantization_config"] = qcfg.to_dict()
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if quantize_config_only:
        (root / "quantize_config.json").write_text(json.dumps(qcfg.to_dict(), indent=2), encoding="utf-8")
    save_file(tensors, str(root / "model.safetensors"))
    return qcfg, tensors


def _run_package(root, out_dir, *, out_name="model.gguf", tokenizer=None, **kwargs):
    return package(
        GGUFPackageConfig(
            model_path=str(root),
            out_path=str(out_dir / out_name),
            tokenizer=tokenizer if tokenizer is not None else _tiny_tokenizer(),
            **kwargs,
        )
    )


def _tensors(path):
    return {tensor.name: tensor for tensor in gguf.GGUFReader(str(path)).tensors}


def _name_map(model_type):
    return gguf.get_tensor_name_map(ARCH_ENUM[model_type], LAYERS)


def test_packed_rows_are_copied_byte_for_byte(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root, dynamic={".*down_proj": {"bits": "q6_k"}})
    out = _run_package(root, tmp_path)

    written = _tensors(out)
    name_map = _name_map("qwen3")
    packed = {name: value for name, value in tensors.items() if name.endswith(".qweight")}
    assert packed
    for hf_name, source in packed.items():
        gguf_name = name_map.get_name(hf_name[: -len(".qweight")] + ".weight", try_suffixes=(".weight",))
        assert gguf_name is not None, hf_name
        assert gguf_name in written, gguf_name
        expected_qtype = "Q6_K" if "down_proj" in hf_name else "Q4_K"
        assert written[gguf_name].tensor_type.name == expected_qtype
        np.testing.assert_array_equal(np.array(written[gguf_name].data), source.numpy())


def test_dense_tensors_keep_their_declared_width(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root)
    out = _run_package(root, tmp_path)

    written = _tensors(out)
    assert written["token_embd.weight"].tensor_type.name == "F16"
    assert written["output_norm.weight"].tensor_type.name == "F32"
    assert written["blk.0.attn_norm.weight"].tensor_type.name == "F32"
    assert written["blk.0.attn_q_norm.weight"].tensor_type.name == "F32"
    np.testing.assert_array_equal(
        np.array(written["token_embd.weight"].data),
        tensors["model.embed_tokens.weight"].numpy(),
    )


def test_bf16_embedding_is_stored_as_bf16(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root, embed_dtype=torch.bfloat16)
    out = _run_package(root, tmp_path)

    written = _tensors(out)
    assert written["token_embd.weight"].tensor_type.name == "BF16"
    source = tensors["model.embed_tokens.weight"]
    np.testing.assert_array_equal(
        np.array(written["token_embd.weight"].data),
        source.view(torch.int16).numpy().view(np.uint8),
    )


def test_metadata_and_tokenizer_arrays(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out = _run_package(root, tmp_path)

    reader = gguf.GGUFReader(str(out))
    assert reader.get_field("general.architecture").contents() == "qwen3"
    assert reader.get_field("qwen3.block_count").contents() == LAYERS
    assert reader.get_field("qwen3.embedding_length").contents() == HIDDEN
    assert reader.get_field("qwen3.attention.head_count").contents() == HEADS
    assert reader.get_field("qwen3.attention.head_count_kv").contents() == KV_HEADS
    assert reader.get_field("qwen3.vocab_size").contents() == VOCAB
    assert reader.get_field("qwen3.attention.key_length").contents() == HEAD_DIM
    assert reader.get_field("tokenizer.ggml.model").contents() == "gpt2"
    assert reader.get_field("tokenizer.ggml.pre").contents() == "qwen2"
    assert len(reader.get_field("tokenizer.ggml.tokens").contents()) == VOCAB
    assert reader.get_field("tokenizer.ggml.bos_token_id").contents() == 0
    assert reader.get_field("tokenizer.ggml.eos_token_id").contents() == 1


def test_native_reader_agrees_with_the_oracle(tmp_path):
    from gptqmodel.utils.internal_gguf import GGUFReader as NativeReader

    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out = _run_package(root, tmp_path)

    native = {tensor.name: tensor for tensor in NativeReader(str(out)).tensors}
    oracle = _tensors(out)
    assert set(native) == set(oracle)
    assert native["blk.0.attn_q.weight"].tensor_type.name == "Q4_K"
    # GGUF stores the logical [out, in] matrix with its dimensions reversed.
    assert tuple(int(value) for value in native["blk.0.attn_q.weight"].shape.tolist()) == (HIDDEN, Q_DIM)


def test_llama_permutes_qk_rows_and_keeps_every_row(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root, model_type="llama")
    out = _run_package(root, tmp_path)

    written = _tensors(out)
    name_map = _name_map("llama")

    def permutation(rows, heads):
        dim = rows // heads // 2
        return np.arange(rows).reshape(heads, 2, dim).swapaxes(1, 2).reshape(-1)

    for module, heads, rows in (
        ("model.layers.0.self_attn.q_proj", HEADS, Q_DIM),
        ("model.layers.0.self_attn.k_proj", KV_HEADS, KV_DIM),
    ):
        gguf_name = name_map.get_name(module + ".weight", try_suffixes=(".weight",))
        assert gguf_name is not None
        source = tensors[module + ".qweight"].numpy()
        packed = np.array(written[gguf_name].data)
        np.testing.assert_array_equal(packed, source[permutation(rows, heads)])
        assert {row.tobytes() for row in packed} == {row.tobytes() for row in source}


def test_llama_permutes_qk_biases_too(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root, model_type="llama", attention_bias=True)
    out = _run_package(root, tmp_path)

    written = _tensors(out)
    name_map = _name_map("llama")

    def permutation(rows, heads):
        dim = rows // heads // 2
        return np.arange(rows).reshape(heads, 2, dim).swapaxes(1, 2).reshape(-1)

    for module, heads, rows in (
        ("model.layers.0.self_attn.q_proj", HEADS, Q_DIM),
        ("model.layers.0.self_attn.k_proj", KV_HEADS, KV_DIM),
    ):
        gguf_name = name_map.get_name(module + ".bias", try_suffixes=(".bias",))
        assert gguf_name is not None
        assert written[gguf_name].tensor_type.name == "F32"
        np.testing.assert_array_equal(
            np.array(written[gguf_name].data),
            tensors[module + ".bias"].numpy()[permutation(rows, heads)],
        )

    value_name = name_map.get_name("model.layers.0.self_attn.v_proj.bias", try_suffixes=(".bias",))
    assert value_name is not None
    np.testing.assert_array_equal(
        np.array(written[value_name].data),
        tensors["model.layers.0.self_attn.v_proj.bias"].numpy(),
    )


def test_llama3_rope_freqs_matches_the_piecewise_spec(tmp_path):
    scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(
        root,
        model_type="llama",
        config_overrides={"rope_theta": 10000.0, "rope_scaling": scaling},
    )
    out = _run_package(root, tmp_path)

    written = _tensors(out)
    assert "rope_freqs.weight" in written
    assert written["rope_freqs.weight"].tensor_type.name == "F32"
    assert written["rope_freqs.weight"].data.shape == (HEAD_DIM // 2,)

    base = 10000.0
    factor = 8.0
    low_wavelen = 8192 / 1.0
    high_wavelen = 8192 / 4.0
    expected = []
    for freq in 1.0 / (base ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM)):
        wavelen = 2 * math.pi / freq
        if wavelen < high_wavelen:
            expected.append(1.0)
        elif wavelen > low_wavelen:
            expected.append(factor)
        else:
            smooth = (8192 / wavelen - 1.0) / (4.0 - 1.0)
            expected.append(1.0 / ((1.0 - smooth) / factor + smooth))
    np.testing.assert_allclose(
        np.array(written["rope_freqs.weight"].data),
        np.asarray(expected, dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )




def test_untied_checkpoint_writes_the_output_projection(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root, tied=False, include_lm_head=True)
    out = _run_package(root, tmp_path)

    written = _tensors(out)
    assert "output.weight" in written
    assert written["output.weight"].tensor_type.name == "Q4_K"
    np.testing.assert_array_equal(
        np.array(written["output.weight"].data), tensors["lm_head.qweight"].numpy()
    )


def test_quantize_config_json_is_accepted(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, quantize_config_only=True)
    out = _run_package(root, tmp_path)
    assert _tensors(out)["blk.0.ffn_down.weight"].tensor_type.name == "Q4_K"


def test_existing_output_requires_overwrite(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    first = _run_package(root, tmp_path)
    with pytest.raises(FileExistsError):
        _run_package(root, tmp_path)
    second = _run_package(root, tmp_path, overwrite=True)
    assert first == second


def test_refuses_a_shape_the_arch_layout_does_not_allow(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["head_dim"] = 32
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(UnsafeTransformError, match=r"blk\.0\.attn_"):
        _run_package(root, out_dir)
    assert list(out_dir.iterdir()) == []


def test_refuses_an_unmapped_tensor(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, extra={"model.layers.0.self_attn.weird.weight": torch.ones(4, 4)})
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(UnsupportedTensorError, match="weird"):
        _run_package(root, out_dir)
    assert list(out_dir.iterdir()) == []


def test_refuses_mixture_of_experts(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, config_overrides={"num_experts": 8})
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(UnsupportedArchitectureError, match="mixture-of-experts"):
        _run_package(root, out_dir)


def test_refuses_a_plain_gguf_quant_method(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"]["quant_method"] = "gguf"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(CheckpointFormatError, match="quant_method"):
        _run_package(root, out_dir)


def test_refuses_untied_checkpoint_without_lm_head(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, tied=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(CheckpointFormatError, match="tie_word_embeddings"):
        _run_package(root, out_dir)


def test_unregistered_architecture_fails_closed(tmp_path):
    with pytest.raises(UnsupportedArchitectureError):
        resolve_spec("gemma3", False)
    assert resolve_spec("gemma3", True).verified is False

    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(UnsupportedArchitectureError, match="no verified GGUF layout"):
        _run_package(root, out_dir, architecture="gemma3")
    assert list(out_dir.iterdir()) == []


def test_failed_write_publishes_nothing(tmp_path, monkeypatch):
    import gptqmodel.packaging.gguf_package as module

    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def boom(path, plan):
        raise RuntimeError("injected verification failure")

    monkeypatch.setattr(module, "_verify_written", boom)
    with pytest.raises(RuntimeError, match="injected"):
        _run_package(root, out_dir)
    assert list(out_dir.iterdir()) == []


def test_missing_gguf_dependency_is_reported(tmp_path, monkeypatch):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(MissingDependencyError):
        _run_package(root, out_dir)


def test_base_config_has_no_writer(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    with pytest.raises(TypeError):
        package(PackageConfig(model_path=str(root), out_path=str(tmp_path / "x.gguf")))
    with pytest.raises(TypeError):
        package("not-a-config")


def test_output_directory_is_rejected(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    with pytest.raises(IsADirectoryError):
        GGUFPackageConfig(model_path=str(root), out_path=str(tmp_path))


def test_tokenizer_metadata_fills_rows_past_the_vocabulary():
    meta = build_tokenizer_metadata(_tiny_tokenizer(), vocab_size=VOCAB + 4, tokenizer_pre="qwen2")
    assert len(meta.tokens) == VOCAB + 4
    assert meta.token_types[-4:] == (TOKEN_TYPE_UNUSED,) * 4
    assert meta.tokens[-1] == f"[PAD{VOCAB + 3}]"


def test_token_type_constants_match_the_gguf_contract():
    expected = tuple(
        int(member)
        for member in (
            gguf.TokenType.NORMAL,
            gguf.TokenType.UNKNOWN,
            gguf.TokenType.CONTROL,
            gguf.TokenType.USER_DEFINED,
            gguf.TokenType.UNUSED,
            gguf.TokenType.BYTE,
        )
    )
    assert (
        TOKEN_TYPE_NORMAL,
        TOKEN_TYPE_UNKNOWN,
        TOKEN_TYPE_CONTROL,
        TOKEN_TYPE_USER_DEFINED,
        TOKEN_TYPE_UNUSED,
        TOKEN_TYPE_BYTE,
    ) == expected


@pytest.mark.skipif(not REAL_TOKENIZER_DIR.is_dir(), reason="local Qwen3 tokenizer fixture is unavailable")
def test_real_qwen3_tokenizer_covers_added_tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(REAL_TOKENIZER_DIR), local_files_only=True)
    meta = build_tokenizer_metadata(tokenizer, vocab_size=151936, tokenizer_pre="qwen2")
    assert len(meta.tokens) == 151936
    assert meta.tokens[151643] == "<|endoftext|>"
    assert meta.token_types[151643] == TOKEN_TYPE_CONTROL
    assert meta.tokens[151668] == "</think>"
    assert meta.token_types[151668] == TOKEN_TYPE_USER_DEFINED
    assert len(meta.merges) == 151387
    assert meta.special_ids["eos"] == 151645


@pytest.mark.skipif(not REAL_TOKENIZER_DIR.is_dir(), reason="local Qwen3 tokenizer fixture is unavailable")
def test_llamacpp_loads_the_packaged_file(tmp_path):
    llama_cpp = pytest.importorskip("llama_cpp")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(REAL_TOKENIZER_DIR), local_files_only=True)
    vocab_size = 151936
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, vocab_size=vocab_size)
    out = _run_package(root, tmp_path, tokenizer=tokenizer)

    model = llama_cpp.Llama(model_path=str(out), n_ctx=32, n_gpu_layers=0, verbose=False)
    assert model.n_vocab() == vocab_size
    for text in ("Hello, world! 你好", "def f(x):\n    return x + 1\n"):
        assert model.tokenize(text.encode("utf-8")) == tokenizer.encode(text, add_special_tokens=False)
