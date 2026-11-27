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

import numpy as np
import pytest
import torch

from gptqmodel.packaging import (
    CheckpointFormatError,
    GGUFPackageConfig,
    PackageConfig,
    PackageError,
    UnsafeTransformError,
    UnsupportedArchitectureError,
    UnsupportedQuantTypeError,
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
    omit_config_keys=(),
    attention_bias=False,
    mlp_bias=False,
    heads=HEADS,
):
    from safetensors.torch import save_file

    qcfg = GGUFConfig(bits="q4_k_m", dynamic=dynamic or {})
    tensors = {}

    def put(name, tensor):
        tensors[name] = tensor.contiguous()

    query_width = heads * HEAD_DIM
    down_qtype = "Q6_K" if dynamic else "Q4_K"
    for leaf, shape in (
        ("self_attn.q_proj", (query_width, HIDDEN)),
        ("self_attn.k_proj", (KV_DIM, HIDDEN)),
        ("self_attn.v_proj", (KV_DIM, HIDDEN)),
        ("self_attn.o_proj", (HIDDEN, query_width)),
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
    if model_type == "qwen3":
        put(_layer(0, "self_attn.q_norm.weight"), torch.ones(HEAD_DIM))
        put(_layer(0, "self_attn.k_norm.weight"), torch.ones(HEAD_DIM))
    if include_lm_head:
        put("lm_head.qweight", _packed((vocab_size, HIDDEN), "Q4_K"))
    if attention_bias or model_type == "qwen2":
        put(_layer(0, "self_attn.q_proj.bias"), torch.randn(query_width) * 0.01)
        put(_layer(0, "self_attn.k_proj.bias"), torch.randn(KV_DIM) * 0.01)
        put(_layer(0, "self_attn.v_proj.bias"), torch.randn(KV_DIM) * 0.01)
    if attention_bias:
        put(_layer(0, "self_attn.o_proj.bias"), torch.randn(HIDDEN) * 0.01)
    if mlp_bias:
        for proj, width in (("gate", INTERMEDIATE), ("up", INTERMEDIATE), ("down", HIDDEN)):
            put(_layer(0, f"mlp.{proj}_proj.bias"), torch.randn(width) * 0.01)
    for name, tensor in (extra or {}).items():
        put(name, tensor)

    config = {
        "architectures": [f"{model_type.capitalize()}ForCausalLM"],
        "model_type": model_type,
        "hidden_size": HIDDEN,
        "num_attention_heads": heads,
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
    if attention_bias:
        config["attention_bias"] = True
    if mlp_bias:
        config["mlp_bias"] = True
    config.update(config_overrides or {})
    for key in omit_config_keys:
        del config[key]
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
    assert reader.get_field("qwen3.context_length").contents() == 128
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


def test_loads_the_tokenizer_from_the_checkpoint_directory(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    _tiny_tokenizer().save_pretrained(root)

    out = package(GGUFPackageConfig(model_path=str(root), out_path=str(tmp_path / "model.gguf")))

    reader = gguf.GGUFReader(str(out))
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


@pytest.mark.parametrize("rope_key", ["rope_parameters", "rope_scaling"])
def test_llama3_rope_freqs_matches_the_piecewise_spec(tmp_path, rope_key):
    scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    root = tmp_path / "ckpt"
    root.mkdir()
    overrides = {"rope_theta": 10000.0, rope_key: scaling}
    if rope_key == "rope_parameters":
        scaling["rope_theta"] = 10000.0
        overrides["rope_theta"] = 1000000.0
    _build_checkpoint(root, model_type="llama", config_overrides=overrides)
    out = _run_package(root, tmp_path)

    reader = gguf.GGUFReader(str(out))
    assert reader.get_field("llama.rope.freq_base").contents() == 10000.0
    written = {tensor.name: tensor for tensor in reader.tensors}
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


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama"])
def test_current_rope_parameters_set_frequency_base(tmp_path, model_type):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(
        root,
        model_type=model_type,
        config_overrides={"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}},
    )

    reader = gguf.GGUFReader(str(_run_package(root, tmp_path)))
    assert reader.get_field(f"{model_type}.rope.freq_base").contents() == 500000.0


def test_linear_rope_scaling_is_preserved(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(
        root,
        config_overrides={"rope_parameters": {"rope_type": "linear", "rope_theta": 500000.0, "factor": 4.0}},
    )

    reader = gguf.GGUFReader(str(_run_package(root, tmp_path)))
    scaling_type = reader.get_field("qwen3.rope.scaling.type")
    assert scaling_type is not None and scaling_type.contents() == "linear"
    assert reader.get_field("qwen3.rope.scaling.factor").contents() == 4.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rope_parameters": {"rope_type": "dynamic", "factor": 2.0}}, "RoPE"),
        ({"use_sliding_window": True, "sliding_window": 64, "max_window_layers": 0}, "sliding"),
        ({"layer_types": ["sliding_attention"], "sliding_window": 64}, "sliding"),
    ],
)
def test_refuses_unsupported_attention_layout(tmp_path, overrides, message):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, config_overrides=overrides)

    with pytest.raises(UnsafeTransformError, match=message):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]


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


def test_published_file_is_world_readable(tmp_path):
    import os
    import stat

    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out = _run_package(root, tmp_path)

    mode = stat.S_IMODE(os.stat(out).st_mode)
    assert mode == 0o644


def test_existing_output_requires_overwrite(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    first = _run_package(root, tmp_path)
    with pytest.raises(FileExistsError):
        _run_package(root, tmp_path)
    second = _run_package(root, tmp_path, overwrite=True)
    assert first == second


def test_output_created_during_verification_is_preserved(tmp_path, monkeypatch):
    import gptqmodel.packaging.gguf_package as module

    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out = tmp_path / "model.gguf"
    existing = b"another publisher's output"
    verify = module._verify_written

    def verify_and_publish(path, plan):
        verify(path, plan)
        with out.open("xb") as handle:
            handle.write(existing)

    monkeypatch.setattr(module, "_verify_written", verify_and_publish)
    with pytest.raises(FileExistsError):
        _run_package(root, tmp_path)
    assert out.read_bytes() == existing
    assert sorted(tmp_path.iterdir()) == [root, out]


@pytest.mark.parametrize("failure_stage", ["construct", "replace"])
def test_writer_failure_leaves_existing_output_and_cleans_temp_file(tmp_path, monkeypatch, failure_stage):
    import gptqmodel.packaging.gguf_package as module

    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    out = _run_package(root, tmp_path)
    existing = out.read_bytes()

    def fail(*args, **kwargs):
        raise OSError(f"injected {failure_stage} failure")

    with monkeypatch.context() as patch:
        if failure_stage == "construct":
            patch.setattr(gguf, "GGUFWriter", fail)
        else:
            patch.setattr(module.os, "replace", fail)
        with pytest.raises(OSError, match=f"injected {failure_stage} failure"):
            _run_package(root, tmp_path, overwrite=True)
    assert out.read_bytes() == existing
    assert sorted(tmp_path.iterdir()) == [root, out]
    assert _run_package(root, tmp_path, overwrite=True) == out
    assert _tensors(out)["blk.0.ffn_down.weight"].tensor_type.name == "Q4_K"


def test_refuses_metadata_that_replaces_generated_fields(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)

    with pytest.raises(PackageError, match=r"general\.architecture"):
        _run_package(root, tmp_path, metadata={"general.architecture": "llama"})
    assert sorted(tmp_path.iterdir()) == [root]

    reader = gguf.GGUFReader(str(_run_package(root, tmp_path, metadata={"general.description": "fixture"})))
    assert reader.get_field("general.description").contents() == "fixture"


def test_signed_extra_metadata_round_trips(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)

    reader = gguf.GGUFReader(str(_run_package(root, tmp_path, metadata={"fixture.offset": -7})))
    assert reader.get_field("fixture.offset").contents() == -7


@pytest.mark.parametrize("alias", ["q8_1", "q8_k"])
def test_refuses_activation_only_ggml_types_as_weights(tmp_path, alias):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(
        root,
        dynamic={".*down_proj": {"bits": alias}},
        extra={_layer(0, "mlp.down_proj.qweight"): _packed((HIDDEN, INTERMEDIATE), alias.upper())},
    )

    with pytest.raises(UnsupportedQuantTypeError, match=alias.upper()):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]


def test_refuses_non_byte_packed_storage(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    name = _layer(0, "self_attn.q_proj.qweight")
    _build_checkpoint(root, extra={name: _packed((Q_DIM, HIDDEN), "Q4_K").to(torch.int16)})

    with pytest.raises(CheckpointFormatError, match=r"q_proj\.qweight.*uint8"):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]


@pytest.mark.parametrize(
    ("model_type", "leaf", "gguf_name"),
    [
        ("qwen2", "self_attn.q_proj.bias", "attn_q.bias"),
        ("qwen3", "self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("qwen3", "self_attn.k_norm.weight", "attn_k_norm.weight"),
    ],
)
def test_refuses_missing_required_attention_tensors(tmp_path, model_type, leaf, gguf_name):
    from safetensors.torch import save_file

    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root, model_type=model_type)
    tensors.pop(_layer(0, leaf))
    save_file(tensors, str(root / "model.safetensors"))

    with pytest.raises(CheckpointFormatError, match=gguf_name):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]


def test_refuses_wide_qwen2_query_projection(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, model_type="qwen2", heads=2 * HEADS)
    with pytest.raises(UnsafeTransformError, match="query width"):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]


@pytest.mark.parametrize("declared", [True, False])
def test_refuses_qwen3_output_projection_bias(tmp_path, declared):
    root = tmp_path / "ckpt"
    root.mkdir()
    extra = {} if declared else {_layer(0, "self_attn.o_proj.bias"): torch.ones(HIDDEN)}
    _build_checkpoint(root, config_overrides={"attention_bias": declared}, extra=extra)
    with pytest.raises(UnsafeTransformError, match="output projection bias"):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]


@pytest.mark.parametrize(
    ("projection", "gguf_name"),
    [("self_attn.o_proj", "attn_output"), ("mlp.down_proj", "ffn_down"), ("mlp.up_proj", "ffn_up")],
)
def test_validates_llama_projection_biases(tmp_path, projection, gguf_name):
    from safetensors.torch import save_file

    root = tmp_path / "ckpt"
    root.mkdir()
    _, tensors = _build_checkpoint(root, model_type="llama", attention_bias=True, mlp_bias=True)
    key = _layer(0, projection + ".bias")
    bias = tensors.pop(key)
    path = str(root / "model.safetensors")
    save_file(tensors, path)
    with pytest.raises(CheckpointFormatError, match=gguf_name):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]

    tensors[key] = bias[:-1]
    save_file(tensors, path)
    with pytest.raises(UnsafeTransformError, match=gguf_name):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]

    tensors[key] = bias
    save_file(tensors, path)
    written = _tensors(_run_package(root, tmp_path))
    np.testing.assert_array_equal(written[f"blk.0.{gguf_name}.bias"].data, bias.numpy())


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


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama"])
def test_refuses_unrepresentable_mlp_activation(tmp_path, model_type):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, model_type=model_type, config_overrides={"hidden_act": "gelu"})

    with pytest.raises(UnsafeTransformError, match="hidden_act.*gelu"):
        _run_package(root, tmp_path)
    assert not (tmp_path / "model.gguf").exists()


@pytest.mark.parametrize(("model_type", "context_length"), [("qwen2", 32768), ("qwen3", 32768), ("llama", 2048)])
def test_implicit_model_defaults_are_written(tmp_path, model_type, context_length):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, model_type=model_type, omit_config_keys=("max_position_embeddings", "rms_norm_eps"))

    reader = gguf.GGUFReader(str(_run_package(root, tmp_path)))
    assert reader.get_field(f"{model_type}.context_length").contents() == context_length
    assert reader.get_field(f"{model_type}.attention.layer_norm_rms_epsilon").contents() == pytest.approx(1e-6)


def test_only_known_derived_buffers_are_ignored(tmp_path):
    from safetensors.torch import save_file

    root = tmp_path / "ckpt"
    root.mkdir()
    buffer = "model.rotary_emb.inv_freq"
    _, tensors = _build_checkpoint(root, extra={buffer: torch.ones(HEAD_DIM // 2)})
    _run_package(root, tmp_path)
    unexpected = buffer + "_scale.weight"
    tensors[unexpected] = torch.ones(1)
    save_file(tensors, str(root / "model.safetensors"))
    with pytest.raises(UnsupportedTensorError, match=unexpected):
        _run_package(root, tmp_path, out_name="invalid.gguf")
    assert not (tmp_path / "invalid.gguf").exists()


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


def test_absent_tie_flag_uses_the_model_default(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["tie_word_embeddings"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(CheckpointFormatError, match="tie_word_embeddings"):
        _run_package(root, tmp_path)
    assert sorted(tmp_path.iterdir()) == [root]


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


@pytest.mark.parametrize("kind", ["unigram", "byte_fallback_bpe"])
def test_refuses_sentencepiece_tokenizers_without_a_gguf_vocabulary(tmp_path, kind):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    if kind == "unigram":
        tokenizer_model = models.Unigram([("a", 0.0), ("<unk>", -1.0)], unk_id=1)
    else:
        tokenizer_model = models.BPE(vocab={"a": 0, "<0xFF>": 1}, merges=[], byte_fallback=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(tokenizer_model))
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root, model_type="llama")

    with pytest.raises(PackageError, match="tokenizer"):
        _run_package(root, tmp_path, tokenizer=tokenizer)
    assert sorted(tmp_path.iterdir()) == [root]


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


def test_tokenizer_metadata_covers_added_tokens():
    from tokenizers import AddedToken

    tokenizer = _tiny_tokenizer()
    tokenizer.add_tokens([AddedToken("<|endoftext|>", normalized=False, special=True)], special_tokens=True)
    tokenizer.add_tokens([AddedToken("</think>", normalized=False)])
    meta = build_tokenizer_metadata(tokenizer, vocab_size=len(tokenizer) + 1, tokenizer_pre="qwen2")

    assert meta.tokens[VOCAB:] == ("<|endoftext|>", "</think>", f"[PAD{VOCAB + 2}]")
    assert meta.token_types[VOCAB:] == (TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_UNUSED)
    assert meta.special_ids["eos"] == 1


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama"])
def test_llamacpp_loads_the_packaged_file(tmp_path, model_type):
    llama_cpp = pytest.importorskip("llama_cpp")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["ab"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")

    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(
        root,
        model_type=model_type,
        vocab_size=len(tokenizer),
        heads=HEADS if model_type == "qwen2" else 2 * HEADS,
        attention_bias=model_type == "llama",
        mlp_bias=model_type == "llama",
        config_overrides={"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}},
        omit_config_keys=("max_position_embeddings", "rms_norm_eps"),
    )
    tokenizer.save_pretrained(root)
    out = package(GGUFPackageConfig(model_path=root, out_path=tmp_path / "model.gguf"))

    model = llama_cpp.Llama(
        model_path=str(out), n_ctx=32, n_gpu_layers=0, n_threads=1, logits_all=True, verbose=False
    )
    try:
        assert model.n_vocab() == len(tokenizer)
        for text in ("ab cab! 你好", "def f(x):\n    return x + 1\n"):
            ids = model.tokenize(text.encode("utf-8"), add_bos=False)
            assert ids == tokenizer.encode(text, add_special_tokens=False)
        model.eval(ids)
        logits = model.scores[model.n_tokens - 1]
        assert np.isfinite(logits).all()
        assert np.any(logits != 0)
    finally:
        model.close()

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def test_chat_template_is_packaged(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    tokenizer = _tiny_tokenizer()
    tokenizer.chat_template = CHAT_TEMPLATE
    out = _run_package(root, tmp_path, tokenizer=tokenizer)

    reader = gguf.GGUFReader(str(out))
    assert reader.get_field("tokenizer.chat_template").contents() == CHAT_TEMPLATE


def test_chat_template_override_wins(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    tokenizer = _tiny_tokenizer()
    tokenizer.chat_template = CHAT_TEMPLATE
    override = CHAT_TEMPLATE + "{# packaged by caller #}"
    out = _run_package(root, tmp_path, tokenizer=tokenizer, chat_template=override)

    reader = gguf.GGUFReader(str(out))
    assert reader.get_field("tokenizer.chat_template").contents() == override


def test_sampling_defaults_are_packaged(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    (root / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
                "min_p": 0.0,
                "repetition_penalty": 1.05,
            }
        ),
        encoding="utf-8",
    )
    out = _run_package(root, tmp_path)

    reader = gguf.GGUFReader(str(out))
    assert float(reader.get_field("general.sampling.temp").contents()) == pytest.approx(0.6)
    assert int(reader.get_field("general.sampling.top_k").contents()) == 20
    assert float(reader.get_field("general.sampling.top_p").contents()) == pytest.approx(0.95)
    assert float(reader.get_field("general.sampling.min_p").contents()) == pytest.approx(0.0)
    assert float(reader.get_field("general.sampling.penalty_repeat").contents()) == pytest.approx(1.05)
    # llama.cpp would otherwise replace its default sampler chain.
    assert reader.get_field("general.sampling.sequence") is None


def test_greedy_generation_maps_to_zero_temperature(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False, "temperature": 0.9}), encoding="utf-8"
    )
    out = _run_package(root, tmp_path)

    reader = gguf.GGUFReader(str(out))
    assert float(reader.get_field("general.sampling.temp").contents()) == 0.0


def test_caller_metadata_wins_over_generation_config(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    (root / "generation_config.json").write_text(json.dumps({"temperature": 0.6}), encoding="utf-8")
    out = _run_package(root, tmp_path, metadata={"general.sampling.temp": 0.0})

    reader = gguf.GGUFReader(str(out))
    assert float(reader.get_field("general.sampling.temp").contents()) == 0.0


def test_malformed_generation_config_is_ignored(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _build_checkpoint(root)
    (root / "generation_config.json").write_text("{not json", encoding="utf-8")
    out = _run_package(root, tmp_path)

    reader = gguf.GGUFReader(str(out))
    assert reader.get_field("general.sampling.temp") is None
