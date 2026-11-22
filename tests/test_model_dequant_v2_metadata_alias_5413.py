# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Equivalent GPTQ v2 metadata must produce the same dense weights (#5413).

The loader accepts ``quantize_config.json`` and its ``quant_config.json`` alias,
and it recognizes a v2 layout declared through ``checkpoint_format`` or through
``format``. ``dequantize_model`` used to read only the canonical sidecar and to
decide the v1 zero correction from ``checkpoint_format`` alone, so the same
packed payload converted differently depending on which accepted spelling the
checkpoint used.
"""

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gptqmodel.utils.model_dequant import dequantize_model, pack_cols, unpack_cols
from tests.test_model_dequant_dense import _tiny_llama_checkpoint


BITS = 4


def _dequantized(out_dir):
    with safe_open(str(out_dir / "model.safetensors"), framework="pt") as reader:
        return {key: reader.get_tensor(key) for key in reader.keys()}


def _variant(tmp_path, name, *, sidecar="quantize_config.json", patch=None, drop=()):
    """Write the v2 fixture with one accepted metadata spelling."""

    source = tmp_path / name
    source.mkdir()
    expected = _tiny_llama_checkpoint(source, bits=BITS, lm_head_bits=BITS, legacy_config=True)
    payload = json.loads((source / "quantize_config.json").read_text(encoding="utf-8"))
    for key in drop:
        payload.pop(key, None)
    payload.update(patch or {})
    (source / "quantize_config.json").unlink()
    (source / sidecar).write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / f"{name}-out"
    dequantize_model(source, out_dir)
    return _dequantized(out_dir), expected


def _assert_same_weights(actual, reference):
    assert actual.keys() == reference.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)


def test_quant_config_alias_matches_the_canonical_sidecar(tmp_path):
    reference, _ = _variant(tmp_path, "canonical")
    alias, _ = _variant(tmp_path, "alias", sidecar="quant_config.json")
    _assert_same_weights(alias, reference)


def test_format_alias_matches_checkpoint_format(tmp_path):
    reference, _ = _variant(tmp_path, "canonical")
    # Same payload, but the layout is declared through the `format` alias.
    alias, _ = _variant(
        tmp_path,
        "format-alias",
        patch={"format": "gptq_v2"},
        drop=("checkpoint_format",),
    )
    _assert_same_weights(alias, reference)


def test_declared_v1_still_gets_the_zero_correction(tmp_path):
    reference, expected = _variant(tmp_path, "v2-reference")

    # Rewrite the fixture's qzeros into the v1 encoding (per-field -1) and
    # declare v1. The dequantizer must add the offset back and land on the same
    # dense weights as the v2 payload.
    source = tmp_path / "v1"
    source.mkdir()
    _tiny_llama_checkpoint(source, bits=BITS, lm_head_bits=BITS, legacy_config=True)
    payload = json.loads((source / "quantize_config.json").read_text(encoding="utf-8"))
    payload["checkpoint_format"] = "gptq"
    (source / "quantize_config.json").write_text(json.dumps(payload), encoding="utf-8")

    tensors = load_file(str(source / "model.safetensors"))
    for key in list(tensors):
        if key.endswith(".qzeros"):
            zeros = unpack_cols(tensors[key], BITS)
            tensors[key] = pack_cols((zeros - 1) & ((1 << BITS) - 1), BITS, pack_dtype=torch.int32)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(source / "model.safetensors"))

    out_dir = tmp_path / "v1-out"
    dequantize_model(source, out_dir)
    actual = _dequantized(out_dir)
    _assert_same_weights(actual, reference)

    # The fixture's own oracle still holds for the shifted payload.
    for key, oracle in expected.items():
        torch.testing.assert_close(actual[key].to(torch.float64), oracle, rtol=1e-2, atol=2e-2)
