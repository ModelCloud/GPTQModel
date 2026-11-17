# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``save()`` must not silently discard keywords on the quantized path (#4686).

The quantized branch has no ``save_pretrained`` passthrough, so every extra
keyword used to vanish without an error or a log line. A typo of a real
parameter, and an HF option such as ``push_to_hub=True``, both reported success
and did nothing.

These tests drive ``save()`` through a stub so the argument handling is checked
without paying for a real quantization run.
"""

import pytest

from gptqmodel.models.base import BaseQModel


def _quantized_stub(*, weight_only: bool = False):
    """A ``BaseQModel`` shell with only what the quantized save path touches."""

    stub = object.__new__(BaseQModel)
    stub.quantized = True
    stub.quant_override_files = {}
    stub.quant_region_timer = None
    stub._model_free_weight_only_replacements_only = weight_only
    # Never reached by the cases below; present so a regression fails loudly
    # here instead of in the file system.
    stub._warn_fused_unit_dynamic_overrides = lambda: None
    return stub


def test_quantized_save_rejects_a_typo_with_a_close_match_hint(tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument") as excinfo:
        BaseQModel.save(_quantized_stub(), str(tmp_path), max_shard_sise="20MB")

    message = str(excinfo.value)
    assert "`max_shard_sise`" in message
    assert "did you mean `max_shard_size`" in message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"safe_serialization": False},
        {"push_to_hub": True, "repo_id": "nonexistent/xyz"},
    ],
)
def test_quantized_save_rejects_options_it_cannot_honor(tmp_path, kwargs):
    """These have no close match, but must still raise rather than no-op."""

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        BaseQModel.save(_quantized_stub(), str(tmp_path), **kwargs)


def test_quantized_save_accepts_its_own_parameters(tmp_path):
    """The guard must not reject ``save()``'s documented parameters.

    ``save_quantized`` is stubbed: this asserts argument handling, not writing.
    """

    stub = _quantized_stub()
    seen = {}
    stub.save_quantized = lambda **call: seen.update(call)

    BaseQModel.save(
        stub,
        str(tmp_path),
        safetensors_metadata={"a": "b"},
        max_shard_size="20MB",
        meta_quantizer="m",
        eora_path="adapters",
        split_by="size",
    )

    assert seen["max_shard_size"] == "20MB"
    assert seen["meta_quantizer"] == "m"
    assert seen["eora_path"] == "adapters"
    assert seen["split_by"] == "size"


def test_weight_only_replacement_save_rejects_eora_path_and_split_by(tmp_path):
    """The module-replacement writer cannot store adapters or split by size."""

    with pytest.raises(ValueError, match="not supported when saving a weight-only") as excinfo:
        BaseQModel.save(_quantized_stub(weight_only=True), str(tmp_path), eora_path="adapters")

    assert "`eora_path`" in str(excinfo.value)

    with pytest.raises(ValueError, match="not supported when saving a weight-only"):
        BaseQModel.save(_quantized_stub(weight_only=True), str(tmp_path), split_by="size")
