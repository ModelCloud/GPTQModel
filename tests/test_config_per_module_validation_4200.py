# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4200: per-module `dynamic` overrides must be validated as merged settings.

Every case below passed construction but failed (or was silently coerced)
later, when the module was cloned or packed:

* a BitsAndBytes bits-only override inherited the global subtype (`nf4` is not
  an 8-bit format);
* an FP8 `weight_scale_method="row"` override inherited the global
  `weight_block_size`;
* a GGUF subtype `bits` override was rejected unless `format` was repeated, so
  the Q4_K_M + Q6_K recipe could not be expressed;
* per-module `sym`/`desc_act` were coerced with `bool()`, and AWQ ignored a
  per-module `sym` in its staging math while labelling the packed layer with it.
"""

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import (
    FORMAT,
    AWQConfig,
    BitsAndBytesConfig,
    FP8Config,
    GGUFConfig,
    GPTQConfig,
    clone_weight_only_config_for_module,
)
from gptqmodel.quantization.module_config import resolve_module_constructor_config


NAME = "model.layers.0.mlp.down_proj"
OTHER = "model.layers.1.mlp.down_proj"


# --- BitsAndBytes -----------------------------------------------------------


def test_bnb_bits_only_override_derives_its_own_subtype():
    config = BitsAndBytesConfig(bits=4, format="nf4", dynamic={NAME: {"bits": 8}})

    assert dict(config.dynamic)[NAME]["format"] == "int8"
    clone = clone_weight_only_config_for_module(config, NAME)
    assert clone.bits == 8
    assert clone.format == "int8"


def test_bnb_bits_only_override_derives_the_default_for_four_bits():
    config = BitsAndBytesConfig(bits=8, format="int8", dynamic={NAME: {"bits": 4}})

    clone = clone_weight_only_config_for_module(config, NAME)
    assert clone.bits == 4
    assert clone.format == "fp4"


def test_bnb_explicitly_incompatible_subtype_is_still_rejected():
    with pytest.raises(ValueError, match="`format` must be one of"):
        BitsAndBytesConfig(
            bits=4,
            format="nf4",
            dynamic={NAME: {"bits": 8, "format": "nf4"}},
        )


def test_bnb_module_constructor_derives_the_subtype_too():
    module_config = resolve_module_constructor_config(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        format=FORMAT.BITSANDBYTES,
        init_kwargs={"format": "nf4", "block_size": 64, "compress_statistics": True},
        overrides={"bits": 8},
    )

    assert module_config.bits == 8
    assert module_config.options_dict()["format"] == "int8"


# --- FP8 --------------------------------------------------------------------


def test_fp8_row_override_drops_the_inherited_block_size():
    config = FP8Config(
        weight_scale_method="block",
        weight_block_size=[128, 128],
        dynamic={NAME: {"weight_scale_method": "row"}},
    )

    rule = dict(config.dynamic)[NAME]
    assert rule["weight_scale_method"] == "row"
    assert rule["weight_block_size"] is None
    clone = clone_weight_only_config_for_module(config, NAME)
    assert clone.weight_scale_method == "row"
    assert clone.weight_block_size is None


def test_fp8_block_override_without_a_size_keeps_the_global_size():
    config = FP8Config(
        weight_scale_method="block",
        weight_block_size=[64, 64],
        dynamic={NAME: {"weight_scale_method": "block"}},
    )

    clone = clone_weight_only_config_for_module(config, NAME)
    assert clone.weight_scale_method == "block"
    assert clone.weight_block_size == [64, 64]


def test_fp8_row_with_an_explicit_block_size_is_rejected():
    with pytest.raises(ValueError, match="only valid when"):
        FP8Config(
            weight_scale_method="row",
            dynamic={NAME: {"weight_scale_method": "row", "weight_block_size": [128, 128]}},
        )


def test_fp8_module_constructor_derives_the_effective_scale_pair():
    module_config = resolve_module_constructor_config(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        format=FORMAT.FP8,
        init_kwargs={
            "format": "float8_e4m3fn",
            "weight_scale_method": "block",
            "weight_block_size": [128, 128],
            "weight_scale_semantics": "inverse",
        },
        overrides={"weight_scale_method": "row"},
    )

    options = module_config.options_dict()
    assert options["weight_scale_method"] == "row"
    assert options["weight_block_size"] is None


# --- GGUF -------------------------------------------------------------------


def test_gguf_subtype_bits_override_implies_its_own_format():
    config = GGUFConfig(bits="q4_k_m", dynamic={NAME: {"bits": "q6_k"}})

    clone = clone_weight_only_config_for_module(config, NAME)
    assert clone.bits == 6
    assert clone.runtime_bits.to_string() == "q6_k"
    assert clone_weight_only_config_for_module(config, OTHER).runtime_bits.to_string() == "q4_k_m"


def test_gguf_conflicting_bits_and_format_are_still_rejected():
    with pytest.raises(ValueError, match="Unsupported GGUF combination"):
        GGUFConfig(bits="q4_k_m", dynamic={NAME: {"bits": "q6_k", "format": "q4_k_m"}})


def test_gguf_module_constructor_accepts_a_subtype_only_override():
    module_config = resolve_module_constructor_config(
        bits="q4_k_m",
        group_size=-1,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        format=FORMAT.GGUF,
        overrides={"bits": "q6_k"},
    )

    assert module_config.bits.to_string() == "q6_k"


# --- per-module booleans ----------------------------------------------------


@pytest.mark.parametrize("config_cls", [GPTQConfig, AWQConfig], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("key", ["sym", "desc_act"])
def test_per_module_booleans_are_type_checked(config_cls, key):
    with pytest.raises(TypeError, match=rf"dynamic\['.*'\]\.{key}` must be a bool"):
        config_cls(bits=4, dynamic={NAME: {key: "False"}})


def test_per_module_booleans_accept_real_bools():
    config = GPTQConfig(bits=4, dynamic={NAME: {"sym": False, "desc_act": True}})

    rule = dict(config.dynamic)[NAME]
    assert rule["sym"] is False
    assert rule["desc_act"] is True


def test_module_constructor_config_refuses_a_coerced_boolean():
    with pytest.raises(ValueError, match="boolean sym"):
        resolve_module_constructor_config(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            pack_dtype=torch.int32,
            format=FORMAT.GPTQ,
            overrides={"sym": "False"},
        )


# --- AWQ per-module symmetry ------------------------------------------------


def _awq_stub(bits=4, dynamic=None, sym=True):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=-1, sym=sym, dynamic=dynamic)
        _module_sym = AWQProcessor._module_sym
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
        _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into

    return Stub()


def test_awq_module_sym_follows_the_dynamic_override():
    stub = _awq_stub(dynamic={NAME: {"sym": False}})

    assert stub._module_sym(NAME) is False
    assert stub._module_sym(OTHER) is True
    assert stub._module_sym(None) is True


def test_awq_staging_uses_the_per_module_sym():
    stub = _awq_stub(dynamic={NAME: {"sym": False}})
    weight = torch.randn(64, 64) + 2.0

    sym_wq, sym_scales, sym_zeros = stub.pseudo_quantize_tensor(weight, group_size=-1)
    affine_wq, affine_scales, affine_zeros = stub.pseudo_quantize_tensor(
        weight, group_size=-1, sym=False
    )

    assert not torch.allclose(sym_wq, affine_wq)
    assert not torch.allclose(sym_scales, affine_scales)
    # Symmetric AWQ shifts signed codes into unsigned space with 2^(bits-1).
    assert torch.allclose(sym_zeros, torch.full_like(sym_zeros, 2 ** 3))
    assert not torch.allclose(affine_zeros, torch.full_like(affine_zeros, 2 ** 3))
