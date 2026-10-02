# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation of the mixed FP8-attention / NVFP4-MLP activation policy."""

import json

import pytest

from gptqmodel.quantization.config import QuantizeConfig


NVFP4 = "w4a_nvfp4"
FP8 = "w4afp8"


def _config(activation):
    return QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation=activation, rotation="hadamard",
    )


def test_attention_split_accepts_fp8_attention():
    config = _config({
        "version": 4, "mode": NVFP4, "recipe": "least_squares",
        "attention": {"mode": FP8},
    })
    assert config.activation_mode == NVFP4
    assert config.activation_recipe == "least_squares"
    assert config.activation_attention_mode == FP8
    assert config.activation_attention_recipe is None


def test_attention_split_inherits_nvfp4_recipe():
    config = _config({
        "version": 3, "mode": NVFP4, "recipe": "least_squares",
        "attention": {"mode": NVFP4},
    })
    assert config.activation_attention_mode == NVFP4
    assert config.activation_attention_recipe == "least_squares"


def test_attention_split_rejects_fp8_recipe():
    with pytest.raises(ValueError, match="does not use an NVFP4 scale recipe"):
        _config({
            "version": 3, "mode": NVFP4,
            "attention": {"mode": FP8, "recipe": "least_squares"},
        })


def test_attention_split_rejects_fp8_stream():
    with pytest.raises(ValueError, match="only refines an NVFP4"):
        _config({"version": 3, "mode": FP8, "attention": {"mode": FP8}})


def test_attention_split_rejects_version_two():
    with pytest.raises(ValueError, match="version 3 or 4"):
        _config({"version": 2, "mode": NVFP4, "attention": {"mode": FP8}})


def test_attention_split_rejects_unknown_fields():
    with pytest.raises(ValueError, match="only supports"):
        _config({"version": 3, "mode": NVFP4, "attention": {"mode": FP8, "extra": 1}})


def test_split_view_is_metadata_only(tmp_path):
    from tests.models.w4a_quality_regression import prepare_attention_split_view

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"native-int4-weights")
    (source / "tokenizer.json").write_text("{}")
    (source / "quantize_config.json").write_text(json.dumps({
        "bits": 4, "pack_dtype": "int32",
        "activation": {"version": 4, "mode": NVFP4, "recipe": "lsq"},
    }))
    view = tmp_path / "view"
    prepare_attention_split_view(source, view)
    assert (view / "model.safetensors").samefile(source / "model.safetensors")
    assert (view / "model.safetensors").read_bytes() == b"native-int4-weights"
    config = json.loads((view / "quantize_config.json").read_text())
    assert config["activation"]["attention"] == {"mode": FP8}
    assert config["activation"]["mode"] == NVFP4


def test_mlp_override_accepts_fp8_layers():
    config = _config({
        "version": 4, "mode": NVFP4, "recipe": "least_squares",
        "attention": {"mode": FP8},
        "mlp": {"mode": FP8, "layers": [0, 7]},
    })
    assert config.activation_mlp_fp8_layers == (0, 7)
    assert config.activation_mode == NVFP4
    assert config.activation_attention_mode == FP8


def test_mlp_override_sorts_and_deduplicates_shape():
    config = _config({
        "version": 3, "mode": NVFP4, "recipe": "least_squares",
        "mlp": {"mode": FP8, "layers": (5, 1, 3)},
    })
    assert config.activation_mlp_fp8_layers == (1, 3, 5)


def test_mlp_override_defaults_to_none():
    config = _config({"version": 4, "mode": NVFP4, "recipe": "least_squares"})
    assert config.activation_mlp_fp8_layers is None


def test_mlp_override_rejects_nvfp4_mode():
    with pytest.raises(ValueError, match="only promotes layers to `w4afp8`"):
        _config({
            "version": 4, "mode": NVFP4, "recipe": "least_squares",
            "mlp": {"mode": NVFP4, "layers": [0]},
        })


def test_mlp_override_rejects_empty_layers():
    with pytest.raises(ValueError, match="non-empty list"):
        _config({
            "version": 4, "mode": NVFP4, "recipe": "least_squares",
            "mlp": {"mode": FP8, "layers": []},
        })


def test_mlp_override_rejects_duplicate_layers():
    with pytest.raises(ValueError, match="unique non-negative"):
        _config({
            "version": 4, "mode": NVFP4, "recipe": "least_squares",
            "mlp": {"mode": FP8, "layers": [1, 1]},
        })


def test_mlp_override_rejects_negative_layers():
    with pytest.raises(ValueError, match="unique non-negative"):
        _config({
            "version": 4, "mode": NVFP4, "recipe": "least_squares",
            "mlp": {"mode": FP8, "layers": [-1]},
        })


def test_mlp_override_rejects_unknown_fields():
    with pytest.raises(ValueError, match="only supports `mode` and `layers`"):
        _config({
            "version": 4, "mode": NVFP4,
            "mlp": {"mode": FP8, "layers": [0], "recipe": "least_squares"},
        })


def test_mlp_override_rejects_fp8_stream():
    with pytest.raises(ValueError, match="only refines an NVFP4 activation stream"):
        _config({"version": 3, "mode": FP8, "mlp": {"mode": FP8, "layers": [0]}})


def test_mlp_override_rejects_version_two():
    with pytest.raises(ValueError, match="requires activation version 3 or 4"):
        _config({
            "version": 2, "mode": NVFP4,
            "mlp": {"mode": "w4afp8", "layers": [0]},
        })


def test_mlp_view_is_metadata_only(tmp_path):
    from tests.models.w4a_quality_regression import prepare_mlp_override_view

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"native-int4-weights")
    (source / "tokenizer.json").write_text("{}")
    (source / "quantize_config.json").write_text(json.dumps({
        "bits": 4, "pack_dtype": "int32",
        "activation": {"version": 4, "mode": NVFP4, "recipe": "lsq",
                       "attention": {"mode": FP8}},
    }))
    view = tmp_path / "view"
    prepare_mlp_override_view(source, view, [7, 0])
    assert (view / "model.safetensors").samefile(source / "model.safetensors")
    assert (view / "model.safetensors").read_bytes() == b"native-int4-weights"
    config = json.loads((view / "quantize_config.json").read_text())
    assert config["activation"]["mlp"] == {"mode": FP8, "layers": [0, 7]}
    assert config["activation"]["attention"] == {"mode": FP8}
    assert config["activation"]["mode"] == NVFP4


def test_mlp_view_rejects_existing_override(tmp_path):
    from tests.models.w4a_quality_regression import prepare_mlp_override_view

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"native-int4-weights")
    (source / "quantize_config.json").write_text(json.dumps({
        "bits": 4, "pack_dtype": "int32",
        "activation": {"version": 4, "mode": NVFP4, "recipe": "lsq",
                       "mlp": {"mode": FP8, "layers": [0]}},
    }))
    with pytest.raises(ValueError, match="already carries an MLP override"):
        prepare_mlp_override_view(source, tmp_path / "view", [1])


def test_mlp_view_rejects_duplicate_layers(tmp_path):
    from tests.models.w4a_quality_regression import prepare_mlp_override_view

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"native-int4-weights")
    (source / "quantize_config.json").write_text(json.dumps({
        "bits": 4, "pack_dtype": "int32",
        "activation": {"version": 4, "mode": NVFP4, "recipe": "lsq"},
    }))
    with pytest.raises(ValueError, match="unique decoder layer indices"):
        prepare_mlp_override_view(source, tmp_path / "view", [1, 1])


# ---------------------------------------------------------------------------
# ActivationConfig: the typed schema behind QuantizeConfig.activation
# ---------------------------------------------------------------------------

def _activation_config():
    from gptqmodel.quantization.config import ActivationConfig

    return ActivationConfig


def test_activation_config_string_shorthand_is_version_three():
    config = _activation_config().from_value("w4afp8")
    assert config.version == 3
    assert config.mode == FP8
    assert config.recipe is None
    assert config.to_dict() == {"version": 3, "mode": FP8}


def test_activation_config_nvfp4_shorthand_defaults_recipe():
    config = _activation_config().from_value(NVFP4)
    assert config.recipe == "least_squares"
    assert config.to_dict() == {"version": 3, "mode": NVFP4, "recipe": "least_squares"}


def test_activation_config_none_stays_none():
    assert _activation_config().from_value(None) is None


def test_activation_config_version_two_keeps_four_six():
    config = _activation_config().from_value({"version": 2, "mode": NVFP4})
    assert config.recipe == "four_six"


def test_activation_config_expands_legacy_recipe_alias():
    config = _activation_config().from_value(
        {"version": 4, "mode": NVFP4, "recipe": "lsq"}, rotation="hadamard"
    )
    assert config.recipe == "least_squares"


def test_activation_config_round_trips_mixed_stream():
    raw = {
        "version": 4, "mode": NVFP4, "recipe": "least_squares_grid",
        "attention": {"mode": FP8},
        "mlp": {"mode": FP8, "layers": [15, 11]},
    }
    config = _activation_config().from_value(raw, rotation="hadamard")
    assert config.to_dict() == {
        "version": 4, "mode": NVFP4, "recipe": "least_squares_grid",
        "attention": {"mode": FP8},
        "mlp": {"mode": FP8, "layers": [11, 15]},
    }


def test_activation_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="unsupported field"):
        _activation_config().from_value({"version": 3, "mode": FP8, "extra": 1})


def test_activation_missing_version_or_mode_is_rejected():
    with pytest.raises(ValueError, match="must contain `version` and `mode`"):
        _activation_config().from_value({"version": 3})


def test_activation_version_four_requires_rotation():
    with pytest.raises(ValueError, match="requires NVFP4 and rotation"):
        _activation_config().from_value({"version": 4, "mode": NVFP4})


def test_activation_version_four_rejects_fp8_mode():
    with pytest.raises(ValueError, match="requires NVFP4 and rotation"):
        _activation_config().from_value({"version": 4, "mode": FP8}, rotation="hadamard")


def test_activation_rejects_legacy_version_one():
    with pytest.raises(ValueError, match="must be 2, 3, or 4"):
        _activation_config().from_value({"version": 1, "mode": FP8})


def test_activation_global_scales_require_version_four_nvfp4():
    with pytest.raises(ValueError, match="require version-4 NVFP4"):
        _activation_config().from_value({
            "version": 3, "mode": NVFP4,
            "global_scales": {"model.layers.0.input": 0.5},
        })


def test_activation_normalizes_global_scales():
    config = _activation_config().from_value({
        "version": 4, "mode": NVFP4,
        "global_scales": {"model.layers.0.input": 0.5},
    }, rotation="hadamard")
    assert config.global_scales == {"model.layers.0.input": 0.5}


def test_activation_global_scales_reject_unknown_producer():
    with pytest.raises(ValueError, match="positive finite global scales"):
        _activation_config().from_value({
            "version": 4, "mode": NVFP4,
            "global_scales": {"embed_tokens": 0.5},
        }, rotation="hadamard")


def test_quantize_config_stores_canonical_activation_dict():
    config = _config({
        "version": 4, "mode": NVFP4, "recipe": "lsq",
        "attention": {"mode": FP8},
        "mlp": {"mode": FP8, "layers": [3, 1]},
    })
    assert config.activation == {
        "version": 4,
        "mode": NVFP4,
        "recipe": "least_squares",
        "attention": {"mode": FP8},
        "mlp": {"mode": FP8, "layers": [1, 3]},
    }
    # The typed view and the stored dict agree.
    assert config.activation_attention_mode == FP8
    assert config.activation_mlp_fp8_layers == (1, 3)
