# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""#5580: remote configs that declare their layer layout outside ``layer_types``.

Nemotron-H publishes a ``hybrid_override_pattern`` (and a ``layers_block_type``
property built from it), DeciLM publishes ``num_key_value_heads_per_layer``, and
Nemotron-NAS publishes ``block_configs[i].attention``. The estimator used to
read only ``layer_types`` and a global ``num_key_value_heads``, so it counted
every layer as full MHA attention and still reported ``is_exact=True``: 14x too
high on Nemotron-H, 15x on DeciLM and 40x on Nemotron-NAS.
"""

from types import SimpleNamespace

import torch

from gptqmodel.utils.kv_cache import estimate_kv_cache


def _estimate(config):
    estimate = estimate_kv_cache(config, dtype=torch.bfloat16)
    assert estimate is not None
    return estimate


def _base(**overrides):
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 6,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "torch_dtype": torch.bfloat16,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def test_nemotron_h_hybrid_override_pattern_only_counts_attention_layers():
    """The remote config exposes the layout as a pattern, not ``layer_types``."""

    config = _base(
        num_hidden_layers=10,
        hidden_size=4480,
        num_attention_heads=40,
        num_key_value_heads=8,
        hybrid_override_pattern="M-M-M*-M-M",
    )
    estimate = _estimate(config)

    # One '*' layer: 2 (K and V) * 8 kv heads * 128 head_dim * 2 bytes.
    assert estimate.kv_bytes_per_token == 2 * 8 * 128 * 2
    assert estimate.growing_layers == 1
    assert estimate.linear_layers == 5
    assert estimate.block_layers == 4
    assert estimate.is_exact is True
    assert estimate.partial == ()


def test_nemotron_h_layers_block_type_property_is_read():
    """The remote class builds ``layers_block_type`` from the pattern."""

    config = _base(
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        layers_block_type=["mamba", "attention", "mlp", "moe"],
    )
    estimate = _estimate(config)

    assert estimate.kv_bytes_per_token == 2 * 2 * 16 * 2
    assert estimate.growing_layers == 1
    assert estimate.linear_layers == 1
    assert estimate.block_layers == 2
    assert estimate.is_exact is True


def test_decilm_num_key_value_heads_per_layer_is_summed():
    """DeciLM declares one KV-head count per layer and no global value."""

    config = _base(
        num_key_value_heads=None,
        num_key_value_heads_per_layer=[4, 4, 2, 2, 1, 1],
    )
    estimate = _estimate(config)

    assert [layer.dims for layer in estimate.layers] == [
        2 * heads * 128 for heads in (4, 4, 2, 2, 1, 1)
    ]
    assert estimate.kv_bytes_per_token == 2 * (4 + 4 + 2 + 2 + 1 + 1) * 128 * 2
    assert estimate.growing_layers == 6
    assert estimate.is_exact is True


def test_nemotron_nas_no_op_layers_cache_nothing_and_groups_set_kv_heads():
    """``block_configs[i].attention`` is the only layout Nemotron-NAS declares."""

    def attention(n_heads_in_group, *, no_op=False, replace_with_linear=False):
        return {
            "attention": {
                "n_heads_in_group": n_heads_in_group,
                "no_op": no_op,
                "replace_with_linear": replace_with_linear,
            },
        }

    config = _base(
        num_hidden_layers=6,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=None,
        head_dim=None,
        block_configs=[
            attention(4),
            attention(0, no_op=True),
            attention(4),
            attention(0, replace_with_linear=True),
            attention(4),
            attention(4),
        ],
    )
    estimate = _estimate(config)

    # hidden_size // heads = 128 head_dim; 16 // 4 = 4 kv heads.
    per_layer = 2 * 4 * 128
    assert [layer.dims for layer in estimate.layers] == [per_layer, 0, per_layer, 0, per_layer, per_layer]
    assert estimate.kv_bytes_per_token == 4 * per_layer * 2
    assert estimate.growing_layers == 4
    assert estimate.block_layers == 2
    assert estimate.is_exact is True


def test_nemotron_nas_object_style_block_configs_are_read():
    """A loaded remote config builds ``BlockConfig`` dataclasses, not dicts."""

    def block(n_heads_in_group):
        return SimpleNamespace(
            attention=SimpleNamespace(
                n_heads_in_group=n_heads_in_group, no_op=False, replace_with_linear=False,
            ),
        )

    config = _base(
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=None,
        block_configs=[block(8), block(8)],
    )
    estimate = _estimate(config)

    assert [layer.dims for layer in estimate.layers] == [2 * 2 * 128] * 2
    assert estimate.kv_bytes_per_token == 2 * 2 * 2 * 128 * 2
    assert estimate.is_exact is True


def test_unmodelled_block_config_options_mark_the_estimate_inexact():
    config = _base(
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=None,
        block_configs=[
            {"attention": {"n_heads_in_group": 8, "window_length": 4096}},
            {"attention": {"n_heads_in_group": 8}},
        ],
    )
    estimate = _estimate(config)

    assert estimate.kv_bytes_per_token == 2 * 2 * 2 * 128 * 2
    assert estimate.is_exact is False
    assert any("not modelled" in note for note in estimate.partial)


def test_short_per_layer_layout_is_inexact_instead_of_silently_wrong():
    config = _base(num_key_value_heads=None, num_key_value_heads_per_layer=[4, 4])
    estimate = _estimate(config)

    assert estimate.is_exact is False
    assert any("shorter than the stack" in note for note in estimate.partial)


def test_mismatched_hybrid_pattern_is_inexact_instead_of_full_attention():
    config = _base(
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        hybrid_override_pattern="M*-",
    )
    estimate = _estimate(config)

    assert estimate.is_exact is False
    assert any("does not match the layer count" in note for note in estimate.partial)
