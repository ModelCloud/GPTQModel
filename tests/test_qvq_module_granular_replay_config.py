# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization import (
    ModuleGranularReplayConfig,
    QuantizeConfig,
    QVQConfig,
)
from scripts.validate_qvq_p4_live_prefix import _parser


def _base_config(**kwargs):
    return QVQConfig(
        bits=2,
        format="qvq_v2b2_p32",
        rounding="yaqa",
        offload_to_disk=False,
        **kwargs,
    )


def test_module_granular_replay_is_default_off_and_boolean_true_uses_safe_defaults():
    baseline = _base_config()
    enabled = _base_config(module_granular_replay=True)

    assert baseline.module_granular_replay is None
    assert isinstance(enabled.module_granular_replay, ModuleGranularReplayConfig)
    assert enabled.module_granular_replay.subsets == ("attention_qkvo",)
    assert enabled.module_granular_replay.roles() == ("q_proj", "v_proj", "k_proj", "o_proj")
    assert enabled.module_granular_replay.alternative_bank_ids == (1, 2, 3)
    assert enabled.module_granular_replay.search_folds == 2
    assert enabled.module_granular_replay.require_disjoint_confirmation is True


def test_module_granular_replay_round_trips_without_changing_inference_layout():
    baseline = _base_config()
    configured = _base_config(
        module_granular_replay={
            "subsets": ["attention_qk", "mlp_gate_up_down"],
            "module_order": [
                "q_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "alternative_bank_ids": [1, 3],
            "search_folds": 3,
            "minimum_relative_kl_improvement": 0.002,
            "topn_regression_limit": 0.001,
        },
    )

    payload = configured.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert reloaded.to_dict() == payload
    assert reloaded.module_granular_replay.roles() == (
        "q_proj",
        "k_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert reloaded.module_granular_replay.includes_module("model.layers.2.self_attn.q_proj")
    assert not reloaded.module_granular_replay.includes_module("model.layers.2.self_attn.v_proj")
    assert configured.quant_linear_init_kwargs() == baseline.quant_linear_init_kwargs()


def test_live_prefix_driver_exposes_module_granular_replay_name_and_subset_scope():
    args = _parser().parse_args(
        [
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--prefix-artifact",
            "prefix.safetensors",
            "--yaqa-factor-cache",
            "factors.pt",
            "--yaqa-metadata",
            "factors.json",
            "--output",
            "report.json",
            "--module-granular-replay",
            "--replay-subsets",
            "attention_vo",
        ]
    )

    assert args.module_granular_replay is True
    assert args.replay_subsets == ["attention_vo"]
    assert args.module_replay_search_folds == 2
    assert args.replay_folds == 1


@pytest.mark.parametrize(
    ("replay", "message"),
    (
        ({"subsets": ["unknown"]}, "unsupported subsets"),
        ({"subsets": ["attention_qk"], "module_order": ["q_proj"]}, "missing.*k_proj"),
        ({"alternative_bank_ids": [0]}, "alternative bank IDs"),
        ({"alternative_bank_ids": [1, 1]}, "must not contain duplicates"),
        ({"strategy": "beam"}, "greedy"),
        ({"replay_horizon": "subset"}, "final_logits"),
        ({"search_folds": 1}, "at least two"),
        ({"require_disjoint_confirmation": False}, "disjoint confirmation"),
        ({"fallback": "best_local"}, "canonical_v2_yaqa"),
    ),
)
def test_module_granular_replay_rejects_unvalidated_controls(replay, message):
    with pytest.raises(ValueError, match=message):
        _base_config(module_granular_replay=replay)


@pytest.mark.parametrize(
    "config_kwargs",
    (
        {"format": "qvq", "rounding": "yaqa"},
        {"format": "qvq_v2b2_p32", "rounding": "block_ldlq"},
        {
            "format": "qvq_v2b2_p32",
            "rounding": "yaqa",
            "yaqa": {"sample_strategy": "64_16x16"},
        },
    ),
)
def test_module_granular_replay_requires_exact_v2b2_yaqa_candidates(config_kwargs):
    with pytest.raises(ValueError, match="module-granular replay|sample_strategy"):
        QVQConfig(
            bits=2,
            module_granular_replay=True,
            offload_to_disk=False,
            **config_kwargs,
        )
