# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Schema, validation, serialization, and compatibility tests for
``ViterbiPruningConfig`` and ``QVQConfig.viterbi_pruning``."""

import pytest

import gptqmodel.quantization as quantization_public
from gptqmodel.quantization.config import FORMAT, METHOD, QVQConfig, ViterbiPruningConfig
from gptqmodel.quantization.qvq_pruning import (
    VITERBI_PRUNING_AUTO,
    VITERBI_PRUNING_AUTO_ERROR,
    VITERBI_PRUNING_OFF,
    VITERBI_PRUNING_REQUIRED,
    resolve_viterbi_pruning_policy,
    viterbi_pruning_dispatch_code,
)


def _config(**overrides):
    kwargs = {"bits": 3.0, "format": FORMAT.QVQ_V2B2_P32, "bank_count": 2}
    kwargs.update(overrides)
    return QVQConfig(**kwargs)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_viterbi_pruning_config_is_exported_beside_qvq_config():
    assert quantization_public.ViterbiPruningConfig is ViterbiPruningConfig
    assert quantization_public.QVQConfig is QVQConfig


def test_defaults_are_the_historical_automatic_behavior():
    policy = ViterbiPruningConfig()
    assert (policy.mode, policy.strategy, policy.exact, policy.fallback) == (
        "auto",
        "norm_band",
        True,
        "baseline",
    )
    assert viterbi_pruning_dispatch_code(policy) == VITERBI_PRUNING_AUTO


def test_qvq_config_defaults_to_an_auto_pruning_sibling_of_yaqa():
    config = _config()
    assert isinstance(config.viterbi_pruning, ViterbiPruningConfig)
    assert config.viterbi_pruning == ViterbiPruningConfig()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("auto", "off", "required", "AUTO", " off "))
def test_supported_modes_normalize(mode):
    assert ViterbiPruningConfig(mode=mode).mode == mode.strip().lower()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="`mode` must be one of"):
        ViterbiPruningConfig(mode="enabled")


def test_non_string_mode_is_rejected():
    with pytest.raises(TypeError, match="`mode` must be a string"):
        ViterbiPruningConfig(mode=1)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="`strategy` must be one of"):
        ViterbiPruningConfig(strategy="octet_band")


def test_approximate_pruning_is_rejected_until_a_strategy_exists():
    with pytest.raises(ValueError, match="`exact=False` is rejected"):
        ViterbiPruningConfig(exact=False)


def test_non_boolean_exact_is_rejected():
    with pytest.raises(TypeError, match="`exact` must be boolean"):
        ViterbiPruningConfig(exact="yes")


def test_unknown_fallback_is_rejected():
    with pytest.raises(ValueError, match="`fallback` must be one of"):
        ViterbiPruningConfig(fallback="baseline_then_error")


@pytest.mark.parametrize("mode", ("off", "required"))
def test_non_auto_modes_must_leave_fallback_at_its_default(mode):
    """``off`` and ``required`` fully determine their own behavior, so pairing
    them with ``fallback='error'`` would be redundant or contradictory."""

    with pytest.raises(ValueError, match="`fallback` must stay at its default"):
        ViterbiPruningConfig(mode=mode, fallback="error")


@pytest.mark.parametrize(
    "kwargs,code",
    (
        ({}, VITERBI_PRUNING_AUTO),
        ({"fallback": "error"}, VITERBI_PRUNING_AUTO_ERROR),
        ({"mode": "off"}, VITERBI_PRUNING_OFF),
        ({"mode": "required"}, VITERBI_PRUNING_REQUIRED),
    ),
)
def test_every_legal_combination_maps_to_a_distinct_dispatch_code(kwargs, code):
    assert viterbi_pruning_dispatch_code(ViterbiPruningConfig(**kwargs)) == code


def test_policy_resolution_accepts_none_mappings_and_configs():
    assert resolve_viterbi_pruning_policy(None).mode == "auto"
    assert resolve_viterbi_pruning_policy({"mode": "off"}).mode == "off"
    assert resolve_viterbi_pruning_policy(ViterbiPruningConfig(mode="required")).mode == "required"
    resolved = resolve_viterbi_pruning_policy(ViterbiPruningConfig())
    assert resolve_viterbi_pruning_policy(resolved) is resolved


def test_policy_resolution_rejects_foreign_objects_and_keys():
    with pytest.raises(TypeError, match="must be a ViterbiPruningConfig"):
        resolve_viterbi_pruning_policy(object())
    with pytest.raises(ValueError, match="unexpected keys"):
        resolve_viterbi_pruning_policy({"mode": "auto", "band_width": 8})


# ---------------------------------------------------------------------------
# QVQConfig coercion
# ---------------------------------------------------------------------------


def test_dict_is_coerced_like_yaqa():
    config = _config(viterbi_pruning={"mode": "required"})
    assert isinstance(config.viterbi_pruning, ViterbiPruningConfig)
    assert config.viterbi_pruning.mode == "required"


def test_dict_coercion_validates():
    with pytest.raises(ValueError, match="`mode` must be one of"):
        _config(viterbi_pruning={"mode": "sometimes"})


def test_none_restores_the_automatic_default():
    assert _config(viterbi_pruning=None).viterbi_pruning == ViterbiPruningConfig()


def test_foreign_type_is_rejected():
    with pytest.raises(TypeError, match="`viterbi_pruning` must be a ViterbiPruningConfig"):
        _config(viterbi_pruning="required")


def test_instances_are_revalidated_on_config_construction():
    smuggled = ViterbiPruningConfig()
    smuggled.mode = "enabled"
    with pytest.raises(ValueError, match="`mode` must be one of"):
        _config(viterbi_pruning=smuggled)


# ---------------------------------------------------------------------------
# Serialization / round-trip / compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pruning",
    (
        ViterbiPruningConfig(),
        ViterbiPruningConfig(mode="off"),
        ViterbiPruningConfig(mode="required"),
        ViterbiPruningConfig(fallback="error"),
    ),
)
def test_nested_config_round_trips_through_from_quant_config(pruning):
    payload = _config(viterbi_pruning=pruning).to_dict()
    assert payload["viterbi_pruning"] == {
        "mode": pruning.mode,
        "strategy": pruning.strategy,
        "exact": pruning.exact,
        "fallback": pruning.fallback,
    }
    restored = QVQConfig.from_quant_config(dict(payload))
    assert restored.viterbi_pruning == pruning


def test_missing_field_reproduces_todays_automatic_behavior():
    """An older checkpoint has no `viterbi_pruning` key at all; it must
    deserialize to `auto`, which is exactly the pre-policy dispatch."""

    payload = _config().to_dict()
    payload.pop("viterbi_pruning")
    restored = QVQConfig.from_quant_config(payload)
    assert restored.viterbi_pruning == ViterbiPruningConfig()
    assert viterbi_pruning_dispatch_code(restored.viterbi_pruning) == VITERBI_PRUNING_AUTO


def test_unknown_keys_are_ignored_so_older_readers_stay_compatible():
    """Older readers ignore the new key; newer readers ignore keys they do not
    know. Neither direction changes the serialized weight layout."""

    payload = _config().to_dict()
    payload["viterbi_pruning_band_width"] = 8
    restored = QVQConfig.from_quant_config(payload)
    assert restored.viterbi_pruning == ViterbiPruningConfig()


def test_the_new_key_does_not_change_the_model_format_identity():
    with_default = _config().to_dict()
    with_off = _config(viterbi_pruning={"mode": "off"}).to_dict()
    # `meta` carries a per-process temporary offload path, so it is not comparable.
    ignored = {"viterbi_pruning", "meta"}
    assert {k: v for k, v in with_default.items() if k not in ignored} == {
        k: v for k, v in with_off.items() if k not in ignored
    }
    assert with_default["checkpoint_format"] == with_off["checkpoint_format"]
    assert with_default["quant_method"] == METHOD.QVQ


# ---------------------------------------------------------------------------
# B1: strict policies must fail at the outer Python dispatch guards too
# ---------------------------------------------------------------------------


def _cpu_banked_case(seed=20260910, batch=1, bits=3.0):
    import torch

    from gptqmodel.quantization.qvq_codecs import pgc16_codebook_v2_bank

    generator = torch.Generator().manual_seed(seed)
    sequences = torch.randn((batch, 128, 2), generator=generator, dtype=torch.float32)
    codebooks = torch.stack(
        (pgc16_codebook_v2_bank(0, bits=bits, dtype=torch.float32),
         pgc16_codebook_v2_bank(1, bits=bits, dtype=torch.float32))
    )
    return sequences, codebooks


@pytest.mark.parametrize(
    "policy",
    (
        pytest.param({"mode": "required"}, id="required"),
        pytest.param({"mode": "auto", "fallback": "error"}, id="auto-fallback-error"),
    ),
)
def test_strict_policy_rejects_cpu_eager_fallback_before_dispatch(policy):
    """A CPU call never reaches the native CUDA op, so its strict-policy
    refusal must come from the Python dispatch guard, not the kernel."""

    from gptqmodel.quantization.qvq import _batched_v2_banked_viterbi_quantize

    sequences, codebooks = _cpu_banked_case()
    with pytest.raises(RuntimeError, match="cannot use it: the call runs on device `cpu`"):
        _batched_v2_banked_viterbi_quantize(
            sequences,
            codebooks,
            bits=3.0,
            segment_steps=16,
            viterbi_pruning=ViterbiPruningConfig(**policy),
        )


@pytest.mark.parametrize(
    "policy",
    (
        pytest.param({"mode": "required"}, id="required"),
        pytest.param({"mode": "auto", "fallback": "error"}, id="auto-fallback-error"),
    ),
)
def test_strict_policy_rejects_cpu_tail_biting_wrapper(policy):
    """The public tail-biting wrapper threads the policy into the same guard."""

    from gptqmodel.quantization.qvq import tail_biting_v2b2_p32_quantize

    sequences, codebooks = _cpu_banked_case(seed=20260911)
    with pytest.raises(RuntimeError, match="cannot use it: the call runs on device `cpu`"):
        tail_biting_v2b2_p32_quantize(
            sequences,
            codebooks,
            bits=3.0,
            viterbi_pruning=ViterbiPruningConfig(**policy),
        )


@pytest.mark.parametrize(
    "policy",
    (
        pytest.param(None, id="default"),
        pytest.param({"mode": "auto", "fallback": "baseline"}, id="auto-baseline"),
        pytest.param({"mode": "off"}, id="off"),
    ),
)
def test_non_strict_policies_keep_the_cpu_eager_fallback(policy):
    """`auto`+`baseline` and `off` retain the exact eager baseline on CPU and
    agree with the policy-less call bit for bit."""

    import torch

    from gptqmodel.quantization.qvq import _batched_v2_banked_viterbi_quantize

    sequences, codebooks = _cpu_banked_case(seed=20260912)
    reference = _batched_v2_banked_viterbi_quantize(
        sequences, codebooks, bits=3.0, segment_steps=16
    )
    actual = _batched_v2_banked_viterbi_quantize(
        sequences,
        codebooks,
        bits=3.0,
        segment_steps=16,
        viterbi_pruning=None if policy is None else ViterbiPruningConfig(**policy),
    )
    assert torch.equal(reference.states, actual.states)
    assert torch.equal(reference.squared_error, actual.squared_error)
    assert torch.equal(reference.segment_bank_ids, actual.segment_bank_ids)
