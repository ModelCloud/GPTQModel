# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#5401: weight-only configs must not record a smoother they never apply.

``#5330`` rejects a smoother on the calibrated methods, but it returned early
for every weight-only lifecycle. Two cases still slipped through:

- ``fallback.smooth`` on RTN / GGUF / BitsAndBytes / NF12 / TT block-float: the
  weight-only packing path reads only the top-level ``smooth``, so the fallback
  smoother was persisted and never read (the ``#5302`` symptom);
- ``NF12Config(smoother=...)``: accepted at construction and rejected only
  inside ``pack_original``, after the model was loaded and the first layer was
  processed.

The tests below pin the construction-time rejection, the controls that must
keep working, and the reload path for a checkpoint written before the fix.
"""

from functools import partial

import pytest

from gptqmodel.quantization import config as C
from gptqmodel.quantization.config import (
    BitsAndBytesConfig,
    Fallback,
    FP8Config,
    GPTQConfig,
    NF12Config,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    SmoothMAD,
    SmoothMSE,
    TTBlockfloatConfig,
)


WEIGHT_ONLY_CONFIGS = [
    ("rtn", partial(RTNConfig, bits=4, group_size=128)),
    ("gguf", partial(C.GGUFConfig, bits=4, format="q_k_m")),
    ("bitsandbytes", BitsAndBytesConfig),
    ("fp8", FP8Config),
    ("nf12", NF12Config),
    ("tt-blockfloat", TTBlockfloatConfig),
]

#: Weight-only formats whose packing path applies the top-level `smooth`.
SMOOTHER_APPLYING = {"rtn", "gguf", "bitsandbytes"}

#: Weight-only formats with no smoothing step at all.
SMOOTHERLESS = {"fp8", "nf12", "tt-blockfloat"}


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in WEIGHT_ONLY_CONFIGS],
    ids=[name for name, _ in WEIGHT_ONLY_CONFIGS],
)
def test_weight_only_configs_reject_a_fallback_smoother(factory):
    """Only the GPTQ/QQQ fallback branch reads `fallback.smooth`."""

    with pytest.raises(ValueError, match="weight-only packing path reads only"):
        factory(fallback=Fallback(smooth=SmoothMSE()))


@pytest.mark.parametrize(
    "factory",
    [factory for name, factory in WEIGHT_ONLY_CONFIGS if name in SMOOTHERLESS],
    ids=sorted(SMOOTHERLESS),
)
def test_smootherless_weight_only_formats_reject_a_top_level_smoother(factory):
    """NF12/TT block-float/FP8 have no smoothing step; fail at construction."""

    with pytest.raises(ValueError, match="no weight-smoothing step"):
        factory(smoother=SmoothMAD(k=1.0))
    with pytest.raises(ValueError, match="no weight-smoothing step"):
        factory(smooth=SmoothMAD(k=1.0))


@pytest.mark.parametrize(
    "factory",
    [factory for name, factory in WEIGHT_ONLY_CONFIGS if name in SMOOTHER_APPLYING],
    ids=sorted(SMOOTHER_APPLYING),
)
def test_smoother_applying_weight_only_formats_keep_the_top_level_smoother(factory):
    """RTN/GGUF/BitsAndBytes do apply it, so the option must still work."""

    config = factory(smoother=SmoothMAD(k=1.25))

    assert config.smooth.k == pytest.approx(1.25)
    assert config.uses_weight_only_lifecycle() is True


def test_nf12_refuses_a_smoother_before_packing():
    """The old failure point was `NF12Linear.pack_original`, after a model load."""

    with pytest.raises(ValueError, match="NF12"):
        NF12Config(smoother=SmoothMAD(k=1.0))


def test_pre_fix_weight_only_checkpoint_drops_the_fallback_smoother():
    """A checkpoint that recorded one still loads and describes its weights."""

    payload = RTNConfig(bits=4, group_size=128).to_dict()
    payload["fallback"] = {"strategy": "rtn", "threshold": "0.5%", "smooth": {"type": "mse"}}

    restored = QuantizeConfig.from_quant_config(payload)

    assert isinstance(restored, RTNConfig)
    assert restored.fallback.smooth is None
    assert "smooth" not in restored.to_dict().get("fallback", {})


def test_pre_fix_nf12_checkpoint_drops_the_top_level_smoother():
    payload = NF12Config().to_dict()
    payload["smoother"] = {"type": "mad", "k": 1.0}

    restored = QuantizeConfig.from_quant_config(payload)

    assert isinstance(restored, NF12Config)
    assert restored.smoother is None
    assert "smoother" not in restored.to_dict()


def test_gptq_fallback_smoother_survives_the_reload():
    """Control: the fallback branch is the one consumer, so it must not be dropped."""

    payload = GPTQConfig(
        bits=4, group_size=128, fallback=Fallback(smooth=SmoothMAD(k=1.0))
    ).to_dict()

    restored = QuantizeConfig.from_quant_config(payload)

    assert restored.fallback.smooth.k == pytest.approx(1.0)


def test_qqq_fallback_smoother_is_still_accepted():
    config = QQQConfig(bits=4, group_size=128, fallback=Fallback(smooth=SmoothMSE()))

    assert isinstance(config.fallback.smooth, SmoothMSE)


def test_weight_only_dynamic_rule_rejects_a_fallback_smoother():
    """A per-module rule is merged at match time, so it must fail up front."""

    with pytest.raises(ValueError, match=r"dynamic\['model\.layers\.0'\]\.fallback\.smooth"):
        RTNConfig(
            bits=4,
            group_size=128,
            dynamic={"model.layers.0": {"fallback": {"smooth": {"type": "mse"}}}},
        )


def test_weight_only_dynamic_rule_keeps_the_top_level_smoother():
    """RTN applies `smoother`, including per-module overrides."""

    config = RTNConfig(
        bits=4,
        group_size=128,
        dynamic={"model.layers.0": {"smoother": {"type": "mad", "k": 1.0}}},
    )

    assert config.dynamic["model.layers.0"]["smoother"] == {"type": "mad", "k": 1.0}


def test_nf12_dynamic_rule_rejects_a_smoother():
    with pytest.raises(ValueError, match="no weight-smoothing step"):
        NF12Config(dynamic={"model.layers.0": {"smoother": {"type": "mad", "k": 1.0}}})


def test_gptq_dynamic_fallback_smoother_is_still_accepted():
    """Control: the GPTQ fallback branch reads it, so the rule stays valid."""

    config = GPTQConfig(
        bits=4,
        group_size=128,
        dynamic={"model.layers.0": {"fallback": {"smooth": {"type": "mse"}}}},
    )

    assert config.dynamic["model.layers.0"]["fallback"]["smooth"] == {"type": "mse"}


def test_pre_fix_weight_only_dynamic_fallback_smoother_is_dropped():
    payload = RTNConfig(bits=4, group_size=128).to_dict()
    payload["dynamic"] = {
        "model.layers.0": {"fallback": {"smooth": {"type": "mse"}}, "bits": 8}
    }

    restored = QuantizeConfig.from_quant_config(payload)

    assert isinstance(restored, RTNConfig)
    assert restored.dynamic["model.layers.0"] == {"bits": 8}
