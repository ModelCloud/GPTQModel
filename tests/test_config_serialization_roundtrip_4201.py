# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4201: config serialization must round-trip every accepted setting.

Four separate losses are covered:

* object-valued per-module `dynamic` overrides (`Fallback`, `MSEConfig`,
  `torch.dtype`, smoother objects) made `to_dict()`/`save_pretrained()` raise
  `TypeError` from `json.dumps`;
* GPTQ/QQQ/VPTQ/Paro dropped `preprocessors` because their `_update_meta_payload`
  never reached `PreProcessorConfig`;
* GPTQ wrote `foem` only when `gptaq` was set and never wrote `gptaq`, so both
  were lost on a direct `to_dict()`/`save_pretrained()` save;
* `QuantizeConfig(**config.to_dict())` ignored the `meta` payload, restoring
  defaults and turning an RTN GEMM config into AWQ.
"""

import json

import pytest
import torch

from gptqmodel.quantization.config import (
    AWQConfig,
    Fallback,
    FOEMConfig,
    GPTAQConfig,
    GPTQConfig,
    MSEConfig,
    ParoConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    SmoothMAD,
    VPTQConfig,
)


OBJECT_OVERRIDES = [
    {"fallback": Fallback(threshold="5%")},
    {"mse": MSEConfig(norm=2.0)},
    {"pack_dtype": torch.int16},
    {"smoother": SmoothMAD(k=2.0)},
]


def _json_round_trip(config):
    """Serialize like `save_pretrained` does and rebuild through the factory."""

    return QuantizeConfig(**json.loads(json.dumps(config.to_dict())))


@pytest.mark.parametrize("overrides", OBJECT_OVERRIDES, ids=lambda value: next(iter(value)))
def test_object_valued_dynamic_overrides_are_json_serializable(overrides):
    config = GPTQConfig(bits=4, dynamic={r".*down_proj": overrides})

    payload = json.loads(json.dumps(config.to_dict()))

    assert payload["dynamic"][r".*down_proj"]


@pytest.mark.parametrize("overrides", OBJECT_OVERRIDES, ids=lambda value: next(iter(value)))
def test_object_valued_dynamic_overrides_round_trip_through_the_factory(overrides):
    config = GPTQConfig(bits=4, dynamic={r".*down_proj": overrides})
    payload = json.loads(json.dumps(config.to_dict()))

    restored = QuantizeConfig(**payload)

    assert dict(restored.dynamic)[r".*down_proj"] == payload["dynamic"][r".*down_proj"]


def test_save_pretrained_writes_object_valued_overrides(tmp_path):
    config = GPTQConfig(
        bits=4,
        dynamic={
            r".*down_proj": {
                "fallback": Fallback(threshold="5%"),
                "mse": MSEConfig(norm=2.0),
                "pack_dtype": torch.int16,
                "smoother": SmoothMAD(k=2.0),
            }
        },
    )

    config.save_pretrained(str(tmp_path))

    written = json.loads((tmp_path / "quantize_config.json").read_text(encoding="utf-8"))
    assert written["dynamic"][r".*down_proj"]["pack_dtype"] == "int16"


@pytest.mark.parametrize(
    "config_cls",
    [GPTQConfig, QQQConfig, VPTQConfig, ParoConfig],
    ids=lambda cls: cls.__name__,
)
def test_gptq_family_keeps_smoother_preprocessors(config_cls):
    config = config_cls(smoother={"smooth": {"type": "mad", "k": 2.5}})
    assert config.smoother is not None

    restored = _json_round_trip(config)

    assert restored.smoother is not None
    assert restored.smoother.smooth.k == pytest.approx(2.5)


def test_foem_alone_is_serialized_and_reloaded():
    config = GPTQConfig(bits=4, foem=FOEMConfig(alpha=0.5))

    payload = config.to_dict()

    assert payload["meta"]["foem"]["alpha"] == pytest.approx(0.5)
    restored = _json_round_trip(config)
    assert restored.foem is not None
    assert restored.foem.alpha == pytest.approx(0.5)


def test_gptaq_and_foem_are_serialized_independently():
    config = GPTQConfig(
        bits=4,
        gptaq=GPTAQConfig(alpha=0.5),
        foem=FOEMConfig(alpha=0.25, beta=0.1),
    )

    payload = config.to_dict()

    assert payload["meta"]["gptaq"]["alpha"] == pytest.approx(0.5)
    assert payload["meta"]["foem"]["alpha"] == pytest.approx(0.25)
    restored = _json_round_trip(config)
    assert restored.gptaq is not None and restored.gptaq.alpha == pytest.approx(0.5)
    assert restored.foem is not None and restored.foem.beta == pytest.approx(0.1)


def test_vptq_serializes_sym():
    payload = VPTQConfig(bits=2).to_dict()

    assert payload["sym"] is True


def test_factory_lifts_meta_calibration_settings():
    config = GPTQConfig(
        bits=4,
        fallback="2%",
        damp_percent=0.1,
        mse=2.0,
        gptaq={"alpha": 0.5},
    )

    restored = QuantizeConfig(**config.to_dict())

    assert isinstance(restored, GPTQConfig)
    assert restored.fallback.threshold == "2%"
    assert restored.damp_percent == pytest.approx(0.1)
    assert restored.mse.norm == pytest.approx(2.0)
    assert restored.gptaq is not None
    assert restored.gptaq.alpha == pytest.approx(0.5)


def test_true_sequential_false_round_trips_through_both_reload_paths():
    """#4382: `true_sequential=False` must not reload as the class default."""

    config = GPTQConfig(bits=4, true_sequential=False)

    payload = config.to_dict()

    assert payload["meta"]["true_sequential"] is False
    assert QuantizeConfig(**payload).true_sequential is False
    assert QuantizeConfig.from_quant_config(payload).true_sequential is False


def test_true_sequential_default_true_round_trips_through_both_reload_paths():
    config = GPTQConfig(bits=4)
    assert config.true_sequential is True

    payload = config.to_dict()

    assert payload["meta"]["true_sequential"] is True
    assert QuantizeConfig(**payload).true_sequential is True
    assert QuantizeConfig.from_quant_config(payload).true_sequential is True


def test_factory_keeps_an_explicitly_disabled_fallback():
    restored = QuantizeConfig(**AWQConfig(bits=4, fallback=None).to_dict())

    assert isinstance(restored, AWQConfig)
    assert restored.fallback is None


def test_factory_keeps_rtn_instead_of_switching_to_awq():
    restored = QuantizeConfig(**RTNConfig(bits=4, format="gemm").to_dict())

    assert isinstance(restored, RTNConfig)
    assert restored.method.value == "gptq"
