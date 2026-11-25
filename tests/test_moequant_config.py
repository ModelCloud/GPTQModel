# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""MoEQuant configuration, validation, and checkpoint provenance."""

from __future__ import annotations

import pytest

from gptqmodel import (
    AWQConfig,
    EBSSConfig,
    GPTQConfig,
    MoEQuantConfig,
    QuantizeConfig,
)
from gptqmodel.quantization import METHOD
from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig


pytestmark = [pytest.mark.cpu]


def test_ebss_defaults_and_validation():
    config = EBSSConfig()
    assert (config.num_sentences, config.seq_len, config.width) == (128, 512, 4)
    assert config.temperature == 1.2
    assert config.seed == 0

    for field_name, value in (
        ("num_sentences", 0),
        ("seq_len", 0),
        ("width", 0),
        ("seed", -1),
    ):
        with pytest.raises(ValueError, match=field_name):
            EBSSConfig(**{field_name: value})
    with pytest.raises(ValueError, match="temperature"):
        EBSSConfig(temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        EBSSConfig(temperature=float("nan"))


def test_moequant_requires_one_mechanism():
    assert MoEQuantConfig().agq is True
    assert MoEQuantConfig(agq=False, ebss=EBSSConfig()).ebss is not None
    with pytest.raises(ValueError, match="enable"):
        MoEQuantConfig(agq=False)
    with pytest.raises(TypeError, match="agq"):
        MoEQuantConfig(agq="yes")


def test_gptq_moequant_roundtrip_and_provenance():
    config = GPTQConfig(
        bits=4,
        group_size=128,
        moequant=MoEQuantConfig(
            ebss=EBSSConfig(num_sentences=2, seq_len=8, width=2, seed=3)
        ),
    )
    payload = config.to_dict()
    assert payload["meta"]["moequant"] == {
        "agq": True,
        "ebss": {
            "num_sentences": 2,
            "seq_len": 8,
            "width": 2,
            "temperature": 1.2,
            "seed": 3,
        },
    }

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, GPTQConfig)
    assert reloaded.moequant == config.moequant
    assert reloaded.is_quantized_by_moequant() is True
    assert GPTQConfig(bits=4, group_size=128).is_quantized_by_moequant() is False


def test_moequant_serialized_kwargs_roundtrip():
    config = GPTQConfig(
        bits=4,
        group_size=128,
        moequant=MoEQuantConfig(
            ebss=EBSSConfig(num_sentences=2, seq_len=8, width=2, seed=3)
        ),
    )
    rebuilt = QuantizeConfig(**config.to_dict())
    assert isinstance(rebuilt, GPTQConfig)
    assert rebuilt.moequant == config.moequant
    assert rebuilt.is_quantized_by_moequant() is True


def test_awq_moequant_is_ebss_only():
    config = AWQConfig(
        bits=4,
        group_size=128,
        moequant=MoEQuantConfig(
            agq=False, ebss=EBSSConfig(num_sentences=1, seq_len=2, width=1)
        ),
    )
    assert config.moequant.ebss is not None
    assert config.to_dict()["meta"]["moequant"]["agq"] is False

    with pytest.raises(ValueError, match="AWQ"):
        AWQConfig(bits=4, group_size=128, moequant=MoEQuantConfig())


def test_moequant_rejects_conflicting_calibrators():
    with pytest.raises(ValueError, match="gptaq"):
        GPTQConfig(bits=4, group_size=128, moequant=MoEQuantConfig(), gptaq=True)
    with pytest.raises(ValueError, match="foem"):
        GPTQConfig(bits=4, group_size=128, moequant=MoEQuantConfig(), foem=True)
    with pytest.raises(ValueError, match="g2ptq"):
        GPTQConfig(bits=4, group_size=128, moequant=MoEQuantConfig(), g2ptq=True)
    with pytest.raises(ValueError, match="mock_quantization"):
        GPTQConfig(
            bits=4,
            group_size=128,
            moequant=MoEQuantConfig(),
            mock_quantization=True,
        )


def test_moequant_rejects_w4a_activation():
    with pytest.raises(ValueError, match="activation"):
        GPTQConfig(
            bits=4,
            group_size=128,
            activation="w4afp8",
            moequant=MoEQuantConfig(),
        )


def test_moequant_is_rejected_for_other_methods():
    with pytest.raises(ValueError, match="moequant"):
        QuantizeConfig(
            bits=4,
            group_size=128,
            method=METHOD.QQQ,
            moequant=MoEQuantConfig(),
        )


def test_agq_rejects_routing_bypass():
    with pytest.raises(ValueError, match="native MoE routing"):
        GPTQConfig(
            bits=4,
            group_size=128,
            moequant=MoEQuantConfig(),
            moe=MoEConfig(routing=ExpertsRoutingBypass()),
        )
    # EBSS is data-side only and remains valid with bypass routing.
    config = GPTQConfig(
        bits=4,
        group_size=128,
        moequant=MoEQuantConfig(agq=False, ebss=EBSSConfig()),
        moe=MoEConfig(routing=ExpertsRoutingBypass()),
    )
    assert config.moequant.ebss is not None


def test_moequant_is_run_level_only():
    with pytest.raises(ValueError, match="per-module"):
        GPTQConfig(
            bits=4,
            group_size=128,
            dynamic={"model.layers.0": {"moequant": {"agq": True}}},
        )
