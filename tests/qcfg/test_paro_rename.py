# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The Paro rename must publish `paro` and still load pre-rename checkpoints."""

from gptqmodel._quant_methods import LEGACY_PARO_QUANT_METHOD
from gptqmodel.quantization import FORMAT, METHOD, ParoConfig, QuantizeConfig


def test_paro_config_publishes_the_renamed_method_and_format():
    config = ParoConfig(bits=4, group_size=128)
    payload = config.to_dict()

    assert config.method is METHOD.PARO
    assert config.format is FORMAT.PARO
    assert payload["method"] == "paro"
    assert payload["format"] == "paro"
    assert payload["quant_method"] == "paro"


def test_legacy_paro_checkpoint_payload_still_loads():
    payload = ParoConfig(bits=4, group_size=128).to_dict()
    payload["method"] = LEGACY_PARO_QUANT_METHOD
    payload["format"] = LEGACY_PARO_QUANT_METHOD
    payload["quant_method"] = LEGACY_PARO_QUANT_METHOD

    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, ParoConfig)
    assert reloaded.method is METHOD.PARO
    assert reloaded.format is FORMAT.PARO
