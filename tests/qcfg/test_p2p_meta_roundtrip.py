# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

# GPU=-1

import pytest

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.p2p import P2PConfig, configure_p2p


@pytest.fixture(autouse=True)
def _restore_p2p_policy():
    try:
        yield
    finally:
        configure_p2p(None)


def test_quantize_config_round_trips_p2p_mode_from_meta():
    cfg = QuantizeConfig(p2p={"mode": "off"}, offload_to_disk=False)

    payload = cfg.to_dict()
    assert payload["meta"]["p2p"]["mode"] == "off"

    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded.p2p, P2PConfig)
    assert reloaded.p2p.mode == "off"
    assert configure_p2p(reloaded.p2p).mode == "off"
    assert reloaded.to_dict()["meta"]["p2p"] == reloaded.p2p.to_dict()


def test_quantize_config_round_trips_full_p2p_policy():
    policy = P2PConfig(
        mode="on",
        probe=True,
        probe_mib=7,
        min_gbps=12.0,
        min_ratio=2.0,
        use_torch_replicate=True,
        use_rocm_nested_peer_copy=True,
    )
    cfg = QuantizeConfig(p2p=policy, offload_to_disk=False)

    reloaded = QuantizeConfig.from_quant_config(cfg.to_dict())

    assert reloaded.p2p == policy


def test_quantize_config_p2p_survives_save_pretrained(tmp_path):
    cfg = QuantizeConfig(p2p={"mode": "off", "probe": True, "min_gbps": 9.5}, offload_to_disk=False)

    cfg.save_pretrained(tmp_path)
    loaded = QuantizeConfig.from_pretrained(tmp_path)

    assert isinstance(loaded.p2p, P2PConfig)
    assert loaded.p2p.mode == "off"
    assert loaded.p2p.probe is True
    assert loaded.p2p.min_gbps == 9.5


def test_top_level_p2p_overrides_stale_meta_policy():
    cfg = QuantizeConfig(p2p={"mode": "off"}, offload_to_disk=False)
    payload = cfg.to_dict()

    restored = QuantizeConfig.from_quant_config({**payload, "p2p": {"mode": "on"}})

    assert restored.p2p.mode == "on"


def test_quantize_config_without_p2p_keeps_default_none():
    cfg = QuantizeConfig(offload_to_disk=False)

    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert cfg.p2p is None
    assert payload["meta"]["p2p"] is None
    assert reloaded.p2p is None
