# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

# GPU=-1

import pytest

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.p2p import P2PConfig, configure_p2p, get_p2p_config


@pytest.fixture(autouse=True)
def _restore_p2p_policy():
    previous = get_p2p_config()
    try:
        yield
    finally:
        configure_p2p(previous)


def test_quantize_config_round_trips_p2p_mode_from_meta():
    cfg = QuantizeConfig(p2p={"mode": "off"}, offload_to_disk=False)

    payload = cfg.to_dict()
    assert payload["meta"]["p2p"]["mode"] == "off"

    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded.p2p, P2PConfig)
    assert reloaded.p2p.mode == "off"
    assert configure_p2p(reloaded.p2p).mode == "off"
    assert reloaded.to_dict()["meta"]["p2p"] == reloaded.p2p.to_dict()


@pytest.mark.parametrize("as_dict", [False, True], ids=["config", "dict"])
def test_quantize_config_round_trips_full_p2p_policy(as_dict):
    policy = P2PConfig(
        mode="on",
        probe=True,
        probe_mib=7,
        min_gbps=12.0,
        min_ratio=2.0,
        use_torch_replicate=True,
        use_rocm_nested_peer_copy=True,
    )
    cfg = QuantizeConfig(p2p=policy.to_dict() if as_dict else policy, offload_to_disk=False)

    reloaded = QuantizeConfig.from_quant_config(cfg.to_dict())

    assert reloaded.p2p == policy


@pytest.mark.parametrize("as_dict", [False, True], ids=["config", "dict"])
def test_quantize_config_p2p_survives_save_pretrained(tmp_path, as_dict):
    policy = P2PConfig(
        mode="off",
        probe=True,
        probe_mib=7,
        min_gbps=9.5,
        min_ratio=1.5,
        use_torch_replicate=True,
        use_rocm_nested_peer_copy=True,
    )
    cfg = QuantizeConfig(p2p=policy.to_dict() if as_dict else policy, offload_to_disk=False)

    cfg.save_pretrained(tmp_path)
    loaded = QuantizeConfig.from_pretrained(tmp_path)

    assert isinstance(loaded.p2p, P2PConfig)
    assert loaded.p2p == policy
    assert loaded.to_dict()["meta"]["p2p"] == policy.to_dict()


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"mode": "on"}, P2PConfig(mode="on")),
        (P2PConfig(mode="on"), P2PConfig(mode="on")),
        ({}, P2PConfig()),
        (None, None),
    ],
    ids=["dict", "config", "empty-dict", "none"],
)
def test_top_level_p2p_overrides_stale_meta_policy(override, expected):
    cfg = QuantizeConfig(p2p={"mode": "off"}, offload_to_disk=False)
    payload = cfg.to_dict()

    restored = QuantizeConfig.from_quant_config({**payload, "p2p": override})

    assert restored.p2p == expected
    assert restored.to_dict()["meta"]["p2p"] == (expected.to_dict() if expected is not None else None)
    assert payload["meta"]["p2p"]["mode"] == "off"


@pytest.mark.parametrize("missing", [False, True], ids=["null", "legacy-missing"])
def test_quantize_config_without_p2p_keeps_default_none(missing):
    cfg = QuantizeConfig(offload_to_disk=False)
    payload = cfg.to_dict()
    assert payload["meta"]["p2p"] is None
    if missing:
        del payload["meta"]["p2p"]

    reloaded = QuantizeConfig.from_quant_config(payload)

    assert cfg.p2p is None
    assert reloaded.p2p is None


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("off", "must be a P2PConfig, dict, or None"),
        ({"mode": "invalid"}, "P2PConfig.mode must be"),
        ({"probe_mib": "invalid"}, "P2PConfig.probe_mib must be"),
    ],
    ids=["invalid-type", "invalid-mode", "invalid-probe-size"],
)
def test_quantize_config_rejects_invalid_meta_p2p(policy, message):
    payload = QuantizeConfig(offload_to_disk=False).to_dict()
    payload["meta"]["p2p"] = policy

    with pytest.raises(ValueError, match=message):
        QuantizeConfig.from_quant_config(payload)
