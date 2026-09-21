# SPDX-License-Identifier: Apache-2.0

import torch

import gptqmodel.utils.exllamav2 as exllamav2


def _select(path, m, **kwargs):
    return exllamav2.select_exllamav2_path(
        path=path,
        m=m,
        n=2048,
        k=2048,
        device=torch.device("cpu"),
        group_size=128,
        desc_act=False,
        sym=True,
        **kwargs,
    )


def test_legacy_keeps_the_m_50_boundary():
    assert _select("legacy", 49) == "fused"
    assert _select("legacy", 50) == "fused"
    assert _select("legacy", 51) == "dense"


def test_explicit_paths_and_force_cuda_priority():
    assert _select("fused", 2048) == "fused"
    assert _select("dense", 1) == "dense"
    assert _select("dense", 1, force_cuda=True) == "fused"
    assert _select("auto", 1, force_cuda=True) == "fused"


def test_auto_unknown_device_shape_falls_back_to_legacy():
    assert _select("auto", 50) == "fused"
    assert _select("auto", 51) == "dense"


def test_auto_rule_matches_all_layout_metadata(monkeypatch):
    metadata = ("test-gpu", (9, 0), 120, 2)
    rule = {
        "device_name": "test-gpu",
        "capability": (9, 0),
        "sm_count": 120,
        "device_count": 2,
        "n": 2048,
        "k": 2048,
        "group_size": 64,
        "desc_act": True,
        "sym": False,
        "layout": "gptq4:qzero1",
        "m_ranges": ((49, 52, "dense"),),
    }
    monkeypatch.setattr(exllamav2, "EXLLAMAV2_MEASURED_RULES", (rule,))

    assert exllamav2.select_exllamav2_path(
        path="auto",
        m=50,
        n=2048,
        k=2048,
        device=torch.device("cuda:0"),
        group_size=64,
        desc_act=True,
        sym=False,
        layout="gptq4:qzero1",
        device_metadata=metadata,
    ) == "dense"

    # A changed layout fact must not reuse the measured rule.
    assert exllamav2.select_exllamav2_path(
        path="auto",
        m=50,
        n=2048,
        k=2048,
        device=torch.device("cuda:0"),
        group_size=128,
        desc_act=True,
        sym=False,
        layout="gptq4:qzero1",
        device_metadata=metadata,
    ) == "fused"


def test_invalid_path_is_rejected():
    try:
        _select("not-a-path", 1)
    except ValueError as exc:
        assert "legacy" in str(exc)
    else:
        raise AssertionError("invalid execution path was accepted")
