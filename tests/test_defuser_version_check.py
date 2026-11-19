# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`defuser>=0.0.32` must be enforced where the conversion happens.

With an older release the routed experts of afmoe / flint / granitemoe_swa stay
fused 3-D tensors and are silently left unquantized, and the GLM-4.x defused MoE
blocks crash on the Transformers 5.22 router tuple; before 0.0.32 Defuser's
cache flush claims cuda:0 when converting a CPU or meta model (#5105). The
loader reports the installed version before it runs the conversion.
"""

from types import SimpleNamespace

import gptqmodel.models.loader as loader


def test_versions_below_the_floor_are_reported():
    for installed in ("0.0.28", "0.0.30", "0.0.31", "0.0.32rc1", "0.0.0"):
        assert loader.defuser_version_too_old(installed) == installed


def test_supported_versions_pass():
    for installed in ("0.0.32", "0.1.0", "1.0.0"):
        assert loader.defuser_version_too_old(installed) is None


def test_missing_distribution_is_not_reported(monkeypatch):
    def missing(name):
        raise loader.PackageNotFoundError(name)

    monkeypatch.setattr(loader, "version", missing)
    assert loader.defuser_version_too_old() is None


def test_check_reports_the_installed_version_once(monkeypatch):
    messages = []
    monkeypatch.setattr(loader, "version", lambda name: "0.0.28")
    monkeypatch.setattr(loader, "log", SimpleNamespace(error=lambda msg, *args: messages.append(msg % args)))
    monkeypatch.setattr(loader, "_defuser_version_warning_emitted", False)

    loader._check_defuser_version()
    loader._check_defuser_version()

    assert len(messages) == 1
    assert "0.0.28" in messages[0]
    assert loader.DEFUSER_MIN_VERSION in messages[0]


def test_supported_version_logs_nothing(monkeypatch):
    messages = []
    monkeypatch.setattr(loader, "version", lambda name: "0.0.32")
    monkeypatch.setattr(loader, "log", SimpleNamespace(error=lambda msg, *args: messages.append(msg)))
    monkeypatch.setattr(loader, "_defuser_version_warning_emitted", False)

    loader._check_defuser_version()

    assert messages == []


def test_conversion_path_runs_the_check(monkeypatch):
    calls = []
    monkeypatch.setattr(loader, "_check_defuser_version", lambda: calls.append(True))
    monkeypatch.setattr(loader.defuser, "convert_model", lambda model, cleanup_original: False)

    class _Definition:
        pass

    assert loader._convert_model_with_defuser(_Definition, object(), cleanup_original=False) is False
    assert calls == [True]
