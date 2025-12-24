# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import builtins
from importlib.metadata import PackageNotFoundError

import pytest

from gptqmodel.models import loader


class DummyRequirePkgModel:
    require_pkgs = ["fakepkg>=1.0.0"]


def test_check_versions_installs_missing_pkg_with_confirmation(monkeypatch):
    calls = {"version": 0, "run": []}

    def fake_version(pkg):
        calls["version"] += 1
        if calls["version"] == 1:
            raise PackageNotFoundError(pkg)
        return "1.0.0"

    def fake_run(cmd, check=False):
        calls["run"].append((cmd, check))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(loader, "version", fake_version)
    monkeypatch.setattr(loader.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _: "y")
    monkeypatch.setattr(loader.subprocess, "run", fake_run)

    loader.check_versions(DummyRequirePkgModel, DummyRequirePkgModel.require_pkgs)

    assert calls["run"]
    assert "pip" in calls["run"][0][0]
    assert "install" in calls["run"][0][0]


def test_check_versions_rejects_missing_pkg_without_confirmation(monkeypatch):
    def fake_version(pkg):
        raise PackageNotFoundError(pkg)

    monkeypatch.setattr(loader, "version", fake_version)
    monkeypatch.setattr(loader.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _: "n")

    with pytest.raises(ValueError, match="not installed"):
        loader.check_versions(DummyRequirePkgModel, DummyRequirePkgModel.require_pkgs)
