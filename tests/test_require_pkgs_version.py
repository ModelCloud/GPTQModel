# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from importlib.metadata import PackageNotFoundError

import pytest

from gptqmodel.models import loader


class DummyModel:
    pass


def test_check_versions_accepts_satisfied_requirements(monkeypatch):
    def fake_version(pkg):
        return {"transformers": "4.44.2"}[pkg]

    monkeypatch.setattr(loader, "version", fake_version)

    loader.check_versions(DummyModel, ["transformers<=4.44.2"])


def test_check_versions_rejects_unsatisfied_requirements(monkeypatch):
    monkeypatch.setattr(loader, "version", lambda pkg: "4.50.0")

    with pytest.raises(ValueError, match="requires version transformers<=4.44.2"):
        loader.check_versions(DummyModel, ["transformers<=4.44.2"])


def test_check_versions_rejects_missing_package(monkeypatch):
    def fake_version(pkg):
        raise PackageNotFoundError(pkg)

    monkeypatch.setattr(loader, "version", fake_version)

    with pytest.raises(ValueError, match="not installed"):
        loader.check_versions(DummyModel, ["retention>=1.0.7"])


def test_check_versions_handles_multiple_requirements(monkeypatch):
    def fake_version(pkg):
        versions = {"transformers": "4.38.2", "tokenizers": "0.15.2"}
        return versions[pkg]

    monkeypatch.setattr(loader, "version", fake_version)

    loader.check_versions(DummyModel, ["transformers<=4.38.2", "tokenizers<=0.15.2"])
