# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Installed NF12 package checks before accepting its Torch operators."""

import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("available_ops", [(), ("encode",), ("encode", "decode")])
def test_installed_nf12_requires_all_runtime_operators(monkeypatch, available_ops):
    from gptqmodel.utils import nf12_ad

    package = ModuleType("nf12")
    package.__file__ = "/tmp/incompatible-nf12-package/__init__.py"
    package.load = lambda: None
    monkeypatch.setitem(sys.modules, "nf12", package)
    namespace = SimpleNamespace(**{name: lambda: None for name in available_ops})
    monkeypatch.setattr(nf12_ad, "torch", SimpleNamespace(ops=SimpleNamespace(nf12=namespace)))

    assert nf12_ad._load_installed() is False


def test_installed_nf12_accepts_complete_runtime_operators(monkeypatch):
    from gptqmodel.utils import nf12_ad

    package = ModuleType("nf12")
    package.__file__ = "/tmp/compatible-nf12-package/__init__.py"
    package.load = lambda: None
    monkeypatch.setitem(sys.modules, "nf12", package)
    namespace = SimpleNamespace(encode=lambda: None, decode=lambda: None, linear=lambda: None)
    monkeypatch.setattr(nf12_ad, "torch", SimpleNamespace(ops=SimpleNamespace(nf12=namespace)))

    assert nf12_ad._load_installed() is True


def test_installed_nf12_rejects_noncallable_operator(monkeypatch):
    from gptqmodel.utils import nf12_ad

    package = ModuleType("nf12")
    package.__file__ = "/tmp/incompatible-nf12-package/__init__.py"
    package.load = lambda: None
    monkeypatch.setitem(sys.modules, "nf12", package)
    namespace = SimpleNamespace(encode=lambda: None, decode=object(), linear=lambda: None)
    monkeypatch.setattr(nf12_ad, "torch", SimpleNamespace(ops=SimpleNamespace(nf12=namespace)))

    assert nf12_ad._load_installed() is False
