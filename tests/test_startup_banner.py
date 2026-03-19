# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "gptqmodel" / "_banner.py"
MODULE_SPEC = importlib.util.spec_from_file_location("gptqmodel_banner_test_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None

banner_module = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(banner_module)


def test_build_startup_banner_aligns_versions():
    banner = banner_module.build_startup_banner(
        "LOGO\n",
        gptqmodel_version="5.8.0",
        transformers_version="5.3.0",
        torch_version="2.10.0+cu130",
        triton_version="3.6.0",
    )

    lines = banner.splitlines()
    assert lines[0] == "LOGO"
    assert lines[1].strip().endswith("5.8.0")
    assert lines[2].strip().endswith("5.3.0")
    assert lines[3].strip().endswith("2.10.0+cu130")
    assert lines[4].strip().endswith("3.6.0")
    assert lines[1].startswith("GPT-QModel")
    assert lines[2].startswith("Transformers")
    assert lines[3].startswith("Torch")
    assert lines[4].startswith("Triton")
    assert {line.index(":") for line in lines[1:]} == {13}


def test_build_startup_banner_skips_missing_optional_versions():
    banner = banner_module.build_startup_banner(
        "LOGO\n",
        gptqmodel_version="5.8.0",
        transformers_version="5.3.0",
        torch_version="2.10.0+cu130",
    )

    assert "Triton version" not in banner
    assert "Triton" not in banner


def test_get_startup_banner_resolves_optional_versions(monkeypatch):
    def fake_resolve(package_names):
        if tuple(package_names) == banner_module.TRITON_PACKAGE_CANDIDATES:
            return "3.6.0"
        raise AssertionError(f"Unexpected package candidates: {package_names}")

    monkeypatch.setattr(
        banner_module,
        "resolve_installed_package_version",
        fake_resolve,
    )

    banner = banner_module.get_startup_banner(
        "LOGO\n",
        gptqmodel_version="5.8.0",
        transformers_version="5.3.0",
        torch_version="2.10.0+cu130",
    )

    assert any(line.startswith("Triton") and line.endswith("3.6.0") for line in banner.splitlines())
