# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The PyTorch safety gate must fail closed on an affected runtime.

GPT-QModel loads checkpoints and runs kernels through the imported PyTorch, so a
release affected by GHSA-rrmf-rvhw-rf47 / CVE-2025-3000 is unsupported. The gate
must run before CUDA initialization, and its only escape hatch is the explicit
``GPTQ_BYPASS_SAFETY_CHECKS`` opt-in.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from gptqmodel._torch_safety import (
    BYPASS_ENV,
    FIRST_SAFE_TORCH_VERSION,
    LAST_VULNERABLE_TORCH_VERSION,
    TORCH_ADVISORY_CVE,
    TORCH_ADVISORY_ID,
    UnsafeTorchVersionError,
    ensure_safe_torch,
    safety_checks_bypassed,
    torch_version_is_vulnerable,
)


@pytest.mark.parametrize(
    "version",
    [
        "1.13.1",
        "2.0.0",
        "2.6.0+cu124",
        "2.10.0",
        "2.11.0",
        "2.12.0+cu128",
        LAST_VULNERABLE_TORCH_VERSION,
        f"{LAST_VULNERABLE_TORCH_VERSION}+cu121",
        f"{LAST_VULNERABLE_TORCH_VERSION}+rocm6.4",
    ],
)
def test_affected_versions_are_rejected(version):
    assert torch_version_is_vulnerable(version) is True

    with pytest.raises(UnsafeTorchVersionError) as error:
        ensure_safe_torch(version, bypass=False)

    message = str(error.value)
    assert TORCH_ADVISORY_ID in message
    assert TORCH_ADVISORY_CVE in message
    assert FIRST_SAFE_TORCH_VERSION in message
    assert BYPASS_ENV in message
    assert "pip install --upgrade" in message


@pytest.mark.parametrize(
    "version",
    [
        FIRST_SAFE_TORCH_VERSION,
        f"{FIRST_SAFE_TORCH_VERSION}+cu130",
        "2.13.1",
        "2.14.1+cu128",
        "2.15.0",
        "2.16.0.dev20261102+cu130",
    ],
)
def test_patched_and_newer_versions_pass(version):
    assert torch_version_is_vulnerable(version) is False
    ensure_safe_torch(version, bypass=False)


def test_local_label_does_not_hide_the_affected_range():
    """``2.12.1+cu121`` sorts above ``2.12.1`` under PEP 440.

    A naive ``Version(raw) <= Version("2.12.1")`` misses every CUDA build in the
    affected range, which is exactly the population installing these wheels.
    """

    from packaging.version import Version

    assert Version(f"{LAST_VULNERABLE_TORCH_VERSION}+cu121") > Version(LAST_VULNERABLE_TORCH_VERSION)
    assert torch_version_is_vulnerable(f"{LAST_VULNERABLE_TORCH_VERSION}+cu121") is True


def test_unparseable_version_fails_closed():
    with pytest.raises(UnsafeTorchVersionError, match="cannot verify"):
        torch_version_is_vulnerable("not-a-version")

    with pytest.raises(UnsafeTorchVersionError, match="cannot verify"):
        ensure_safe_torch("main", bypass=False)


def test_bypass_flag_reads_only_truthy_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on", "y", " 1 "):
        monkeypatch.setenv(BYPASS_ENV, value)
        assert safety_checks_bypassed() is True, value

    for value in ("", "0", "false", "no", "off", "2", "disabled"):
        monkeypatch.setenv(BYPASS_ENV, value)
        assert safety_checks_bypassed() is False, value

    monkeypatch.delenv(BYPASS_ENV)
    assert safety_checks_bypassed() is False


def test_bypass_defaults_to_the_environment(monkeypatch):
    monkeypatch.setenv(BYPASS_ENV, "1")
    # No explicit ``bypass`` argument: the environment decides.
    ensure_safe_torch(LAST_VULNERABLE_TORCH_VERSION)

    monkeypatch.setenv(BYPASS_ENV, "0")
    with pytest.raises(UnsafeTorchVersionError):
        ensure_safe_torch(LAST_VULNERABLE_TORCH_VERSION)


def test_current_runtime_is_accepted():
    """This process imported GPT-QModel, so the gate accepted its own runtime."""

    assert torch_version_is_vulnerable(torch.__version__) is False


def _import_probe(version: str, env_extra: dict) -> subprocess.CompletedProcess:
    """Import ``gptqmodel`` in a fresh interpreter with a forced Torch version."""

    repo_root = Path(__file__).resolve().parents[1]
    script = (
        "import torch\n"
        f"torch.__version__ = {version!r}\n"
        "import gptqmodel\n"
        "print('IMPORTED')\n"
    )
    env = dict(os.environ)
    env.pop(BYPASS_ENV, None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_package_import_fails_before_cuda_initialization():
    """The gate must fire at import, before the allocator or a device is touched."""

    result = _import_probe(f"{LAST_VULNERABLE_TORCH_VERSION}+cu121", {})

    assert result.returncode != 0
    assert "UnsafeTorchVersionError" in result.stderr
    assert TORCH_ADVISORY_ID in result.stderr
    assert "IMPORTED" not in result.stdout
    # The startup banner is printed after the gate, so it must not appear.
    assert "GPT-QModel  :" not in result.stdout


def test_package_import_succeeds_with_the_explicit_bypass():
    result = _import_probe(f"{LAST_VULNERABLE_TORCH_VERSION}+cu121", {BYPASS_ENV: "1"})

    assert result.returncode == 0, result.stderr
    assert "IMPORTED" in result.stdout
