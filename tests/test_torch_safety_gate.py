# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The PyTorch safety gate must fail closed on every affected release range.

GPT-QModel loads checkpoints and runs kernels through the imported PyTorch, so a
release covered by an advisory on that surface is unsupported. The gate must run
before CUDA initialization, name the specific advisory that matched, and only
yield to the explicit ``GPTQ_BYPASS_SAFETY_CHECKS`` opt-in.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from gptqmodel._torch_safety import (
    BYPASS_ENV,
    FIRST_SAFE_TORCH_VERSION,
    LAST_VULNERABLE_TORCH_VERSION,
    TORCH_ADVISORIES,
    TORCH_ADVISORY_CVE,
    TORCH_ADVISORY_ID,
    UnsafeTorchVersionError,
    ensure_safe_torch,
    matching_torch_advisories,
    safe_torch_releases_hint,
    safety_checks_bypassed,
    torch_version_is_vulnerable,
)


def _by_id(advisory_id):
    for item in TORCH_ADVISORIES:
        if item.advisory_id == advisory_id:
            return item
    raise AssertionError(f"unknown advisory {advisory_id!r}")


def _for_patched_version(patched):
    for item in TORCH_ADVISORIES:
        if item.patched_version() == patched:
            return item
    raise AssertionError(f"no advisory patched in {patched!r}")


# The two shipped advisories, resolved from the table so a rename cannot silently
# detach these tests from the policy.
JIT_SCRIPT_ADVISORY = _for_patched_version("2.13.0")
WEIGHTS_ONLY_ADVISORY = _for_patched_version("2.15.0")

# Releases inside each range, including local `+cu`/`+rocm` labels and a
# pre-release of the patched release, which predates the fix.
AFFECTED_VERSIONS = {
    JIT_SCRIPT_ADVISORY.advisory_id: [
        "1.13.1",
        "2.0.0",
        "2.6.0+cu124",
        "2.10.0",
        "2.11.0+cpu",
        "2.12.0+cu128",
        LAST_VULNERABLE_TORCH_VERSION,
        f"{LAST_VULNERABLE_TORCH_VERSION}+cu121",
        f"{LAST_VULNERABLE_TORCH_VERSION}+rocm6.4",
        f"{FIRST_SAFE_TORCH_VERSION}.dev2",
    ],
    WEIGHTS_ONLY_ADVISORY.advisory_id: [
        "2.14.0",
        "2.14.0.dev1",
        "2.14.1",
        "2.14.1+cu128",
        "2.14.1+cu121",
        f"{WEIGHTS_ONLY_ADVISORY.patched_version()}.dev1",
    ],
}

SAFE_VERSIONS = [
    FIRST_SAFE_TORCH_VERSION,
    f"{FIRST_SAFE_TORCH_VERSION}+cu130",
    "2.13.1",
    "2.13.9",
    WEIGHTS_ONLY_ADVISORY.patched_version(),
    f"{WEIGHTS_ONLY_ADVISORY.patched_version()}+cu130",
    "2.15.1",
    "2.16.0.dev20261102+cu130",
    "3.0.0",
]


@pytest.mark.parametrize(
    ("version", "advisory_id"),
    [
        (version, advisory_id)
        for advisory_id, versions in AFFECTED_VERSIONS.items()
        for version in versions
    ],
)
def test_affected_versions_are_rejected(version, advisory_id):
    advisory = _by_id(advisory_id)

    assert torch_version_is_vulnerable(version) is True
    assert [item.advisory_id for item in matching_torch_advisories(version)] == [advisory_id]

    with pytest.raises(UnsafeTorchVersionError) as error:
        ensure_safe_torch(version, bypass=False)

    message = str(error.value)
    # The message must name the advisory that actually matched, and only that one.
    assert advisory_id in message
    assert advisory.patched_version() in message
    assert advisory.url in message
    assert "pip install --upgrade" in message
    assert BYPASS_ENV in message
    assert safe_torch_releases_hint() in message
    for other in TORCH_ADVISORIES:
        if other.advisory_id != advisory_id:
            assert other.advisory_id not in message


@pytest.mark.parametrize("version", SAFE_VERSIONS)
def test_safe_versions_are_accepted(version):
    assert torch_version_is_vulnerable(version) is False
    assert matching_torch_advisories(version) == ()
    ensure_safe_torch(version, bypass=False)


def test_advisory_table_is_self_consistent():
    ids = [advisory.advisory_id for advisory in TORCH_ADVISORIES]
    assert len(set(ids)) == len(ids), "advisory ids must be unique"

    for advisory in TORCH_ADVISORIES:
        assert advisory.url.startswith("https://")
        assert advisory.cve_id is None or advisory.cve_id.startswith("CVE-")
        # The patch boundary must not be inside its own range.
        assert advisory.matches(Version(advisory.patched_version())) is False
        if advisory.introduced is not None:
            introduced = ".".join(str(part) for part in advisory.introduced)
            assert advisory.matches(Version(introduced)) is True


def test_affected_ranges_do_not_overlap():
    """No release may need two different upgrade targets."""

    for left in TORCH_ADVISORIES:
        for right in TORCH_ADVISORIES:
            if left is right:
                continue
            overlap_start = max(left.introduced or (0, 0, 0), right.introduced or (0, 0, 0))
            overlap_end = min(left.patched, right.patched)
            assert overlap_start >= overlap_end


def test_safe_release_hint_excludes_every_affected_line():
    hint = safe_torch_releases_hint()
    assert "2.13.x" in hint
    assert ">= 2.15.0" in hint
    # 2.14.x is affected by the weights_only advisory and must never be offered.
    assert "2.14" not in hint


def test_legacy_aliases_describe_the_jit_script_advisory():
    assert TORCH_ADVISORY_ID == JIT_SCRIPT_ADVISORY.advisory_id
    assert TORCH_ADVISORY_CVE == JIT_SCRIPT_ADVISORY.cve_id
    assert LAST_VULNERABLE_TORCH_VERSION == "2.12.1"
    assert FIRST_SAFE_TORCH_VERSION == JIT_SCRIPT_ADVISORY.patched_version()


def test_local_label_does_not_hide_the_affected_range():
    """A local label sorts above its base version under PEP 440."""

    assert Version(f"{LAST_VULNERABLE_TORCH_VERSION}+cu121") > Version(LAST_VULNERABLE_TORCH_VERSION)
    assert torch_version_is_vulnerable(f"{LAST_VULNERABLE_TORCH_VERSION}+cu121") is True
    assert Version("2.14.1+cu128") > Version("2.14.1")
    assert torch_version_is_vulnerable("2.14.1+cu128") is True


def test_unparseable_version_fails_closed():
    with pytest.raises(UnsafeTorchVersionError, match="cannot verify") as error:
        torch_version_is_vulnerable("not-a-version")
    assert safe_torch_releases_hint() in str(error.value)

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
    ensure_safe_torch("2.14.1")

    monkeypatch.setenv(BYPASS_ENV, "0")
    with pytest.raises(UnsafeTorchVersionError):
        ensure_safe_torch(LAST_VULNERABLE_TORCH_VERSION)


def test_current_runtime_is_accepted():
    """The imported runtime is safe, or was imported behind the override.

    The pinned CI matrices install a Torch below a patch boundary and set the
    documented override, so an unconditional "not vulnerable" assertion would
    fail them for the intended configuration (issue #3933).
    """

    if safety_checks_bypassed():
        pytest.skip("pinned matrix runs an affected Torch behind the documented override")

    assert torch_version_is_vulnerable(torch.__version__) is False


def test_imported_runtime_was_cleared_by_the_gate():
    """A vulnerable runtime may only have been imported with the override."""

    assert (not torch_version_is_vulnerable(torch.__version__)) or safety_checks_bypassed()


def test_gate_error_is_publicly_exported():
    import gptqmodel

    assert gptqmodel.UnsafeTorchVersionError is UnsafeTorchVersionError


def _import_probe(version, env_extra):
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
    assert UnsafeTorchVersionError.__name__ in result.stderr
    assert JIT_SCRIPT_ADVISORY.advisory_id in result.stderr
    assert "IMPORTED" not in result.stdout
    # The startup banner is printed after the gate, so it must not appear.
    assert "GPT-QModel  :" not in result.stdout


def test_package_import_fails_for_the_weights_only_advisory():
    """The 2.14.x range must be rejected at import, not only by the helper."""

    result = _import_probe("2.14.1+cu128", {})

    assert result.returncode != 0
    assert UnsafeTorchVersionError.__name__ in result.stderr
    assert WEIGHTS_ONLY_ADVISORY.advisory_id in result.stderr
    assert WEIGHTS_ONLY_ADVISORY.patched_version() in result.stderr
    assert JIT_SCRIPT_ADVISORY.advisory_id not in result.stderr
    assert "IMPORTED" not in result.stdout


def test_package_import_succeeds_between_the_advisory_ranges():
    """2.13.x clears the jit.script patch and predates the weights_only defect."""

    result = _import_probe(FIRST_SAFE_TORCH_VERSION, {})

    assert result.returncode == 0, result.stderr
    assert "IMPORTED" in result.stdout


def test_package_import_succeeds_with_the_explicit_bypass():
    result = _import_probe(f"{LAST_VULNERABLE_TORCH_VERSION}+cu121", {BYPASS_ENV: "1"})

    assert result.returncode == 0, result.stderr
    assert "IMPORTED" in result.stdout

def test_pinned_tenstorrent_runtime_agrees_with_its_installer():
    """The pinned TT Torch must be safe, or its installer must opt out explicitly.

    This is the cross-file contract for the Tenstorrent environment: bumping the
    manifest without updating the installer (or the reverse) breaks it.
    """

    repo = Path(__file__).resolve().parents[1]
    manifest = (repo / "requirements" / "tenstorrent-py312.txt").read_text()
    pins = [
        line.split("==", 1)[1].strip()
        for line in manifest.splitlines()
        if line.startswith("torch==")
    ]
    assert len(pins) == 1, f"expected exactly one torch pin, got {pins}"

    installer = (repo / "requirements" / "install-tenstorrent-py312.sh").read_text()
    if torch_version_is_vulnerable(pins[0]):
        assert f"{BYPASS_ENV}=1" in installer, "installer must export the documented override"
        assert "import gptqmodel" in installer, "installer must verify the import"
    else:
        assert f"{BYPASS_ENV}=" not in installer, "pinned Torch is safe; drop the override"


def test_python310_requirement_floor_avoids_every_range():
    """Python 3.10 has no patched wheel, so the floor must pick the safe 2.13.x line."""

    repo = Path(__file__).resolve().parents[1]
    requirement = (repo / "requirements.txt").read_text()
    lines = [
        line
        for line in requirement.splitlines()
        if line.startswith("torch>=") and 'python_version < "3.11"' in line
    ]
    assert len(lines) == 1, lines

    specifier = SpecifierSet(lines[0].split(";", 1)[0].removeprefix("torch"))
    assert specifier.contains("2.13.0")
    assert not specifier.contains(LAST_VULNERABLE_TORCH_VERSION)
    assert not specifier.contains("2.14.0")
    assert not specifier.contains("2.14.1")
