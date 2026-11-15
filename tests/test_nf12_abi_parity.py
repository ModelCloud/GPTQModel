# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The vendored NF12 copy must keep consuming the ABI upstream exposes.

``gptqmodel_nf12/abi.py`` pins that surface (C declarations, struct layouts,
status values, torch schemas, symbol map, public Python names) for one upstream
commit. These tests fail when the vendored tree drifts from it, and the only
sanctioned difference is an *addition* listed in :data:`EXTENSIONS`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import gptqmodel_nf12
from gptqmodel_nf12 import abi


ROOT = Path(__file__).resolve().parent.parent

# The C entry points this vendored tree adds on top of upstream. Everything else
# must match upstream exactly; adding a name here is an ABI change that needs a
# prebuilt library bump.
EXTENSIONS = frozenset(
    {
        "nf12_decode_fp32",
        "nf12_sub_fp32",
        "nf12_embedding_repeat4",
        "nf12_embedding_i64_repeat4",
        "nf12_linear_staged_b1",
        "nf12_linear_staged_b1_workspace_bytes",
    }
)


def test_vendored_header_matches_upstream_abi():
    abi.verify()


def test_python_surface_matches_upstream():
    abi.verify_python_surface(gptqmodel_nf12)


def test_vendored_copy_adds_only_documented_extensions():
    functions, _, _, _ = abi.parse_header(ROOT / abi.HEADER)
    assert set(functions) - set(abi.C_FUNCTIONS) == EXTENSIONS
    assert set(abi.C_FUNCTIONS) - set(functions) == set()


def test_pin_names_a_full_upstream_commit():
    assert len(abi.UPSTREAM_COMMIT) == 40
    assert all(character in "0123456789abcdef" for character in abi.UPSTREAM_COMMIT)


def test_package_version_tracks_the_pinned_header():
    major, minor = abi.UPSTREAM_VERSION
    assert gptqmodel_nf12.__version__.split(".")[:2] == [str(major), str(minor)]


def test_regenerator_reproduces_the_pin(tmp_path):
    source = os.environ.get("NF12_SOURCE")
    if not source:
        pytest.skip("set NF12_SOURCE to an nf12 checkout to check the regenerator")
    output = tmp_path / "abi.py"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_nf12_abi.py"), "--source", source, "--output", str(output)],
        check=True,
        capture_output=True,
    )
    assert output.read_text() == (ROOT / "gptqmodel_nf12" / "abi.py").read_text()
