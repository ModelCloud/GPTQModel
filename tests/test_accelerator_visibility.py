# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Importing GPT-QModel must not change the caller's device isolation.

``ROCR_VISIBLE_DEVICES`` is the ROCr/HSA hardware mask. HIP filters agents by it
and then resolves ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` as
*positions* in that filtered list. Deleting the mask when ``CUDA_VISIBLE_DEVICES``
is also set therefore reinterprets those positions as physical ids, widens the
process past the caller's isolation, and leaks the wider environment to child
processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

import gptqmodel.utils.rocm as rocm_utils
from gptqmodel.utils.rocm import (
    VISIBILITY_MASK_ENV,
    check_rocm_visibility,
    warn_rocm_visibility_conflict,
)


MASK = VISIBILITY_MASK_ENV
CUDA = "CUDA_VISIBLE_DEVICES"
HIP = "HIP_VISIBLE_DEVICES"


@pytest.fixture(autouse=True)
def _reset_warned(monkeypatch):
    monkeypatch.setattr(rocm_utils, "_VISIBILITY_WARNED", False)


@pytest.fixture
def on_rocm(monkeypatch):
    """Pin the vendor probe so these cases assert HIP semantics on any host."""

    monkeypatch.setattr(rocm_utils, "IS_ROCM", True)


def _env(monkeypatch, **values: str | None) -> None:
    for name in (MASK, HIP, CUDA):
        monkeypatch.delenv(name, raising=False)
    for key, value in values.items():
        if value is not None:
            monkeypatch.setenv({"mask": MASK, "hip": HIP, "cuda": CUDA}[key], value)


def _issues():
    return {issue.name: issue for issue in check_rocm_visibility()}


# --- the reported regression -------------------------------------------------


def test_positions_inside_the_mask_are_accepted(on_rocm, monkeypatch):
    """ROCR=2,3 with CUDA=0,1 is the documented, consistent spelling."""

    _env(monkeypatch, mask="2,3", cuda="0,1")

    assert check_rocm_visibility() == ()
    assert os.environ[MASK] == "2,3"
    assert os.environ[CUDA] == "0,1"


def test_physical_ids_report_the_equivalent_positions(on_rocm, monkeypatch):
    """ROCR=2,3 with CUDA=2,3 names positions that do not exist."""

    _env(monkeypatch, mask="2,3", cuda="2,3")

    issue = _issues()[CUDA]
    assert issue.value == "2,3"
    assert issue.positional == "0,1"
    # Reported, never rewritten.
    assert os.environ[CUDA] == "2,3"
    assert os.environ[MASK] == "2,3"


def test_a_single_physical_id_maps_to_its_position(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3", cuda="3")

    assert _issues()[CUDA].positional == "1"
    assert os.environ[CUDA] == "3"


def test_reordering_is_preserved_on_rocm(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3", cuda="3,2")

    assert _issues()[CUDA].positional == "1,0"


def test_hip_visibility_uses_the_same_positional_rules(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3", hip="2,3")

    assert _issues()[HIP].positional == "0,1"
    assert os.environ[HIP] == "2,3"


# --- the mask is never modified ---------------------------------------------


@pytest.mark.parametrize(
    ("mask", "cuda"),
    [
        ("2,3", "0,1"),
        ("2,3", "2,3"),
        ("2,3", "0"),
        ("0,2,3", "1"),
        ("2", "0"),
        ("2,3", "0,2"),
    ],
)
def test_checking_never_mutates_the_environment(on_rocm, monkeypatch, mask, cuda):
    _env(monkeypatch, mask=mask, cuda=cuda)

    before = dict(os.environ)
    check_rocm_visibility()
    warn_rocm_visibility_conflict(logger=_Silent())
    after = dict(os.environ)

    assert after == before


# --- ambiguity and opaque selections ----------------------------------------


@pytest.mark.parametrize("cuda", ["0,2", "1,5", "2,9"])
def test_mixed_selections_have_no_positional_equivalent(on_rocm, monkeypatch, cuda):
    _env(monkeypatch, mask="2,3", cuda=cuda)

    issue = _issues()[CUDA]
    assert issue.value == cuda
    assert issue.positional is None
    assert os.environ[CUDA] == cuda


@pytest.mark.parametrize("cuda", ["GPU-1234", "MIG-GPU-1234/1/0", "2,,3", "-1"])
def test_opaque_selections_are_left_alone(on_rocm, monkeypatch, cuda):
    _env(monkeypatch, mask="2,3", cuda=cuda)

    assert check_rocm_visibility() == ()
    assert os.environ[CUDA] == cuda


def test_a_non_positional_mask_is_left_alone(on_rocm, monkeypatch):
    _env(monkeypatch, mask="GPU-1234", cuda="0,1")

    assert check_rocm_visibility() == ()
    assert os.environ[CUDA] == "0,1"


# --- inert off ROCm and without a mask --------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"cuda": "0,1"},
        {"hip": "0,1"},
        {"cuda": "2,3", "hip": "2,3"},
    ],
)
def test_is_inert_without_a_hardware_mask(on_rocm, monkeypatch, env):
    _env(monkeypatch, **env)

    assert check_rocm_visibility() == ()


def test_mask_alone_requires_nothing(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3")

    assert check_rocm_visibility() == ()


def test_mask_is_inert_on_a_cuda_build(monkeypatch):
    """A shared wrapper may export the ROCr mask on an NVIDIA host."""

    monkeypatch.setattr(rocm_utils, "IS_ROCM", False)
    _env(monkeypatch, mask="2,3", cuda="2,3")

    assert check_rocm_visibility() == ()
    assert os.environ[CUDA] == "2,3"


# --- warning behaviour -------------------------------------------------------


class _Silent:
    def __init__(self):
        self.messages: list[str] = []

    def warn(self, message: str) -> None:
        self.messages.append(message)


def test_warning_names_the_positional_alternative(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3", cuda="2,3")
    logger = _Silent()

    assert warn_rocm_visibility_conflict(logger=logger) is True

    assert len(logger.messages) == 1
    assert "CUDA_VISIBLE_DEVICES='2,3'" in logger.messages[0]
    assert "CUDA_VISIBLE_DEVICES='0,1'" in logger.messages[0]
    # The mask is never advertised as something to delete.
    assert "unset" not in logger.messages[0]


def test_ambiguous_warning_points_at_the_mask(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3", cuda="0,2")
    logger = _Silent()

    warn_rocm_visibility_conflict(logger=logger)

    assert len(logger.messages) == 1
    assert "0..1" in logger.messages[0]
    assert "unset CUDA_VISIBLE_DEVICES" in logger.messages[0]


def test_warning_is_emitted_once(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3", cuda="2,3")
    logger = _Silent()

    assert warn_rocm_visibility_conflict(logger=logger) is True
    assert warn_rocm_visibility_conflict(logger=logger) is False
    assert len(logger.messages) == 1


def test_no_warning_for_a_consistent_environment(on_rocm, monkeypatch):
    _env(monkeypatch, mask="2,3", cuda="0,1")
    logger = _Silent()

    assert warn_rocm_visibility_conflict(logger=logger) is False
    assert logger.messages == []


# --- end-to-end: the environment survives import ----------------------------

_PROBE = textwrap.dedent(
    """
    import json, os
    import gptqmodel  # noqa: F401

    print("PROBE:" + json.dumps({
        name: os.environ.get(name)
        for name in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
    }))
    """
)


def _run_import(env: dict[str, str]) -> dict[str, str | None]:
    child = {k: v for k, v in os.environ.items() if not k.endswith("VISIBLE_DEVICES")}
    child.update(env)
    child["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (os.getcwd(), child.get("PYTHONPATH")) if entry
    )
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=child,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    lines = [line for line in result.stdout.splitlines() if line.startswith("PROBE:")]
    assert lines, result.stdout[-2000:]
    return json.loads(lines[-1][len("PROBE:"):])


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_SKIP_SUBPROCESS_IMPORT_TEST") == "1",
    reason="subprocess import disabled in this environment",
)
def test_import_preserves_the_callers_visibility_environment():
    """The reported case: ROCR=2,3 with CUDA=0,1 keeps both variables."""

    env = _run_import({MASK: "2,3", CUDA: "0,1"})

    assert env[MASK] == "2,3"
    assert env[CUDA] == "0,1"
    assert env[HIP] is None


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_SKIP_SUBPROCESS_IMPORT_TEST") == "1",
    reason="subprocess import disabled in this environment",
)
def test_import_does_not_invent_a_visibility_selection():
    env = _run_import({})

    assert env[MASK] is None
    assert env[CUDA] is None
    assert env[HIP] is None


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_SKIP_SUBPROCESS_IMPORT_TEST") == "1",
    reason="subprocess import disabled in this environment",
)
def test_child_process_inherits_the_callers_mask():
    """A wider child environment is the actual leak the deletion caused."""

    script = textwrap.dedent(
        """
        import os, subprocess, sys
        import gptqmodel  # noqa: F401

        child = subprocess.run(
            [sys.executable, "-c",
             "import os; print('CHILD', os.environ.get('ROCR_VISIBLE_DEVICES'),"
             " os.environ.get('CUDA_VISIBLE_DEVICES'))"],
            capture_output=True, text=True, env=dict(os.environ),
        )
        print("PARENT:" + str((os.environ.get("ROCR_VISIBLE_DEVICES"),
                               os.environ.get("CUDA_VISIBLE_DEVICES"))))
        print(child.stdout.strip())
        """
    )
    child = {k: v for k, v in os.environ.items() if not k.endswith("VISIBLE_DEVICES")}
    child.update({MASK: "2,3", CUDA: "0,1"})
    child["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (os.getcwd(), child.get("PYTHONPATH")) if entry
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=child,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "PARENT:('2,3', '0,1')" in result.stdout
    assert "CHILD 2,3 0,1" in result.stdout
