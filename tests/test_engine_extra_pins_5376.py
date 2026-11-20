# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""#5376: engine extras must only advertise pins that intersect the core.

`pyproject.toml` advertised `vllm>=0.31.0` / `sglang[srt]>=0.5.21` on 3.10 and
`vllm>=0.32.0` / `sglang[srt]>=0.5.22` on 3.11+. Every one of those ranges
resolves to releases whose published Torch/Transformers pins fall outside
`requirements.txt`, so `pip install gptqmodel[vllm]` and
`pip install gptqmodel[sglang]` ended in "No solution found" before installing
anything (issue #5376).

This guard replays the resolver's decision offline against a snapshot of the
published engine metadata: for each supported interpreter, an advertised engine
must have at least one candidate release whose Torch and Transformers pins
intersect the core floors. `scripts/check_engine_extras.py` runs the same check
against live PyPI metadata.
"""

from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]

REPRESENTATIVE_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
CORE_PACKAGES = ("torch", "transformers")

# Snapshot of the PyPI metadata that decides these lanes (2026-11-19):
# `requires_python`, the exact Torch pin, and the Transformers range of the
# releases that bracket the core floors. Refresh it with any engine floor bump;
# `scripts/check_engine_extras.py` reports the live answer.
ENGINE_RELEASES = {
    "vllm": {
        "0.26.0": ("<3.15,>=3.10", "==2.11.0", ">=5.5.3"),
        "0.27.0": ("<3.15,>=3.10", "==2.13.0", ">=5.5.3"),
        "0.27.1": ("<3.15,>=3.10", "==2.13.0", ">=5.5.3"),
        "0.28.0": ("<3.15,>=3.10", "==2.13.0", ">=5.5.3"),
        "0.29.0": ("<3.15,>=3.10", "==2.13.0", ">=5.10.4"),
        "0.30.0": ("<3.15,>=3.10", "==2.13.0", ">=5.10.4"),
        "0.31.0": ("<3.15,>=3.10", "==2.13.0", "<5.18.0,>=5.10.4"),
        "0.32.0": ("<3.15,>=3.11", "==2.13.0", "<5.20.0,>=5.16.1"),
        "0.33.0": ("<3.15,>=3.11", "==2.14.0", "<5.21.0,>=5.16.1"),
        "0.34.0": ("<3.15,>=3.11", "==2.14.0", "<5.23.0,>=5.16.1"),
    },
    "sglang": {
        "0.5.19": (">=3.10", "==2.13.0", "==5.12.1"),
        "0.5.20": (">=3.10", "==2.13.0", "==5.12.1"),
        "0.5.21": (">=3.10", "==2.13.0", "==5.12.1"),
        "0.5.22": (">=3.11", "==2.14.1", "==5.19.0"),
        "0.5.23": (">=3.11", "==2.14.1", "==5.19.0"),
    },
}

# The declarations this issue reported. The guard must reject them, otherwise
# it cannot detect the defect it was written for.
PRE_FIX_EXTRAS = {
    "vllm": [
        'vllm>=0.31.0; python_version < "3.11"',
        'vllm>=0.32.0; python_version >= "3.11" and python_version < "3.15"',
    ],
    "sglang": [
        'sglang[srt]>=0.5.21; python_version < "3.11"',
        'sglang[srt]>=0.5.22; python_version >= "3.11"',
    ],
}


def _environment(python_version):
    return {
        "python_version": python_version,
        "python_full_version": python_version + ".0",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "os_name": "posix",
    }


def core_floors():
    """`requirements.txt` Torch/Transformers specifier per interpreter."""

    floors = {version: {} for version in REPRESENTATIVE_PYTHONS}
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.name not in CORE_PACKAGES:
            continue
        for version in REPRESENTATIVE_PYTHONS:
            if requirement.marker is None or requirement.marker.evaluate(
                _environment(version)
            ):
                floors[version][requirement.name] = requirement.specifier
    return floors


def _bounds(specifier):
    """Exact pin or (lower, upper) interval of a simple specifier set."""

    exact = None
    lower = upper = None
    lower_inclusive = upper_inclusive = True
    for item in specifier:
        version = Version(item.version)
        if item.operator == "==":
            exact = version
        elif item.operator == ">=":
            lower, lower_inclusive = version, True
        elif item.operator == ">":
            lower, lower_inclusive = version, False
        elif item.operator == "<=":
            upper, upper_inclusive = version, True
        elif item.operator == "<":
            upper, upper_inclusive = version, False
        else:
            raise AssertionError("unsupported specifier %r" % (item,))
    return exact, lower, lower_inclusive, upper, upper_inclusive


def intersects(left, right):
    """True when some version satisfies both specifier sets."""

    left_exact, left_low, left_low_inc, left_high, left_high_inc = _bounds(left)
    right_exact, right_low, right_low_inc, right_high, right_high_inc = _bounds(right)
    if left_exact is not None:
        return right.contains(left_exact, prereleases=True)
    if right_exact is not None:
        return left.contains(right_exact, prereleases=True)
    low, low_inclusive = left_low, left_low_inc
    if right_low is not None and (
        low is None or right_low > low or (right_low == low and not right_low_inc)
    ):
        low, low_inclusive = right_low, right_low_inc
    high, high_inclusive = left_high, left_high_inc
    if right_high is not None and (
        high is None or right_high < high or (right_high == high and not right_high_inc)
    ):
        high, high_inclusive = right_high, right_high_inc
    if low is None or high is None:
        return True
    if low < high:
        return True
    return low == high and low_inclusive and high_inclusive


def candidates(requirement, python_version):
    """Published releases matching the specifier and running on the interpreter."""

    interpreter = Version(python_version + ".0")
    releases = ENGINE_RELEASES.get(requirement.name, {})
    return {
        version: pins
        for version, pins in releases.items()
        if Version(version) in requirement.specifier
        and interpreter in SpecifierSet(pins[0])
    }


def unsatisfied(extras, python_version):
    """Advertised engine requirements with no release inside the core floors."""

    floors = core_floors()[python_version]
    problems = []
    for extra, requirements in extras.items():
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.name not in ENGINE_RELEASES:
                continue
            if requirement.marker is not None and not requirement.marker.evaluate(
                _environment(python_version)
            ):
                continue
            releases = candidates(requirement, python_version)
            if not releases:
                problems.append(
                    "%s on %s: no %s release matches %s on this interpreter"
                    % (extra, python_version, requirement.name, requirement.specifier)
                )
                continue
            compatible = [
                version
                for version, (_, torch_pin, transformers_spec) in releases.items()
                if intersects(SpecifierSet(torch_pin), floors["torch"])
                and intersects(SpecifierSet(transformers_spec), floors["transformers"])
            ]
            if not compatible:
                problems.append(
                    "%s on %s: %s%s pins Torch/Transformers outside the core floors"
                    % (
                        extra,
                        python_version,
                        requirement.name,
                        requirement.specifier,
                    )
                )
    return problems


def _advertised_extras():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_core_floors_cover_every_interpreter():
    floors = core_floors()
    for version in REPRESENTATIVE_PYTHONS:
        assert set(floors[version]) == set(CORE_PACKAGES), (version, floors[version])


def test_advertised_engine_extras_resolve_on_every_interpreter():
    extras = _advertised_extras()
    problems = [
        problem
        for version in REPRESENTATIVE_PYTHONS
        for problem in unsatisfied(extras, version)
    ]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("python_version", REPRESENTATIVE_PYTHONS)
def test_pre_fix_engine_extras_are_rejected(python_version):
    """Red evidence: the #5376 declarations must fail this check."""

    assert unsatisfied(PRE_FIX_EXTRAS, python_version), (
        "the pre-fix extras were accepted on %s" % python_version
    )


def test_known_compatible_release_is_accepted():
    """Positive control: a combination that resolves must not be rejected."""

    extras = {"vllm": ['vllm>=0.30.0,<0.31; python_version < "3.11"']}
    assert unsatisfied(extras, "3.10") == []
