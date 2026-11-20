# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""#5420: the 3.14t unit lane must not resolve cp313-only wheels.

``deps.yaml`` pins a vLLM and a causal-conv1d wheel tagged
``cp313-cp313-linux_x86_64`` for ``test_eval``, ``test_nemotron_ultra`` and
``test_hymba``. Those wheels cannot install on the free-threaded 3.14 lane, so
the install step failed after a job and a GPU had already been scheduled.

The three tests are now gated to a 3.13 lane in ``test.yaml`` (``py: 3.13``),
the planner drops a test whose ``py`` names another lane, and
``ci_deps.py check-tags`` fails the plan before any test job starts when a
scheduled test still resolves a foreign wheel.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / ".github/scripts"
DEPS_SCRIPT = SCRIPTS / "ci_deps.py"
PLAN_SCRIPT = SCRIPTS / "ci_workflow.py"
TEST_YAML = SCRIPTS / "test.yaml"
UNIT_WORKFLOW = REPO_ROOT / ".github/workflows/unit_tests.yml"

# The lane the reusable unit workflow installs.
LANE = "3.14.10t"
# The lane these three need: every pinned wheel is cp313.
GATE_LANE = "3.13"

GATED_TESTS = {
    "test_eval": "tests/test_eval.py",
    "models/test_nemotron_ultra": "tests/models/test_nemotron_ultra.py",
    "models/test_hymba": "tests/models/test_hymba.py",
}

VLLM_WHEEL = (
    "http://10.0.13.31/files/"
    "vllm-0.17.2rc1.dev52+g86b7e3c95.d20260318.cu131-cp313-cp313-linux_x86_64.whl"
)
CAUSAL_CONV1D_WHEEL = (
    "http://10.0.13.31/files/causal_conv1d-1.6.0-cp313-cp313-linux_x86_64.whl"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ci_deps():
    return _load_module(DEPS_SCRIPT, "ci_deps_5420")


@pytest.fixture(scope="module")
def planner():
    return _load_module(PLAN_SCRIPT, "ci_workflow_5420")


@pytest.fixture(scope="module")
def deps_manifest(ci_deps):
    return ci_deps.load_yaml("deps.yaml")


def _specific_pkgs(ci_deps, manifest, name):
    specific, _common = ci_deps.collect_pkgs(
        ci_deps.resolve_test_path(name), manifest, dedupe_common=False
    )
    return specific


def _plan(planner, python_version):
    return planner.build_test_plan(
        ignored_test_files="",
        test_names="",
        test_regex=".*",
        tests_root=REPO_ROOT / "tests",
        python_version=python_version,
    )


def _scheduled(plan):
    return {
        entry["test_script"]
        for group in ("cpu", "torch", "model")
        for entry in plan[f"{group}_matrix"]
    }


def test_the_three_tests_still_pin_wheels_that_314t_cannot_install(ci_deps, deps_manifest):
    """The gate is only correct while these wheels really are cp313-only."""

    for name, relative in GATED_TESTS.items():
        assert (REPO_ROOT / relative).is_file(), f"{name} moved; update the gate"
        incompatible = ci_deps.incompatible_direct_wheels(
            _specific_pkgs(ci_deps, deps_manifest, name), LANE
        )
        assert incompatible, (
            f"{name} no longer resolves a wheel the {LANE} lane cannot install; "
            "drop its `py: 3.13` gate and this regression test"
        )
        assert all("cp313" in " ".join(tags) for _spec, tags in incompatible)


def test_the_gate_target_installs_every_pinned_wheel(ci_deps, deps_manifest):
    """A 3.13 lane is the right home: all three entries resolve there."""

    for name in GATED_TESTS:
        assert ci_deps.incompatible_direct_wheels(
            _specific_pkgs(ci_deps, deps_manifest, name), GATE_LANE
        ) == []


@pytest.mark.parametrize(
    ("lane", "abi", "foreign_abi"),
    [("3.13", "cp313", "cp313t"), ("3.13t", "cp313t", "cp313")],
    ids=["regular", "free-threaded"],
)
def test_the_target_abi_comes_from_the_requested_lane(ci_deps, lane, abi, foreign_abi):
    """#5615: a free-threaded host must not leak its ``t`` ABI into another lane."""

    tags = ci_deps.interpreter_tags(lane)
    assert any(tag.startswith(f"cp313-{abi}-") for tag in tags), sorted(tags)
    assert not any(tag.startswith(f"cp313-{foreign_abi}-") for tag in tags), sorted(tags)


def test_planner_gates_the_three_off_the_314t_lane(planner):
    plan = _plan(planner, LANE)
    scheduled = _scheduled(plan)
    gated = plan["gated_tests"]

    for name in GATED_TESTS:
        assert name not in scheduled, f"{name} is still scheduled on {LANE}"
        assert name in gated, f"{name} disappeared instead of being reported as gated"
        assert GATE_LANE in gated[name] and LANE in gated[name]


def test_planner_schedules_them_on_the_313_lane(planner):
    plan = _plan(planner, GATE_LANE)
    scheduled = _scheduled(plan)

    for name in GATED_TESTS:
        assert name in scheduled, f"{name} is not routed to the {GATE_LANE} lane"
        assert name not in plan["gated_tests"]


def test_the_314t_plan_resolves_no_foreign_wheel(ci_deps, planner):
    plan = _plan(planner, LANE)
    assert ci_deps.check_direct_wheels(sorted(_scheduled(plan)), python_version=LANE) == []


def test_the_manifest_audit_still_finds_exactly_the_three_gated_entries(ci_deps, deps_manifest):
    """`check-tags --all` keeps the gate honest if a new foreign wheel lands."""

    failures = ci_deps.check_direct_wheels(
        ci_deps._manifest_test_names(deps_manifest), python_version=LANE
    )
    wheels = {failure.split(": ", 1)[1].split(" declares")[0] for failure in failures}
    assert wheels == {VLLM_WHEEL, CAUSAL_CONV1D_WHEEL}
    assert len(failures) == 3


def test_workflow_checks_wheel_tags_before_scheduling_jobs():
    workflow = yaml.safe_load(UNIT_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["list-test-files"]["steps"]
    names = [step.get("name") for step in steps]

    list_index = names.index("List files")
    check_index = names.index("Check resolved wheel tags")
    assert list_index < check_index

    list_run = steps[list_index]["run"]
    assert "--python-version" in list_run and LANE in list_run

    check_run = steps[check_index]["run"]
    assert "ci_deps.py check-tags" in check_run
    assert "--plan" in check_run
    assert LANE in check_run

    # The jobs that install dependencies must wait for the plan job.
    for job in ("cpu", "torch", "models"):
        assert "list-test-files" in workflow["jobs"][job]["needs"]


def test_test_yaml_marks_the_three_for_the_313_lane():
    data = yaml.safe_load(TEST_YAML.read_text(encoding="utf-8"))
    for section, name in (
        ("tests", "test_eval"),
        ("tests/models", "test_nemotron_ultra"),
        ("tests/models", "test_hymba"),
    ):
        value = data[section][name]["py"]
        # Must stay a string: YAML reads an unquoted 3.10 as the float 3.1.
        assert isinstance(value, str), (section, name, value)
        assert value == GATE_LANE
