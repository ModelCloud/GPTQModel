# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""`gptqmodel doctor`: the report must describe this machine without failing.

The checks touch the live environment (torch build, GPUs, installed packages), so
these tests assert structure and behaviour rather than exact values - except for
the paths a user acts on: a missing required dependency must be a FAIL, a missing
optional one must not be, and a bad `--model` path must name the problem.
"""

import json

import pytest

from gptqmodel.cli import doctor
from gptqmodel.cli.main import main


STATUSES = {"OK", "WARN", "FAIL", "INFO"}


def test_collect_reports_the_areas_a_bug_report_needs():
    checks = doctor.collect()

    assert checks
    areas = {check.area for check in checks}
    assert {"runtime", "torch", "deps", "env"} <= areas
    assert all(check.status in STATUSES for check in checks)


def test_json_and_table_share_the_same_rows():
    checks = doctor.collect()

    payload = doctor.to_json(checks)
    assert len(payload["checks"]) == len(checks)
    assert payload["failing"] == sum(check.status == "FAIL" for check in checks)

    table = doctor.render(checks)
    assert "area" in table and "result:" in table


def test_exit_code_follows_the_worst_status():
    good = [doctor.Check("x", "y", doctor.OK, "fine")]
    warned = [doctor.Check("x", "y", doctor.WARN, "meh")]
    failed = [doctor.Check("x", "y", doctor.FAIL, "broken")]

    assert doctor.exit_code(good) == 0
    assert doctor.exit_code(warned) == 0
    assert doctor.exit_code(warned, fail_on="warn") == 1
    assert doctor.exit_code(failed) == 1


def test_a_missing_required_dependency_is_a_failure(monkeypatch):
    monkeypatch.setattr(
        doctor, "_module_version", lambda name: None if name in ("numpy", "triton") else "1.0"
    )

    checks = doctor._dependency_checks()

    numpy = next(check for check in checks if check.name == "numpy")
    assert numpy.status == doctor.FAIL
    triton = next(check for check in checks if check.name == "triton")
    assert triton.status == doctor.INFO  # optional, absent is fine


def test_cli_writes_a_clean_json_file(tmp_path, capsys):
    report = tmp_path / "doctor.json"

    code = main(["doctor", "--json-out", str(report)])

    assert code in (0, 1)
    payload = json.loads(report.read_text())
    assert payload["checks"]
    assert "area" in capsys.readouterr().out


def test_cli_reports_a_bad_model_path(tmp_path, capsys):
    code = main(["doctor", "--model", str(tmp_path / "missing")])

    assert code == 1
    assert "model" in capsys.readouterr().out


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == 2
