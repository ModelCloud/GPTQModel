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
    monkeypatch.setattr(
        doctor, "_module_installed", lambda name: name not in ("numpy", "triton")
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


def _write_custom_checkpoint(tmp_path):
    """A checkpoint whose config needs custom Python, like a remote-code model."""

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "custom",
                "architectures": ["Custom"],
                "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            }
        )
    )
    # transformers copies a dynamic module into a temporary directory before
    # importing it, so the marker path comes from the environment.
    (tmp_path / "configuration_custom.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['CLI_DOCTOR_EXECUTED']).write_text('yes')\n"
        "from transformers import PretrainedConfig\n\n"
        "class CustomConfig(PretrainedConfig):\n"
        "    model_type = 'custom'\n"
    )
    return tmp_path


def test_a_custom_config_is_not_executed_by_default(monkeypatch, tmp_path):
    marker = tmp_path / "EXECUTED"
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    monkeypatch.setenv("CLI_DOCTOR_EXECUTED", str(marker))
    model = _write_custom_checkpoint(tmp_path)

    checks = doctor._model_checks(str(model))

    assert checks[0].status == doctor.FAIL
    assert "trust_remote_code" in checks[0].detail.lower()
    assert not marker.exists()


def test_remote_code_needs_both_the_flag_and_the_repository_bypass(monkeypatch, tmp_path):
    marker = tmp_path / "EXECUTED"
    monkeypatch.setenv("CLI_DOCTOR_EXECUTED", str(marker))
    model = _write_custom_checkpoint(tmp_path)
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)

    refused = doctor._model_checks(str(model), trust_remote_code=True)

    assert refused[0].name == "remote code"
    assert refused[0].status == doctor.FAIL
    assert not marker.exists()

    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    allowed = doctor._model_checks(str(model), trust_remote_code=True)

    assert not any(check.status == doctor.FAIL for check in allowed)
    assert marker.exists()


def test_out_of_range_dependency_is_a_warning(monkeypatch):
    from packaging.requirements import Requirement

    monkeypatch.setattr(
        doctor,
        "_supported_specs",
        lambda: {"transformers": [Requirement("transformers>=5.23.0,<5.24")]},
    )
    monkeypatch.setattr(
        doctor, "_module_version", lambda name: "5.24.0" if name == "transformers" else "1.0"
    )

    checks = doctor._dependency_checks()

    row = next(check for check in checks if check.name == "transformers")
    assert row.status == doctor.WARN
    assert "5.24.0" in row.detail and "<5.24" in row.detail


def test_tokenicer_version_comes_from_metadata_without_importing_it(monkeypatch):
    real_version = doctor.importlib_metadata.version

    def version(distribution):
        return "0.0.21" if distribution == "tokenicer" else real_version(distribution)

    monkeypatch.setattr(doctor.importlib_metadata, "version", version)

    row = next(check for check in doctor._dependency_checks() if check.name == "tokenicer")

    assert row.status == doctor.OK
    assert row.detail == "0.0.21"


def test_extension_check_reports_the_torch_jit_cache(monkeypatch, tmp_path):
    artifact = tmp_path / "torch_extensions" / "py312_cu132" / "allspark_spike" / "allspark_spike.so"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    monkeypatch.setattr(doctor, "_torch_extension_artifacts", lambda: [artifact])

    row = doctor._extension_checks()[0]

    assert row.status == doctor.OK
    assert "JIT artifact" in row.detail
    assert "allspark_spike" in row.detail


def test_rocm_devices_use_hip_labels(monkeypatch):
    import types

    import torch

    monkeypatch.setattr(torch.version, "hip", "7.0.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "AMD Radeon RX 9070")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (8 * 2**30, 16 * 2**30))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(gcnArchName="gfx1201"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda index: (_ for _ in ()).throw(AssertionError("ROCm must not be read as CUDA")),
    )

    row = doctor._gpu_checks()[0]

    assert row.name == "hip:0"
    assert "gfx1201" in row.detail
