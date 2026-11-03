# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Successful quantization runs should not leave a processor log behind.

Each run writes one ``logs/<processor>_log_<random>_time_<stamp>.log`` and
nothing removed it, so the directory grew by one file per quantization. The
file is still useful when a run fails, so the flag keeps it on request.
"""

import pytest

from gptqmodel.looper.loop_processor import remove_processor_log


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "gptq_log_test_time_01_01_2026_00h_00m_00s.log"
    path.write_text('{"process": "gptq"}\n')
    return path


def test_removes_the_log_file(log_file, monkeypatch):
    monkeypatch.delenv("GPTQMODEL_KEEP_PROCESSOR_LOGS", raising=False)
    assert remove_processor_log(str(log_file)) is True
    assert not log_file.exists()


def test_keeping_the_log_is_opt_in(log_file, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KEEP_PROCESSOR_LOGS", "1")
    assert remove_processor_log(str(log_file)) is False
    assert log_file.exists()


def test_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GPTQMODEL_KEEP_PROCESSOR_LOGS", raising=False)
    assert remove_processor_log(str(tmp_path / "never_written.log")) is False


def test_none_path_is_ignored(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_KEEP_PROCESSOR_LOGS", raising=False)
    assert remove_processor_log(None) is False


def test_directory_path_is_left_alone(tmp_path, monkeypatch):
    """A directory must never be unlinked, even if it somehow ends up in the field."""
    monkeypatch.delenv("GPTQMODEL_KEEP_PROCESSOR_LOGS", raising=False)
    target = tmp_path / "logs_dir"
    target.mkdir()
    assert remove_processor_log(str(target)) is False
    assert target.is_dir()
