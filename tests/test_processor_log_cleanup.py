# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Processor logs are kept by default and cleaned up on request.

Each run writes one ``logs/<processor>_log_<random>_time_<stamp>.log``. Nothing
removed it, so the directory grew by one file per quantization - 17 files and
552 KB in a checkout two days old. ``QuantizeConfig.keep_processor_logs``
defaults to True so the existing behaviour does not change under anyone; setting
it False removes the file after a successful run. A run that fails keeps its log
either way, because ``finalize()`` only runs once the loop completes.
"""

import pytest

from gptqmodel.looper.loop_processor import remove_processor_log
from gptqmodel.quantization import GPTQConfig, QuantizeConfig


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "gptq_log_test_time_01_01_2026_00h_00m_00s.log"
    path.write_text('{"process": "gptq"}\n')
    return path


def test_keep_is_the_default(log_file):
    assert remove_processor_log(str(log_file)) is False
    assert log_file.exists()


def test_cleanup_removes_the_log_file(log_file):
    assert remove_processor_log(str(log_file), keep=False) is True
    assert not log_file.exists()


def test_missing_file_is_not_an_error(tmp_path):
    assert remove_processor_log(str(tmp_path / "never_written.log"), keep=False) is False


def test_none_path_is_ignored():
    assert remove_processor_log(None, keep=False) is False


def test_directory_path_is_left_alone(tmp_path):
    """A directory must never be unlinked, even if it somehow ends up in the field."""
    target = tmp_path / "logs_dir"
    target.mkdir()
    assert remove_processor_log(str(target), keep=False) is False
    assert target.is_dir()


@pytest.mark.parametrize("config_cls", [QuantizeConfig, GPTQConfig])
def test_config_defaults_to_keeping_logs(config_cls):
    config = config_cls(bits=4, group_size=128)
    assert config.keep_processor_logs is True


def test_config_accepts_an_opt_out():
    config = QuantizeConfig(bits=4, group_size=128, keep_processor_logs=False)
    assert config.keep_processor_logs is False
