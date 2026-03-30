# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import sys

from gptqmodel.utils.logger import setup_logger


def test_setup_logger_suppresses_live_renderables_under_pytest_capture(monkeypatch):
    logger = setup_logger()

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_logger.py::test")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)

    progress = logger.pb(range(3)).manual().set(show_left_steps=False).title("x").subtitle("y").draw()
    spinner = logger.spinner(title="spin", interval=0.01)
    int_progress = logger.pb(2).manual().title("z")

    assert list(progress) == [0, 1, 2]
    assert len(progress) == 3
    assert progress.next().current_iter_step == 1
    assert progress.next(2).current_iter_step == 3
    assert list(int_progress) == [0, 1]
    assert len(int_progress) == 2
    spinner.close()
    progress.close()
    int_progress.close()
