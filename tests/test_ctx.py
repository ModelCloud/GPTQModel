# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from typing import Any, List, Tuple

import pytest

from gptqmodel.utils.ctx import ctx


class RecordingContext(AbstractContextManager[Any]):
    def __init__(self, name: str, log: List[Tuple[str, str, int]]):
        self.name = name
        self.log = log

    def __enter__(self) -> str:
        self.log.append(("enter", self.name, threading.get_ident()))
        return self.name

    def __exit__(self, exc_type, exc, tb) -> None:
        self.log.append(("exit", self.name, threading.get_ident()))
        return False


def test_ctx_enters_in_order_and_exits_in_reverse() -> None:
    log: List[Tuple[str, str, int]] = []
    contexts = [RecordingContext("first", log), RecordingContext("second", log), RecordingContext("third", log)]

    with ctx(*contexts) as values:
        assert values == ("first", "second", "third")
        assert log == [
            ("enter", "first", threading.get_ident()),
            ("enter", "second", threading.get_ident()),
            ("enter", "third", threading.get_ident()),
        ]

    assert log[3:] == [
        ("exit", "third", threading.get_ident()),
        ("exit", "second", threading.get_ident()),
        ("exit", "first", threading.get_ident()),
    ]


def test_ctx_skips_none_and_returns_single_value() -> None:
    log: List[Tuple[str, str, int]] = []
    single = RecordingContext("solo", log)

    with ctx(None, single, None) as value:
        assert value == "solo"
        assert log == [("enter", "solo", threading.get_ident())]

    assert log[1:] == [("exit", "solo", threading.get_ident())]


def test_ctx_returns_none_when_no_contexts() -> None:
    with ctx() as value:
        assert value is None


def test_ctx_exits_all_contexts_when_exception_raised() -> None:
    log: List[Tuple[str, str, int]] = []
    contexts = [RecordingContext("first", log), RecordingContext("second", log)]

    class ExpectedError(RuntimeError):
        pass

    with pytest.raises(ExpectedError):
        with ctx(*contexts):
            raise ExpectedError()

    assert log == [
        ("enter", "first", threading.get_ident()),
        ("enter", "second", threading.get_ident()),
        ("exit", "second", threading.get_ident()),
        ("exit", "first", threading.get_ident()),
    ]


def test_ctx_threaded_execution_respects_order(monkeypatch) -> None:
    monkeypatch.setenv("PYTHON_GIL", "0")

    def worker(thread_index: int) -> List[Tuple[str, str, int]]:
        log: List[Tuple[str, str, int]] = []
        contexts = [
            RecordingContext(f"c{thread_index}-1", log),
            None,
            RecordingContext(f"c{thread_index}-2", log),
        ]
        with ctx(*contexts) as values:
            assert values == (f"c{thread_index}-1", f"c{thread_index}-2")
        return log

    with ThreadPoolExecutor(max_workers=4) as executor:
        logs = list(executor.map(worker, range(4)))

    for idx, log in enumerate(logs):
        entries = [item for item in log if item[0] == "enter"]
        exits = [item for item in log if item[0] == "exit"]
        assert len(entries) == 2
        assert len(exits) == 2
        enter_thread = entries[0][2]
        exit_thread = exits[0][2]
        assert enter_thread == exit_thread
        assert entries == [
            ("enter", f"c{idx}-1", enter_thread),
            ("enter", f"c{idx}-2", enter_thread),
        ]
        assert exits == [
            ("exit", f"c{idx}-2", exit_thread),
            ("exit", f"c{idx}-1", exit_thread),
        ]

