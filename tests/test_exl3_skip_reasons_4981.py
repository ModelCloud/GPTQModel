# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EXL3 tests must not be unconditionally skipped (#4981).

Seven EXL3 tests carried a module-level ``pytest.mark.skip`` since #3592/#3594/
#3599 with "lands in a later stack layer" reasons. The wiring landed; the skips
stayed, so shipped code paths had tests that pass but nobody ran. A real
precondition belongs in a conditional ``pytest.mark.skipif`` (or a runtime
``pytest.skip``) with a precise reason, so a capable host still runs the test.
"""

from __future__ import annotations

import ast
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
EXL3_TEST_GLOBS = (
    "test_exllamav3*.py",
    "test_exl3_*.py",
    # The EXL3 projection-capture tests live in a config-named file.
    "test_quantize_config_offload_tempdir.py",
)


def _unconditional_skip_calls(path: Path) -> list[int]:
    """Line numbers of ``pytest.mark.skip(...)`` calls (not ``skipif``)."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "skip"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "mark"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pytest"
        ):
            continue
        lines.append(node.lineno)
    return lines


def test_no_exl3_test_is_unconditionally_skipped():
    files = sorted(
        path
        for pattern in EXL3_TEST_GLOBS
        for path in TESTS_DIR.glob(pattern)
    )
    assert files, "no EXL3 test files found; the globs are stale"

    offenders = []
    for path in files:
        lines = _unconditional_skip_calls(path)
        if lines:
            offenders.append(f"{path.name}:{','.join(str(line) for line in lines)}")

    assert not offenders, (
        "unconditional EXL3 test skips: "
        f"{offenders}. Gate on the real precondition with pytest.mark.skipif or a "
        "runtime pytest.skip so a capable host still runs the test (#4981)."
    )
