# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""No test may call ``quantize()`` with a calibration kwarg removed by #4724.

#4688 added the 2-GPU "excluded GPU stays unclaimed" gate after #4724 had
already replaced the calibration keywords with ``CalibrationConfig``, so the
gate's subprocess died with ``TypeError`` before measuring anything and the
#4320/#4653 guarantee was unguarded (#4868). The gate is ``gpu``/``slow`` and
needs two devices, which is why CI never saw it.

This module is the cheap CPU-only guard: it parses every test module *and* the
Python source embedded in its string constants, and rejects any ``quantize()``
call that still passes a removed keyword.
"""

import ast
import pathlib


# Removed by 4a871bf (#4724). `quantize()` now takes one `CalibrationConfig`
# plus keyword-only `tokenizer`/`backend`/`adapter`/`embed_*`/`checkpoint`/`fuse`.
REMOVED_QUANTIZE_KWARGS = frozenset(
    {
        "calibration_concat_size",
        "calibration_sort",
        "batch_size",
        "calibration_data_min_length",
        "calibration_concat_separator",
        "calibration_exclude_attention_sinks",
        "adapter_calibration_dataset",
    }
)

TESTS_DIR = pathlib.Path(__file__).resolve().parent


def _embedded_scripts(tree):
    """Yield ``(label, module)`` for Python source held in string constants."""

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        if ".quantize(" not in value.value:
            continue
        try:
            parsed = ast.parse(value.value)
        except SyntaxError:
            # Not standalone Python: a fragment formatted at call time.
            continue
        target = node.targets[0]
        name = target.id if isinstance(target, ast.Name) else "<script>"
        yield name, parsed


def _python_sources():
    """Yield ``(label, module)`` for every test module and embedded script."""

    for path in sorted(TESTS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        yield path.relative_to(TESTS_DIR).as_posix(), tree
        for name, script in _embedded_scripts(tree):
            yield f"{path.relative_to(TESTS_DIR).as_posix()}::{name}", script


def _offending_keywords(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "quantize":
            continue
        for keyword in node.keywords:
            if keyword.arg in REMOVED_QUANTIZE_KWARGS:
                yield node.lineno, keyword.arg


def test_no_test_calls_quantize_with_a_removed_calibration_kwarg():
    offenders = [
        f"{label}:{lineno}: quantize(..., {kwarg}=...)"
        for label, tree in _python_sources()
        for lineno, kwarg in _offending_keywords(tree)
    ]

    assert offenders == [], (
        "quantize() dropped these calibration keywords in #4724; pass a "
        "CalibrationConfig instead:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_reads_embedded_subprocess_scripts():
    """Guard the guard: the 2-GPU gate's script must be among the parsed sources."""

    labels = [label for label, _ in _python_sources()]

    assert any(
        label.startswith("test_excluded_device_warmup.py::") for label in labels
    ), "the embedded-script scan found nothing in test_excluded_device_warmup.py"


def test_the_excluded_device_gate_passes_a_calibration_config():
    """The #4653 gate must reach the measurement instead of raising TypeError."""

    sources = dict(_python_sources())
    label = next(
        name
        for name in sources
        if name.startswith("test_excluded_device_warmup.py::")
    )

    calls = [
        node
        for node in ast.walk(sources[label])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "quantize"
    ]

    assert len(calls) == 1
    first = calls[0].args[0]
    assert isinstance(first, ast.Call) and getattr(first.func, "id", None) == "CalibrationConfig"
