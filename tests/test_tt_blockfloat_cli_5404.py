# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The documented Tenstorrent block-float command must parse (#5404).

The usage example omitted calibration while the parser required
``--calibration``/``--calibration-dataset``, so copying the documented command
exited with code 2 before the model path was even opened. The
``--calibration-dataset`` help also promised a default that the required group
made unreachable.
"""

import pytest

import scripts.tt_quantize_blockfloat as script
from scripts.tt_quantize_blockfloat import build_parser


def _usage_lines():
    """Join the docstring's ``\\`` continuations, then keep the command lines."""
    commands, pending = [], ""
    for raw in script.__doc__.splitlines():
        line = raw.strip()
        if not line:
            continue
        if pending:
            line = f"{pending} {line}"
            pending = ""
        if line.endswith("\\"):
            pending = line[:-1].strip()
            continue
        if line.startswith("python scripts/tt_quantize_blockfloat.py"):
            commands.append(line)
    return commands


def test_documented_dataset_invocation_parses():
    args = build_parser().parse_args(
        [
            "./local-model",
            "--out",
            "./local-output",
            "--bits",
            "4",
            "--calibration-dataset",
            "./samples.parquet",
        ]
    )
    assert args.calibration_dataset == "./samples.parquet"
    assert args.calibration is None


def test_documented_text_invocation_parses():
    args = build_parser().parse_args(
        [
            "./local-model",
            "--out",
            "./local-output",
            "--calibration",
            "A calibration passage.",
        ]
    )
    assert args.calibration == "A calibration passage."
    assert args.calibration_dataset is None


def test_calibration_source_is_still_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["./local-model", "--out", "./local-output"])


def test_every_documented_example_passes_a_calibration_source():
    usage = _usage_lines()
    assert usage, "the usage examples disappeared from the module docstring"
    for line in usage:
        assert "--calibration" in line, f"example omits calibration: {line!r}"


def test_dataset_help_does_not_promise_an_unreachable_default():
    # argparse wraps the help; compare on normalized whitespace.
    help_text = " ".join(build_parser().format_help().split())
    assert "--calibration-dataset" in help_text
    # The parser always takes an explicit source, so advertising the environment
    # fallback here would describe a default no invocation can reach.
    assert "GPTQMODEL_CALIBRATION_PARQUET" not in help_text
    assert "always explicit" in help_text
