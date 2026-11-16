# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Consistency checks for the GIL=0 compatibility ledger (#4207).

``pytorch_gil_0_compat.md`` is only useful if it stays true. Two things rot
silently: a row that describes a call site the code no longer has, and a row
that quotes a probe name ``scripts/gil0_api_probe.py`` does not define. Both
make a reader trust a result that was never measured for that API.

These tests are static: they parse the Markdown and the probe registry, and
never start a GPU, a subprocess, or a free-threaded interpreter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "pytorch_gil_0_compat.md"
PROBE_SCRIPT = REPO_ROOT / "scripts" / "gil0_api_probe.py"

# Rows this issue added or corrected: (API token in the first cell, substrings
# the row must contain). The call sites are the reason the API needs a row at
# all, so they are pinned alongside the result.
REQUIRED_ROWS = (
    (
        "`torch.linalg.cholesky`",
        ("quantization/gptq.py", "quantization/qqq.py", "utils/linalg_warmup.py", "ROCm"),
    ),
    (
        "`torch.linalg.cholesky_ex`",
        ("quantization/gptq.py", "utils/linalg_warmup.py", "ROCm"),
    ),
    (
        "`torch.cholesky_inverse`",
        ("quantization/gptq.py", "quantization/qqq.py", "ROCm"),
    ),
    (
        "`torch.backends.cuda.preferred_blas_library`",
        ("utils/rocm.py", "quant_blas_backend", "ROCm"),
    ),
    (
        "`Tensor.record_stream`",
        ("nn_modules/qlinear/torch.py", "wait_stream", "ROCm"),
    ),
    (
        "`threadpoolctl.threadpool_limits`",
        ("#4206", "UNSAFE", "nn_modules/qlinear/__init__.py"),
    ),
    (
        "`torch.cuda.CUDAGraph`",
        ("paroquant/optimization.py", "stage_cudagraph", "ROCm"),
    ),
)


def _ledger_rows() -> list[tuple[str, str]]:
    """Return ``(first cell, full row)`` for every Markdown table row."""
    rows: list[tuple[str, str]] = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        first = cells[1].strip()
        if not first or first == "API" or set(first) <= {"-", ":"}:
            continue
        rows.append((first, line))
    return rows


def _rows_for(token: str) -> list[str]:
    return [row for first, row in _ledger_rows() if token in first]


def _probe_names() -> set[str]:
    from scripts.gil0_api_probe import build_probes

    return set(build_probes())


@pytest.mark.parametrize(("token", "required"), REQUIRED_ROWS)
def test_ledger_has_a_row_for_every_threaded_api(token: str, required: tuple[str, ...]) -> None:
    rows = _rows_for(token)
    assert rows, f"pytorch_gil_0_compat.md has no ledger row for {token}"
    matching = [row for row in rows if all(fragment in row for fragment in required)]
    assert matching, (
        f"the {token} row does not document {required}; rows were:\n" + "\n".join(rows)
    )
    assert any("SAFE" in row or "UNSAFE" in row for row in matching), (
        f"the {token} row has no SAFE/UNSAFE verdict"
    )


def test_ledger_probe_names_exist_in_the_probe_script() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    names = set()
    for match in re.finditer(r"--only\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))", text):
        name = next(group for group in match.groups() if group is not None).strip("`")
        if name.startswith("<") or not name:
            continue  # documentation placeholder, not a probe name
        names.add(name)
    assert names, "the ledger no longer references any probe name"
    defined = _probe_names()
    missing = sorted(names - defined)
    assert not missing, f"ledger quotes probes that do not exist: {missing}"


def test_cudagraph_row_describes_the_gated_usage() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    assert "not used yet" not in text, "the stale 'not used yet' CUDAGraph claim came back"
    rows = _rows_for("`torch.cuda.CUDAGraph`")
    assert rows, "pytorch_gil_0_compat.md has no ledger row for torch.cuda.CUDAGraph"
    row = rows[0]
    assert "single-thread" in row
    assert "optimize_paroquant_linear" in row
    assert "_should_use_paroquant_stage_cudagraph" in row
    assert "looper/paroquant_processor.py" in row
