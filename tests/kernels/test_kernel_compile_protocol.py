# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every registered kernel declares its compile/graph capability.

The declarations live in the kernel modules (`SUPPORTS_COMPILE_LEVEL` and
`SUPPORTS_GRAPH_CAPTURE`); this test is the registry that keeps them honest:

* every registered kernel declares both, explicitly (no silent inheritance);
* a `CUDAGRAPH` or `SUPPORTS_GRAPH_CAPTURE=True` claim cannot coexist with a
  host-sync / host-transfer / device-address construct in that kernel's own
  `forward`;
* a `SUPPORTS_COMPILE=True` policy claim implies at least `DYNAMO`;
* `docs/kernel-compile-matrix.md` lists exactly the registered kernels and
  matches their declarations.

The per-family `test_kernel_compile_levels_*.py` files re-measure the level on
real hardware; this file checks the declaration surface and its static
consistency.
"""

from pathlib import Path

import pytest

from gptqmodel.nn_modules.qlinear._compile_support import (
    COMPILE_LEVEL,
    compile_level,
    describe_kernel,
    graph_capture_safe,
)
from tests.kernels.compile_probe import graph_unsafe_constructs, protocol_kernels


MATRIX = Path(__file__).resolve().parents[2] / "docs" / "kernel-compile-matrix.md"

pytestmark = [pytest.mark.cpu]


def _kernels():
    return protocol_kernels()


def test_every_registered_kernel_declares_both_attributes():
    missing = [
        cls.__name__
        for cls in _kernels()
        if "SUPPORTS_COMPILE_LEVEL" not in cls.__dict__
        or "SUPPORTS_GRAPH_CAPTURE" not in cls.__dict__
    ]

    assert not missing, (
        "these kernels do not declare SUPPORTS_COMPILE_LEVEL and "
        "SUPPORTS_GRAPH_CAPTURE in their own class body: "
        + ", ".join(missing)
        + ". Measure the forward with tests/kernels/compile_probe.py and declare "
        "the result (see docs/kernel-compile-matrix.md)."
    )


def test_declared_levels_are_compile_levels():
    for cls in _kernels():
        value = cls.__dict__.get("SUPPORTS_COMPILE_LEVEL")
        assert isinstance(value, COMPILE_LEVEL), (
            f"{cls.__name__}.SUPPORTS_COMPILE_LEVEL must be a COMPILE_LEVEL, "
            f"got {type(value).__name__}"
        )
        assert type(cls.__dict__.get("SUPPORTS_GRAPH_CAPTURE")) is bool, (
            f"{cls.__name__}.SUPPORTS_GRAPH_CAPTURE must be a bool"
        )


@pytest.mark.parametrize("cls", _kernels(), ids=lambda cls: cls.__name__)
def test_capture_claims_have_no_host_sync_in_forward(cls):
    claims_capture = graph_capture_safe(cls) or compile_level(cls) >= COMPILE_LEVEL.CUDAGRAPH
    if not claims_capture:
        pytest.skip("kernel does not claim graph capture")

    unsafe = graph_unsafe_constructs(cls)
    assert not unsafe, (
        f"{cls.__name__} claims CUDA-graph capture safety, but its own forward "
        f"contains: {', '.join(unsafe)}. A capture cannot cross a host sync or "
        "transfer; lower the declaration or move the call out of forward."
    )


@pytest.mark.parametrize("cls", _kernels(), ids=lambda cls: cls.__name__)
def test_compiled_generation_claim_is_traceable(cls):
    if not getattr(cls, "SUPPORTS_COMPILE", False):
        pytest.skip("kernel does not opt into compiled generation")
    if compile_level(cls) is COMPILE_LEVEL.NOT_APPLICABLE:
        # MLX/TTNN/XLA kernels are not torch modules; their own compiler owns
        # the graph, so the torch-compile policy flag does not apply.
        pytest.skip("kernel is not a torch module")
    assert compile_level(cls) >= COMPILE_LEVEL.DYNAMO, (
        f"{cls.__name__} sets SUPPORTS_COMPILE=True (compiling is not a "
        "pessimization) but declares a level below DYNAMO, so the compiled "
        "generation path would pay for graph fragments."
    )


def _matrix_rows():
    rows = {}
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        kernel = cells[0].strip("`")
        rows[kernel] = {
            "level": cells[2].strip("`"),
            "graph_capture": cells[3].lower() == "yes",
            "compiled_generation": cells[4].lower() == "yes",
        }
    return rows


def test_docs_matrix_matches_the_declarations():
    rows = _matrix_rows()
    declared = {cls.__name__: describe_kernel(cls) for cls in _kernels()}

    assert set(rows) == set(declared), (
        "docs/kernel-compile-matrix.md is out of sync with the registry; "
        f"missing={sorted(set(declared) - set(rows))} "
        f"stale={sorted(set(rows) - set(declared))}. "
        "Regenerate it with `python -m tests.kernels.gen_compile_matrix`."
    )
    for name, row in rows.items():
        want = declared[name]
        assert row["level"] == want["compile_level"], (
            f"{name}: docs say {row['level']}, kernel declares {want['compile_level']}"
        )
        assert row["graph_capture"] is want["graph_capture"], (
            f"{name}: docs graph capture is stale"
        )
        assert row["compiled_generation"] is want["compiled_generation"], (
            f"{name}: docs compiled-generation policy is stale"
        )
