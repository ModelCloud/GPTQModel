# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The EXL3 ``bfloat164`` vectors must declare the header that defines them (#5760).

``gptqmodel_ext/exllamav3/util.cuh`` builds ``bfloat164`` from the CUDA bf16
spellings (``__nv_bfloat162``, ``__halves2bfloat162``) and has no include of its
own.  On CUDA that compiles because the translation units pull ``cuda_bf16.h``
in transitively; on ROCm hipify rewrites the identifiers to ``__hip_bfloat162``
but adds no header, so ``exl3_comp_unit_frac`` failed with

    util_hip.cuh:18:5: error: unknown type name '__hip_bfloat162'

and the whole EXL3 JIT extension -- which both ``EXL3_TORCH`` and the
reconstruct+hgemm path need -- could not build on ROCm at all.

A compile-level regression test would need a five-minute hipcc build, so this
pins the source contract instead: the platform bf16 header is included, under
the platform guard, *before* the typedef that uses it.
"""

from __future__ import annotations

import pathlib

import pytest


UTIL_CUH = pathlib.Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "exllamav3" / "util.cuh"


def _source() -> str:
    return UTIL_CUH.read_text()


def test_util_cuh_declares_the_platform_bf16_header() -> None:
    source = _source()

    assert "__nv_bfloat162" in source, (
        "util.cuh no longer uses the CUDA bf16 vectors; this contract can be dropped"
    )
    assert "#include <hip/hip_bf16.h>" in source, (
        "ROCm builds hipify __nv_bfloat162 to __hip_bfloat162; without "
        "<hip/hip_bf16.h> the extension fails with 'unknown type name "
        "__hip_bfloat162' (#5760)"
    )
    assert "#include <cuda_bf16.h>" in source, (
        "CUDA builds must keep declaring the header they rely on instead of "
        "depending on a transitive include"
    )


def test_bf16_include_precedes_the_vectors_that_need_it() -> None:
    source = _source()
    typedef_at = source.index("bfloat164")
    for header in ("<hip/hip_bf16.h>", "<cuda_bf16.h>"):
        include_at = source.index(f"#include {header}")
        assert include_at < typedef_at, (
            f"#include {header} must appear before the bfloat164 typedef"
        )


def test_bf16_include_is_platform_guarded() -> None:
    source = _source()
    hip_at = source.index("#include <hip/hip_bf16.h>")
    prefix = source[:hip_at]
    guard_at = prefix.rfind("#if")
    assert guard_at != -1, "the hip bf16 include is not inside a preprocessor guard"
    guard = prefix[guard_at:].splitlines()[0]
    assert "__HIP_PLATFORM_AMD__" in guard or "USE_ROCM" in guard, (
        f"the hip header must be selected by a ROCm guard, got: {guard!r}"
    )
    # The CUDA branch must be the `#else` of the same guard, not a second test.
    between = source[hip_at : source.index("#include <cuda_bf16.h>")]
    assert "#else" in between


def test_hipified_copy_of_the_header_has_no_include_of_its_own() -> None:
    """Documents *why* the include belongs in util.cuh and not in the callers.

    Hipify emits ``util_hip.cuh`` from this file with the identifiers rewritten
    and no include added, so a caller-side include would only help the callers
    that were edited.  If hipify ever starts emitting a header include this test
    will say so, and the contract can be relaxed.
    """

    source = _source()
    assert "#include" in source.split("typedef struct __align__(8) half4")[0]


@pytest.mark.parametrize("name", ["bfloat164", "half4", "half8"])
def test_vectors_are_still_declared(name: str) -> None:
    assert "typedef struct __align__" in _source()
    assert name in _source()
