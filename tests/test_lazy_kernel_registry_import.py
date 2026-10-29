# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Guard the lazy kernel registry and optional-dependency probes.

``import gptqmodel`` used to import every kernel module in
``gptqmodel/nn_modules/qlinear`` from ``utils/importer.py`` at module scope.
That pulled in bitsandbytes, BitBLAS and llama-cpp-python (several seconds of
import time and a large ``sys.modules`` footprint) even for callers that never
touch those backends. Two module-level availability probes
(``BITSANDBYTES_AVAILABLE`` / ``BITBLAS_AVAILABLE``) made it worse by importing
the optional package during kernel discovery.

These tests run in subprocesses so the assertions observe a clean interpreter
instead of a pytest session that already imported everything.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Optional backends that must not be imported by a bare ``import gptqmodel``.
_OPTIONAL_MODULES = ("bitsandbytes", "bitblas", "llama_cpp")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )


def _assert_ok(completed: subprocess.CompletedProcess) -> None:
    assert completed.returncode == 0, completed.stderr[-4000:]


def test_import_gptqmodel_skips_optional_backend_modules():
    _assert_ok(
        _run(
            f"""
            import sys
            import gptqmodel

            leaked = [name for name in {_OPTIONAL_MODULES!r} if name in sys.modules]
            assert not leaked, f"import gptqmodel imported optional backends: {{leaked}}"

            kernel_modules = [
                name for name in sys.modules
                if name.startswith("gptqmodel.nn_modules.qlinear.")
                and name.count(".") == 3
            ]
            assert len(kernel_modules) < 20, (
                f"kernel discovery ran eagerly at import time: {{sorted(kernel_modules)}}"
            )
            """
        )
    )


def test_kernel_registry_is_built_lazily_and_cached():
    _assert_ok(
        _run(
            """
            import gptqmodel.utils.importer as importer

            # Nothing imports the kernel package until a caller asks for it.
            assert importer._import_all_qlinear_kernels.cache_info().currsize == 0
            assert importer.build_kernel_support_maps.cache_info().currsize == 0

            auto = importer.get_auto_backend_kernel_mapping()
            supports = importer.get_backend_to_method_format_mapping()
            assert auto and supports

            # The registry is built once and every accessor observes the same object.
            assert importer.get_auto_backend_kernel_mapping() is auto
            assert importer.get_backend_to_method_format_mapping() is supports
            assert importer.build_kernel_support_maps()[0] is auto
            assert importer.AUTO_BACKEND_KERNEL_MAPPING is auto
            assert importer.BACKEND_TO_METHOD_FORMAT_MAPPING is supports
            assert importer.build_kernel_support_maps.cache_info().currsize == 1
            """
        )
    )


def test_optional_backend_probes_are_deferred_until_use():
    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module
            import gptqmodel.nn_modules.qlinear.bitsandbytes as bnb_module

            # Importing the kernel module must not import the optional package.
            assert not bitblas_module._probe_bitblas_available.cache_info().currsize
            assert not bnb_module._probe_bitsandbytes_available.cache_info().currsize

            # Reading the legacy attribute performs (and caches) the probe.
            assert isinstance(bool(bnb_module.BITSANDBYTES_AVAILABLE), bool)
            assert bnb_module._probe_bitsandbytes_available.cache_info().currsize == 1
            assert bnb_module.bitsandbytes_available() is bnb_module._probe_bitsandbytes_available()
            """
        )
    )


def test_legacy_availability_attributes_still_accept_overrides(monkeypatch):
    import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module
    import gptqmodel.nn_modules.qlinear.bitsandbytes as bnb_module

    monkeypatch.setattr(bnb_module, "BITSANDBYTES_AVAILABLE", False, raising=False)
    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True, raising=False)

    assert bnb_module.BITSANDBYTES_AVAILABLE is False
    assert bnb_module.bitsandbytes_available() is False
    assert bitblas_module.BITBLAS_AVAILABLE is True
    assert bitblas_module.bitblas_available() is True
