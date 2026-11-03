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

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Optional backends that must not be imported by a bare ``import gptqmodel``.
_OPTIONAL_MODULES = ("bitsandbytes", "bitblas", "llama_cpp")


# Variables that change import-time behaviour. They are stripped from the child
# environment so a developer shell (DEBUG=1, an opt-in kernel-map dump, ...)
# cannot flip the meaning of these tests.
_SCRUBBED_ENV = (
    "DEBUG",
    "GPTQMODEL_DEBUG_KERNEL_MAPS",
    "GPTQMODEL_DISABLE_BITBLAS",
    "PYTHONPATH",
    "PYTHONSTARTUP",
)


def _run(code: str) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key not in _SCRUBBED_ENV}
    # Import the tree these tests live in, not whatever is installed site-wide.
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
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
            assert len(kernel_modules) <= 12, (
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
            assert importer._imported_kernel_module_count() == 0
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
            assert bnb_module.BITSANDBYTES_AVAILABLE is bnb_module._probe_bitsandbytes_available()
            assert bnb_module._probe_bitsandbytes_available.cache_info().currsize == 1
            assert not bitblas_module._probe_bitblas_available.cache_info().currsize

            assert bitblas_module.BITBLAS_AVAILABLE is bitblas_module._probe_bitblas_available()
            assert bitblas_module._probe_bitblas_available.cache_info().currsize == 1
            """
        )
    )


def test_gguf_cpp_defers_llama_cpp_import_until_the_kernel_runs():
    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.gguf_cpp as gguf_cpp

            # Importing the kernel module must not import llama-cpp-python.
            assert not gguf_cpp._LLAMA_CPP_LOADED
            import sys

            assert "llama_cpp" not in sys.modules

            gguf_cpp._require_llama_cpp()
            assert gguf_cpp._LLAMA_CPP_LOADED
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


def test_auto_select_maps_ignore_kernels_defined_outside_the_package():
    """A subclass defined after import must not hijack the cached auto-select map.

    The maps are built lazily now, so the subclass walk sees whatever exists at
    first use. Kernels defined outside ``gptqmodel.nn_modules.qlinear`` belong to
    tests or downstream plugins and must stay out of the shipped maps, exactly as
    they did when the maps were frozen during ``import gptqmodel``.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel
            import gptqmodel.utils.importer as importer
            from gptqmodel.models._const import DEVICE, PLATFORM
            from gptqmodel.nn_modules.qlinear import GPTQQuantLinear
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            import torch


            class _Hijacker(GPTQQuantLinear):
                SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH]
                SUPPORTS_METHODS = [METHOD.GPTQ]
                SUPPORTS_FORMATS = {FORMAT.GPTQ: 999}
                SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA]
                SUPPORTS_PLATFORM = [PLATFORM.ALL]
                SUPPORTS_BITS = [4]
                SUPPORTS_DTYPES = [torch.float32]
                SUPPORTS_PACK_DTYPES = [torch.int32]
                SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
                SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

                @classmethod
                def validate_once(cls):
                    return True, None

                @classmethod
                def _validate(cls, **kwargs):
                    return True, None


            # Discovery still reports it; it is a real ``BaseQuantLinear`` subclass.
            assert _Hijacker in importer.iter_quant_linear_kernels()

            # ...but the shipped auto-select map keeps the production kernel.
            auto = importer.get_auto_backend_kernel_mapping()
            owner = auto[METHOD.GPTQ][FORMAT.GPTQ][BACKEND.GPTQ_TORCH]
            assert owner is not _Hijacker, f"auto-select map was hijacked by {owner.__name__}"
            assert owner.__name__ == "TorchLinear", owner.__name__
            """
        )
    )


def test_unparsable_optional_backend_version_is_reported_not_raised():
    """A broken ``__version__`` must not escape ``validate_once``.

    ``validate_once`` contracts to return ``(ok, exception)``. Before the probe
    was deferred the crash happened at ``import gptqmodel``; now it has to be
    swallowed by the probe instead of escaping from the kernel contract.
    """

    _assert_ok(
        _run(
            """
            import sys
            import types

            # Poison both optional packages with a version that cannot be parsed.
            for name in ("bitsandbytes", "bitblas"):
                module = types.ModuleType(name)
                module.__version__ = "not-a-version"
                sys.modules[name] = module

            import gptqmodel
            from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear
            from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear

            for kernel in (BitsAndBytesLinear, BitblasLinear):
                ok, error = kernel.validate_once()
                assert ok is False, f"{kernel.__name__}.validate_once() returned {ok!r}"
                assert isinstance(error, BaseException), error

                ok_cached, error_cached = kernel.cached_validate_once()
                assert ok_cached is False, kernel.__name__
                assert isinstance(error_cached, BaseException), error_cached
            """
        )
    )


def test_llama_cpp_import_is_single_flight():
    """Concurrent first callers must all observe the loaded module objects."""

    _assert_ok(
        _run(
            """
            import builtins
            import threading
            import time

            import gptqmodel.nn_modules.qlinear.gguf_cpp as gguf_cpp

            real_import = builtins.__import__


            def slow_import(name, *args, **kwargs):
                if name == "llama_cpp":
                    time.sleep(0.75)
                return real_import(name, *args, **kwargs)


            builtins.__import__ = slow_import
            try:
                results = {}
                entered = threading.Event()

                def load():
                    entered.set()
                    results["first"] = gguf_cpp._require_llama_cpp()

                thread = threading.Thread(target=load)
                thread.start()
                entered.wait(timeout=10)
                time.sleep(0.25)  # let the first caller enter the slow import
                results["second"] = gguf_cpp._require_llama_cpp()
                thread.join(timeout=30)
            finally:
                builtins.__import__ = real_import

            first, second = results["first"], results["second"]
            assert first == second, (first, second)

            package = gguf_cpp._llama_cpp_pkg
            if package is not None:
                assert second[0] is package, second[0]
            else:
                # ``llama-cpp-python`` is genuinely missing here: the caller must
                # get the recorded import error, never a ``None.__file__`` crash.
                assert gguf_cpp._LLAMA_CPP_IMPORT_ERROR is not None
            """
        )
    )


def test_debug_env_does_not_materialize_the_registry_at_import_time():
    """``DEBUG=1`` must not undo the laziness the module exists to provide.

    ``debug_print_kernel_maps()`` needs the built maps, so calling it under
    ``DEBUG`` at module scope imported every kernel (and optional backend) again.
    A dedicated opt-in variable now guards the dump.
    """

    _assert_ok(
        _run(
            """
            import os
            os.environ["DEBUG"] = "1"
            # The child environment is scrubbed, so this test means the same
            # thing on a developer shell that already exports the opt-in dump.
            assert "GPTQMODEL_DEBUG_KERNEL_MAPS" not in os.environ

            import gptqmodel.utils.importer as importer

            assert importer._imported_kernel_module_count() == 0, (
                "DEBUG=1 built the kernel registry during import"
            )
            assert importer.build_kernel_support_maps.cache_info().currsize == 0
            """
        )
    )


def test_kernel_map_build_is_single_flight_under_concurrency():
    """Concurrent first callers must share one map instance and one build.

    ``lru_cache`` alone is not single-flight: every racing thread computes its
    own value and only the last is retained, so a caller can hold a map that no
    later reader will ever observe.
    """

    _assert_ok(
        _run(
            """
            import threading

            import gptqmodel  # noqa: F401
            import gptqmodel.utils.importer as importer

            assert importer.build_kernel_support_maps.cache_info().currsize == 0

            seen = []
            barrier = threading.Barrier(8)

            def build():
                barrier.wait()
                seen.append(id(importer.get_auto_backend_kernel_mapping()))

            threads = [threading.Thread(target=build) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)

            assert len(seen) == 8, seen
            assert len(set(seen)) == 1, f"threads observed distinct maps: {len(set(seen))}"
            assert importer.build_kernel_support_maps.cache_info().misses == 1
            """
        )
    )


def test_legacy_map_names_are_visible_to_dir_and_star_import():
    """Module ``__getattr__`` names are invisible unless advertised.

    Before the registry became lazy these were real globals, so ``dir()`` and
    ``from ... import *`` exposed them. Keep that surface intact.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.utils.importer as importer

            names = dir(importer)
            assert "AUTO_BACKEND_KERNEL_MAPPING" in names
            assert "BACKEND_TO_METHOD_FORMAT_MAPPING" in names

            namespace = {}
            exec("from gptqmodel.utils.importer import *", namespace)
            assert "AUTO_BACKEND_KERNEL_MAPPING" in namespace
            assert "BACKEND_TO_METHOD_FORMAT_MAPPING" in namespace
            assert namespace["AUTO_BACKEND_KERNEL_MAPPING"] is importer.AUTO_BACKEND_KERNEL_MAPPING

            # Before this module had an ``__all__``, ``import *`` exported every
            # public global. Adding ``__all__`` must not shrink that surface to
            # the two lazy names.
            for name in (
                "select_quant_linear",
                "build_kernel_support_maps",
                "iter_quant_linear_kernels",
                "get_kernel_for_backend",
                "validate_quant_linear",
                "BACKEND",
            ):
                assert name in namespace, f"star-import lost {name}"
            assert len([n for n in namespace if not n.startswith("_")]) >= 40
            """
        )
    )


def test_missing_optional_version_does_not_escape_validate_once():
    """A backend with no ``__version__`` at all must report unusable, not raise.

    The first fix only covered an *unparsable* version: the debug log re-read
    ``bitblas.__version__`` inside the handler, so a module without the attribute
    raised again and escaped ``validate_once``'s ``(bool, exception)`` contract.
    """

    _assert_ok(
        _run(
            """
            import sys
            import types

            for name in ("bitsandbytes", "bitblas"):
                module = types.ModuleType(name)
                # Deliberately no ``__version__`` attribute.
                sys.modules[name] = module

            from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear
            from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear

            for kernel in (BitsAndBytesLinear, BitblasLinear):
                ok, error = kernel.validate_once()
                assert ok is False, f"{kernel.__name__}.validate_once() returned {ok!r}"
                assert isinstance(error, BaseException), error
            """
        )
    )


def test_monkeypatch_undo_restores_the_optional_probe():
    """Patching a legacy name must survive ``undo()`` and stay re-armable.

    The name is materialized as a normal module attribute on first read, so
    pytest records the real old value and restores it on ``undo()``. The
    previous private-store design made pytest delete the name instead, leaving
    the module permanently in a "deleted" state and breaking later imports and
    default-raising patches.
    """

    from _pytest.monkeypatch import MonkeyPatch

    import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module
    import gptqmodel.nn_modules.qlinear.bitsandbytes as bnb_module

    for module, name, accessor, probe in (
        (bnb_module, "BITSANDBYTES_AVAILABLE", bnb_module.bitsandbytes_available,
         bnb_module._probe_bitsandbytes_available),
        (bitblas_module, "BITBLAS_AVAILABLE", bitblas_module.bitblas_available,
         bitblas_module._probe_bitblas_available),
    ):
        baseline = accessor()

        # Two consecutive patch contexts must both restore the baseline.
        for _ in range(2):
            patcher = MonkeyPatch()
            try:
                patcher.setattr(module, name, not baseline)
                assert getattr(module, name) == (not baseline), module.__name__
                assert accessor() == (not baseline), module.__name__
            finally:
                patcher.undo()

            assert getattr(module, name) == baseline, module.__name__
            assert accessor() == baseline, module.__name__

        # A later default-raising patch (``tests/test_bitblas.py``) must work.
        with MonkeyPatch.context() as patcher:
            patcher.setattr(module, name, True)
        assert accessor() == baseline, module.__name__

        # ``cache_clear()`` still drops the materialized value and re-arms.
        probe.cache_clear()
        assert name not in module.__dict__, module.__name__
        assert accessor() == probe(), module.__name__


def test_lazy_availability_names_materialize_as_normal_attributes():
    """A read stores the probe as a normal attribute so patch/delete behave.

    Keeping the value in a private dict left ``monkeypatch`` and ``del`` with
    surprising semantics. A real attribute gives the same behaviour as the
    pre-lazy module global while the probe still runs on first access only.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as module

            baseline = module.bitblas_available()
            assert "BITBLAS_AVAILABLE" not in module.__dict__

            value = module.BITBLAS_AVAILABLE
            assert value == baseline
            assert module.__dict__["BITBLAS_AVAILABLE"] is value

            module.BITBLAS_AVAILABLE = not baseline
            assert module.bitblas_available() == (not baseline)
            assert module.BITBLAS_AVAILABLE == (not baseline)

            del module.BITBLAS_AVAILABLE
            assert "BITBLAS_AVAILABLE" not in module.__dict__
            assert module.bitblas_available() == baseline

            # The next attribute read re-probes and materializes again.
            assert module.BITBLAS_AVAILABLE == baseline
            assert module.__dict__["BITBLAS_AVAILABLE"] == baseline
            """
        )
    )


def test_transient_kernel_import_failure_is_retried():
    """A kernel that fails to import once must not be hidden forever.

    ``_import_all_qlinear_kernels`` used to be ``lru_cache``d, so a transient
    ``ImportError``/``OSError`` (shared library not on the loader path yet,
    extension installed later, ``sys.modules`` poisoned by another test) was
    remembered for the life of the process, and clearing the map cache did not
    re-arm it.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            victim = "qlinear.swordfish"
            broken = [True]
            real_import_module = importlib.import_module

            def flaky_import_module(name, *args, **kwargs):
                if name.endswith(victim) and broken[0]:
                    raise OSError("transient failure")
                return real_import_module(name, *args, **kwargs)

            importlib.import_module = flaky_import_module
            try:
                try:
                    importer.get_kernel_for_backend(BACKEND.SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ)
                except ValueError:
                    pass
                else:
                    raise AssertionError("expected the broken import to hide the kernel")
            finally:
                importlib.import_module = real_import_module

            # The environment is fixed now; the retry must find the kernel even
            # without clearing any cache.
            broken[0] = False
            kernel = importer.get_kernel_for_backend(
                BACKEND.SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ
            )
            assert kernel.__name__ == "SwordfishLinear", kernel
            """
        )
    )


def test_recovered_kernel_import_refreshes_cached_maps():
    """Maps built while an import failed must not stay incomplete.

    The retry above only exercised ``get_kernel_for_backend()``. The cached
    AUTO/support maps were built while the import was broken, so they kept
    omitting the kernel even after it recovered and AUTO selection still could
    not use the backend. A cache hit must re-attempt the failed import.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            victim = "qlinear.swordfish"
            broken = [True]
            real_import_module = importlib.import_module

            def flaky_import_module(name, *args, **kwargs):
                if name.endswith(victim) and broken[0]:
                    raise OSError("transient failure")
                return real_import_module(name, *args, **kwargs)

            def swordfish_backends():
                _, supports = importer.build_kernel_support_maps()
                return supports.get(METHOD.GPTQ, {}).get(FORMAT.GPTQ, [])

            importlib.import_module = flaky_import_module
            try:
                assert BACKEND.SWORDFISH not in swordfish_backends(), (
                    "the broken import unexpectedly registered the kernel"
                )
            finally:
                importlib.import_module = real_import_module

            broken[0] = False
            # The failure is backing off; make the retry due now instead of
            # sleeping in the test.
            importer._QLINEAR_RETRY_AT["swordfish"] = 0.0

            # Reading the maps alone must retry the failed import and rebuild.
            assert BACKEND.SWORDFISH in swordfish_backends(), swordfish_backends()
            auto, _ = importer.build_kernel_support_maps()
            assert auto[METHOD.GPTQ][FORMAT.GPTQ][BACKEND.SWORDFISH].__name__ == (
                "SwordfishLinear"
            )
            """
        )
    )


def test_registry_build_reentry_reports_instead_of_deadlocking():
    """Reading the maps from a kernel import must fail fast, not hang.

    The build lock is a plain ``Lock`` held across every kernel import. A kernel
    that touches the auto-select map while being imported would re-enter it and
    hang forever; it now raises a ``RuntimeError`` that names the mistake.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer

            real_import_module = importlib.import_module

            def reentrant_import_module(name, *args, **kwargs):
                if name.endswith("qlinear.swordfish"):
                    importer.get_auto_backend_kernel_mapping()
                return real_import_module(name, *args, **kwargs)

            importlib.import_module = reentrant_import_module
            try:
                try:
                    importer.build_kernel_support_maps()
                except RuntimeError as exc:
                    assert "re-entered" in str(exc), exc
                else:
                    raise AssertionError("re-entry was not reported")
            finally:
                importlib.import_module = real_import_module
                importer.build_kernel_support_maps.cache_clear()
            """
        )
    )


def test_kernels_defined_in_the_qlinear_package_init_are_registered():
    """A kernel defined in ``qlinear/__init__.py`` must reach the maps.

    The package filter compared against ``"gptqmodel.nn_modules.qlinear."`` with
    a trailing dot, so a class whose ``__module__`` is exactly the package name
    was discovered but silently dropped from auto-selection.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.utils.importer as importer
            from gptqmodel.nn_modules.qlinear import BaseQuantLinear

            class PackageLevelKernel(BaseQuantLinear):
                SUPPORTS_FORMATS = {}
                SUPPORTS_BACKENDS = []
                SUPPORTS_METHODS = []

            # Simulate a kernel that ships in the package ``__init__``.
            PackageLevelKernel.__module__ = "gptqmodel.nn_modules.qlinear"

            assert PackageLevelKernel in importer.iter_quant_linear_kernels()
            assert PackageLevelKernel in importer._registry_kernel_classes()
            """
        )
    )


def test_legacy_map_names_accept_module_attribute_overrides(monkeypatch):
    """``monkeypatch.setattr(importer, NAME, ...)`` must reach the selectors.

    Both names used to be plain module globals that ``select_quant_linear()``
    read directly. Now that they resolve through an accessor, a real module
    attribute has to win over the cache, or patching them silently stops
    affecting behaviour.
    """

    import gptqmodel.utils.importer as importer

    sentinel = {"patched": True}
    monkeypatch.setattr(importer, "AUTO_BACKEND_KERNEL_MAPPING", sentinel)
    assert importer.get_auto_backend_kernel_mapping() is sentinel

    monkeypatch.delenv("GPTQMODEL_DISABLE_BITBLAS", raising=False)
    monkeypatch.undo()
    assert importer.get_auto_backend_kernel_mapping() is not sentinel
    assert importer.get_auto_backend_kernel_mapping()


def test_map_reset_drops_a_map_restored_by_monkeypatch_undo():
    """``undo()`` writes the cached map back as a real attribute.

    A real attribute shadows the accessor's cache lookup, so without cleanup the
    selectors keep reading the old map after a rebuild and the two accessors can
    disagree. Distinct explicit overrides must survive the reset.

    This runs in a subprocess: it rebuilds the module-level map, so doing it
    in-process would invalidate map objects already imported by other test
    modules (``tests/kernels/test_selection.py`` imports the mapping directly).
    """

    _assert_ok(
        _run(
            """
            from _pytest.monkeypatch import MonkeyPatch

            import gptqmodel.utils.importer as importer

            cached_auto = importer.get_auto_backend_kernel_mapping()
            cached_supports = importer.get_backend_to_method_format_mapping()

            patcher = MonkeyPatch()
            patcher.setattr(importer, "AUTO_BACKEND_KERNEL_MAPPING", {"patched": True})
            patcher.setattr(importer, "BACKEND_TO_METHOD_FORMAT_MAPPING", {"patched": True})
            patcher.undo()
            if importer.__dict__.get("AUTO_BACKEND_KERNEL_MAPPING") is not cached_auto:
                # pytest>=9 records ``__dict__`` so ``undo`` deletes the entry;
                # older releases write the pre-patch value back. Either way the
                # reset must not leave the stale map behind.
                importer.__dict__["AUTO_BACKEND_KERNEL_MAPPING"] = cached_auto
                importer.__dict__["BACKEND_TO_METHOD_FORMAT_MAPPING"] = cached_supports

            importer.build_kernel_support_maps.cache_clear()

            assert "AUTO_BACKEND_KERNEL_MAPPING" not in importer.__dict__
            assert "BACKEND_TO_METHOD_FORMAT_MAPPING" not in importer.__dict__
            rebuilt_auto = importer.get_auto_backend_kernel_mapping()
            assert rebuilt_auto is importer.build_kernel_support_maps()[0]
            assert importer.get_backend_to_method_format_mapping() is importer.build_kernel_support_maps()[1]

            # A distinct explicit override is not the cached map and must be kept.
            distinct = {"patched": True}
            importer.__dict__["AUTO_BACKEND_KERNEL_MAPPING"] = distinct
            try:
                importer.build_kernel_support_maps.cache_clear()
                assert importer.__dict__["AUTO_BACKEND_KERNEL_MAPPING"] is distinct
                assert importer.get_auto_backend_kernel_mapping() is distinct
            finally:
                importer.__dict__.pop("AUTO_BACKEND_KERNEL_MAPPING", None)
                importer.build_kernel_support_maps.cache_clear()
            """
        )
    )

def test_deleting_a_lazy_availability_name_rearms_the_probe():
    """``del mod.X_AVAILABLE`` removes the materialized value like a global.

    The next read probes again; reassignment and deletion keep working.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as module

            baseline = module.bitblas_available()
            module.BITBLAS_AVAILABLE = not baseline
            assert module.BITBLAS_AVAILABLE == (not baseline)

            delattr(module, "BITBLAS_AVAILABLE")
            assert "BITBLAS_AVAILABLE" not in module.__dict__
            assert module.bitblas_available() == baseline

            # Re-reading materializes the probe again.
            assert module.BITBLAS_AVAILABLE == baseline
            assert module.__dict__["BITBLAS_AVAILABLE"] == baseline

            # Reassigning and deleting keep working.
            module.BITBLAS_AVAILABLE = not baseline
            assert module.BITBLAS_AVAILABLE == (not baseline)
            del module.BITBLAS_AVAILABLE
            assert "BITBLAS_AVAILABLE" not in module.__dict__
            """
        )
    )


def test_real_module_attribute_wins_over_the_private_override():
    """``vars(mod)[NAME] = ...`` must behave like the pre-lazy module global.

    The accessor reads a private override dict, while the attribute read goes
    through ``__getattr__``. Without the ``__dict__`` check first, writing the
    real attribute directly made the attribute and the accessor disagree, and
    production code (which only calls the accessor) ignored the value.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as module

            baseline = module.bitblas_available()
            module.__dict__["BITBLAS_AVAILABLE"] = not baseline
            try:
                assert module.BITBLAS_AVAILABLE == (not baseline)
                assert module.bitblas_available() == (not baseline), (
                    "the accessor ignored a real module attribute"
                )
            finally:
                del module.__dict__["BITBLAS_AVAILABLE"]
            assert module.bitblas_available() == baseline
            """
        )
    )


def test_assignment_and_delete_act_on_the_materialized_attribute():
    """Assignment and ``del`` must both act on the same normal attribute."""

    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as module

            baseline = module.bitblas_available()
            module.__dict__["BITBLAS_AVAILABLE"] = not baseline
            assert module.bitblas_available() == (not baseline)

            module.BITBLAS_AVAILABLE = baseline
            assert module.__dict__["BITBLAS_AVAILABLE"] is baseline
            assert module.bitblas_available() == baseline
            assert module.BITBLAS_AVAILABLE is baseline

            del module.BITBLAS_AVAILABLE
            assert "BITBLAS_AVAILABLE" not in module.__dict__
            assert module.bitblas_available() == baseline
            """
        )
    )


def test_availability_reset_clears_real_overrides_and_validation_state():
    """``cache_clear()`` must re-arm availability *and* kernel validation.

    A materialized ``vars(mod)[NAME]`` value outranks the probe, so the reset
    has to remove it or availability stays pinned. ``cached_validate_once()``
    also remembered a pre-recovery failure, so a recovered backend stayed
    unselectable.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as module

            baseline = module.bitblas_available()
            module.__dict__["BITBLAS_AVAILABLE"] = not baseline
            module.BITBLAS_AVAILABLE = not baseline
            assert module.bitblas_available() == (not baseline)

            module._probe_bitblas_available.cache_clear()

            assert "BITBLAS_AVAILABLE" not in module.__dict__, (
                "cache_clear() left the real module attribute behind"
            )
            assert module.bitblas_available() == baseline
            assert module._probe_bitblas_available.cache_info().currsize == 1

            # The validation cache is keyed on the kernel's methods, so seed it
            # with a failure and confirm the reset re-probes.
            from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear

            module.BITBLAS_AVAILABLE = False
            assert BitblasLinear.cached_validate_once()[0] is False
            module.BITBLAS_AVAILABLE = True
            module._probe_bitblas_available.cache_clear()
            assert BitblasLinear.cached_validate_once.cache_info().currsize == 0
            """
        )
    )


def test_import_bitblas_reports_a_missing_version_as_import_error():
    """A backend without ``__version__`` must not leak ``AttributeError``.

    ``import_bitblas()`` parsed ``bitblas.__version__`` directly, so a forced
    ``BITBLAS_AVAILABLE = True`` with a version-less package raised
    ``AttributeError`` from inside the kernel constructor instead of the
    documented ``ImportError(BITBLAS_INSTALL_HINT)``.
    """

    _assert_ok(
        _run(
            """
            import sys
            import types

            sys.modules["bitblas"] = types.ModuleType("bitblas")

            import gptqmodel.nn_modules.qlinear.bitblas as module

            module.BITBLAS_AVAILABLE = True
            try:
                module.import_bitblas()
            except ImportError as exc:
                assert "BitBLAS" in str(exc) or "bitblas" in str(exc), exc
            else:
                raise AssertionError("expected ImportError for a version-less bitblas")
            """
        )
    )


def test_falsy_bitblas_override_keeps_the_pre_lazy_semantics():
    """An explicit ``None`` override means "unknown", not "absent".

    ``models/base.py`` and ``models/loader.py`` used ``BITBLAS_AVAILABLE is
    False``; switching to ``not bitblas_available()`` made ``None``/``0`` raise
    the install hint where the previous release did not.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as module

            module.BITBLAS_AVAILABLE = None
            assert module.BITBLAS_AVAILABLE is None
            # ``is False`` is what the call sites use; ``not`` would be True here.
            assert (module.BITBLAS_AVAILABLE is False) is False
            assert not module.bitblas_available()  # the accessor still says unusable
            del module.BITBLAS_AVAILABLE
            """
        )
    )


def test_missing_optional_kernels_do_not_retry_on_every_map_lookup():
    """A permanently absent optional dependency must not poison the cache.

    ``mlx``/``tvm`` are not installed on a CUDA host, so the MLX kernel modules
    and the BitBLAS target detector always fail their top-level import. Marking
    that as "incomplete registry" made every map accessor hit take the global
    build lock and retry all of them, adding failing imports to each kernel
    selection and packing call.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer

            attempts = {"n": 0}
            real_import_module = importlib.import_module


            def counting_import_module(name, *args, **kwargs):
                if name.startswith("gptqmodel.nn_modules.qlinear."):
                    attempts["n"] += 1
                return real_import_module(name, *args, **kwargs)


            importlib.import_module = counting_import_module
            try:
                importer.build_kernel_support_maps()
                assert not importer._retryable_imports_pending(), (
                    "a stable environment must not look permanently incomplete: "
                    f"{importer._QLINEAR_FAILED!r}"
                )
                assert all(
                    kind == "optional" for kind in importer._QLINEAR_FAILED.values()
                ), importer._QLINEAR_FAILED

                # Everything is imported now; further lookups must be free.
                before = attempts["n"]
                for _ in range(50):
                    importer.get_auto_backend_kernel_mapping()
                assert attempts["n"] == before, (
                    f"{attempts['n'] - before} kernel imports re-attempted on 50 cached lookups"
                )
            finally:
                importlib.import_module = real_import_module
            """
        )
    )


def test_recovery_through_discovery_refreshes_maps_without_cache_clear():
    """A kernel recovered by ``get_kernel_for_backend()`` must reach AUTO maps.

    The retry path only re-imported the module; the maps built while it failed
    still omitted the backend, so AUTO selection could not use it until an
    explicit ``cache_clear()``.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            victim = "qlinear.swordfish"
            broken = [True]
            real_import_module = importlib.import_module


            def flaky_import_module(name, *args, **kwargs):
                if name.endswith(victim) and broken[0]:
                    raise OSError("transient failure")
                return real_import_module(name, *args, **kwargs)


            importlib.import_module = flaky_import_module
            try:
                _, supports = importer.build_kernel_support_maps()
                assert BACKEND.SWORDFISH not in supports.get(METHOD.GPTQ, {}).get(FORMAT.GPTQ, [])
            finally:
                importlib.import_module = real_import_module

            broken[0] = False
            # Recover through the OTHER public entry point, not the map accessor.
            kernel = importer.get_kernel_for_backend(
                BACKEND.SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ
            )
            assert kernel.__name__ == "SwordfishLinear", kernel

            auto, supports = importer.build_kernel_support_maps()
            assert BACKEND.SWORDFISH in supports.get(METHOD.GPTQ, {}).get(FORMAT.GPTQ, []), supports
            assert auto[METHOD.GPTQ][FORMAT.GPTQ][BACKEND.SWORDFISH] is kernel
            """
        )
    )


def test_availability_reset_clears_the_selector_validation_cache():
    """Clearing a backend probe must also drop ``validate_quant_linear`` results.

    The kernel-level ``cached_validate_once`` was cleared, but the selector's
    bounded cache still short-circuited before ``validate()`` was called again,
    so a backend that appeared later stayed "unavailable" for the process.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.utils.importer as importer
            from gptqmodel.nn_modules.qlinear import BaseQuantLinear

            state = {"ok": False}


            class SyntheticLinear(BaseQuantLinear):
                SUPPORTS_FORMATS = {}
                SUPPORTS_BACKENDS = []
                SUPPORTS_METHODS = []

                @classmethod
                def validate_once(cls):
                    return (True, None) if state["ok"] else (False, ImportError("missing"))

                # Override the full contract so the test exercises only the
                # selector cache, not unrelated parameter checks on a stub.
                @classmethod
                def validate(cls, **kwargs):
                    ok, err = cls.cached_validate_once()
                    return (True, None) if ok else (False, err)


            SyntheticLinear.__module__ = "gptqmodel.nn_modules.qlinear.synthetic"
            kwargs = dict(bits=4, group_size=128, desc_act=False, sym=True)

            assert importer.validate_quant_linear(SyntheticLinear, **kwargs)[0] is False
            assert len(importer._VALIDATION_CACHE) == 1

            # The documented reset knob for an optional backend must invalidate
            # the selector cache too.
            import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module

            bitblas_module._probe_bitblas_available.cache_clear()
            assert len(importer._VALIDATION_CACHE) == 0, (
                "bitblas reset left a stale selector validation result"
            )

            import gptqmodel.nn_modules.qlinear.bitsandbytes as bnb_module

            importer.validate_quant_linear(SyntheticLinear, **kwargs)
            assert len(importer._VALIDATION_CACHE) == 1
            bnb_module._probe_bitsandbytes_available.cache_clear()
            assert len(importer._VALIDATION_CACHE) == 0, (
                "bitsandbytes reset left a stale selector validation result"
            )

            # And a recovered backend is selectable afterwards. The kernel's
            # own ``cached_validate_once`` lru_cache is per class, so reset it
            # for the synthetic class as a backend reset would for a real one.
            state["ok"] = True
            SyntheticLinear.cached_validate_once.cache_clear()
            assert importer.validate_quant_linear(SyntheticLinear, **kwargs)[0] is True
            """
        )
    )


def test_registry_reset_does_not_race_an_in_progress_build():
    """``cache_clear()`` must take the build lock, not overwrite live metadata."""

    _assert_ok(
        _run(
            """
            import threading

            import gptqmodel.utils.importer as importer

            entered = threading.Event()
            release = threading.Event()
            errors = []
            real_import_module = importer.importlib.import_module


            def slow_import_module(name, *args, **kwargs):
                if name.endswith("qlinear.swordfish") and not entered.is_set():
                    entered.set()
                    release.wait(timeout=30)
                return real_import_module(name, *args, **kwargs)


            importer.importlib.import_module = slow_import_module
            try:
                builder = threading.Thread(target=importer.build_kernel_support_maps)
                builder.start()
                assert entered.wait(timeout=30)

                clearer = threading.Thread(target=importer.build_kernel_support_maps.cache_clear)
                clearer.start()
                clearer.join(timeout=0.5)
                # The reset must block while the build owns the lock.
                assert clearer.is_alive(), "cache_clear() ran during an in-progress build"

                release.set()
                builder.join(timeout=60)
                clearer.join(timeout=60)
                assert not (builder.is_alive() or clearer.is_alive())
            finally:
                importer.importlib.import_module = real_import_module
            """
        )
    )


def test_retryable_failure_recovery_is_never_permanently_blocked():
    """Backoff must not become a permanent retry budget.

    A shared library that appears later has to stay discoverable, but a
    persistently failing optional stack must not be re-imported on every map
    lookup. Failures therefore back off exponentially with no attempt limit;
    an explicit discovery call bypasses the wait.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            victim = "qlinear.swordfish"
            broken = [True]
            attempts = {"n": 0}
            real_import_module = importlib.import_module


            def flaky_import_module(name, *args, **kwargs):
                if name.endswith(victim):
                    attempts["n"] += 1
                    if broken[0]:
                        raise OSError("transient failure")
                return real_import_module(name, *args, **kwargs)


            importlib.import_module = flaky_import_module
            try:
                for _ in range(10):
                    importer.build_kernel_support_maps()
                    assert importer._QLINEAR_FAILED.get("swordfish") == "retryable"
                    assert importer._QLINEAR_RETRY_AT["swordfish"] > 0.0
            finally:
                importlib.import_module = real_import_module

            # The backoff must suppress the failing import on cached lookups.
            assert attempts["n"] < 10, attempts

            broken[0] = False
            # An explicit lookup bypasses the backoff and recovers immediately.
            kernel = importer.get_kernel_for_backend(
                BACKEND.SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ
            )
            auto, supports = importer.build_kernel_support_maps()
            assert kernel.__name__ == "SwordfishLinear", kernel
            assert BACKEND.SWORDFISH in supports[METHOD.GPTQ][FORMAT.GPTQ]
            assert auto[METHOD.GPTQ][FORMAT.GPTQ][BACKEND.SWORDFISH] is kernel
            """
        )
    )


def test_concurrent_discovery_is_serialized_and_keeps_state_consistent():
    """Discovery bookkeeping is shared, so entry points must serialize.

    Two concurrent ``_import_all_qlinear_kernels()`` passes used to race the
    membership check against the failure bookkeeping: one could record a
    failure for a module another had just imported, leaving a permanent
    retryable entry (and a dict-mutation ``RuntimeError`` in the scan).
    """

    _assert_ok(
        _run(
            """
            import threading
            import importlib

            import gptqmodel.utils.importer as importer

            victim = "qlinear.swordfish"
            real_import_module = importlib.import_module
            b_entered = threading.Event()
            release_b = threading.Event()
            a_done = threading.Event()
            b_thread = None


            def slow_import_module(name, *args):
                if name.endswith(victim) and threading.current_thread() is b_thread:
                    b_entered.set()
                    release_b.wait(timeout=30)
                    raise OSError("stale failure from the other thread")
                return real_import_module(name, *args)


            def run_b():
                importer._import_all_qlinear_kernels(force_retry=True)


            def run_a():
                b_entered.wait(timeout=30)
                importer._import_all_qlinear_kernels(force_retry=True)
                a_done.set()


            importlib.import_module = slow_import_module
            try:
                b_thread = threading.Thread(target=run_b)
                b_thread.start()
                assert b_entered.wait(timeout=30)
                a_thread = threading.Thread(target=run_a)
                a_thread.start()
                # A must not interleave with B's in-flight import.
                a_done.wait(timeout=0.3)
                assert not a_done.is_set(), "discovery passes interleaved"
                release_b.set()
                a_thread.join(timeout=60)
                b_thread.join(timeout=60)
                assert not (a_thread.is_alive() or b_thread.is_alive())
            finally:
                importlib.import_module = real_import_module

            assert "swordfish" in importer._QLINEAR_IMPORTED
            assert "swordfish" not in importer._QLINEAR_FAILED
            """
        )
    )


def test_warm_map_lookup_cannot_race_a_reset():
    """A warm cache hit must still be synchronized with ``cache_clear()``.

    The fast path used to check the cache and then call the ``lru_cache``
    outside the build lock. A reset in between turned that hit into an
    unlocked miss, so two builders could publish different map objects and
    race the snapshot. Every retrieval now runs under the build lock.
    """

    _assert_ok(
        _run(
            """
            import threading

            import gptqmodel.utils.importer as importer

            real_cached = importer._build_kernel_support_maps_cached
            owners = []


            def checked_cached():
                owners.append(importer._REGISTRY_BUILD_OWNER == threading.get_ident())
                return real_cached()


            checked_cached.cache_info = real_cached.cache_info
            checked_cached.cache_clear = real_cached.cache_clear
            importer._build_kernel_support_maps_cached = checked_cached
            try:
                importer.build_kernel_support_maps()   # cold build
                for _ in range(20):
                    importer.build_kernel_support_maps()  # warm hits
                assert owners and all(owners), owners
            finally:
                importer._build_kernel_support_maps_cached = real_cached
            """
        )
    )


def test_in_flight_validation_does_not_publish_after_reset():
    """A reset during ``validate()`` must not be undone by the late result.

    ``validate_quant_linear()`` releases the cache lock while the kernel
    validates. A backend reset in that window used to be followed by the stale
    failure being stored, so the recovered backend stayed unselectable.
    """

    _assert_ok(
        _run(
            """
            import threading

            import gptqmodel.utils.importer as importer
            from gptqmodel.nn_modules.qlinear import BaseQuantLinear

            state = {"ok": False}
            entered = threading.Event()
            release = threading.Event()
            result = {}


            class SyntheticLinear(BaseQuantLinear):
                SUPPORTS_FORMATS = {}
                SUPPORTS_BACKENDS = []
                SUPPORTS_METHODS = []

                @classmethod
                def validate_once(cls):
                    return (True, None) if state["ok"] else (False, ImportError("missing"))

                @classmethod
                def validate(cls, **kwargs):
                    entered.set()
                    release.wait(timeout=30)
                    ok, err = cls.cached_validate_once()
                    return (True, None) if ok else (False, err)


            SyntheticLinear.__module__ = "gptqmodel.nn_modules.qlinear.synthetic"
            kwargs = dict(bits=4, group_size=128, desc_act=False, sym=True)


            def run():
                result["value"] = importer.validate_quant_linear(SyntheticLinear, **kwargs)


            thread = threading.Thread(target=run)
            thread.start()
            assert entered.wait(timeout=30)
            importer.clear_validation_cache()
            release.set()
            thread.join(timeout=60)
            assert not thread.is_alive()

            # The in-flight answer predates the reset and must not be cached.
            assert result["value"][0] is False
            assert len(importer._VALIDATION_CACHE) == 0, importer._VALIDATION_CACHE
            assert SyntheticLinear.cached_validate_once.cache_info().currsize == 0

            # The reset re-armed the backend, so a fresh validation sees it.
            state["ok"] = True
            assert importer.validate_quant_linear(SyntheticLinear, **kwargs)[0] is True
            """
        )
    )


def test_explicit_lookup_imports_a_pending_failure_once():
    """``get_kernel_for_backend()`` must not run the same failing import twice.

    ``iter_quant_linear_kernels()`` already performs discovery; the explicit
    call in ``get_kernel_for_backend()`` doubled the cost of every lookup while
    a retryable failure was pending.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            victim = "qlinear.swordfish"
            attempts = {"n": 0}
            real_import_module = importlib.import_module


            def flaky_import_module(name, *args):
                if name.endswith(victim):
                    attempts["n"] += 1
                    raise OSError("still broken")
                return real_import_module(name, *args)


            importlib.import_module = flaky_import_module
            try:
                try:
                    importer.get_kernel_for_backend(
                        BACKEND.SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError("expected the broken import to hide the kernel")
            finally:
                importlib.import_module = real_import_module

            assert attempts["n"] == 1, attempts
            """
        )
    )


def test_blocked_module_entry_is_retried_not_skipped():
    """``sys.modules[name] = None`` is a temporary block, not a missing dep.

    CPython raises ``ModuleNotFoundError(name="triton")`` for a blocked entry,
    which the classifier used to treat as a permanently absent optional
    dependency, hiding the kernel even after the block was removed.
    """

    _assert_ok(
        _run(
            """
            import importlib
            import sys

            import gptqmodel.utils.importer as importer
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            victim = "qlinear.swordfish"
            real_import_module = importlib.import_module


            def blocked_import_module(name, *args):
                if name.endswith(victim):
                    sys.modules["triton"] = None
                    real_import_module("triton")  # raises ModuleNotFoundError(name="triton")
                return real_import_module(name, *args)


            importlib.import_module = blocked_import_module
            try:
                try:
                    importer.get_kernel_for_backend(
                        BACKEND.SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError("expected the blocked import to hide the kernel")
                assert importer._QLINEAR_FAILED.get("swordfish") == "retryable"
            finally:
                sys.modules.pop("triton", None)
                importlib.import_module = real_import_module

            # Removing the block must make the kernel discoverable again.
            kernel = importer.get_kernel_for_backend(
                BACKEND.SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ
            )
            assert kernel.__name__ == "SwordfishLinear", kernel
            """
        )
    )


def test_map_snapshot_is_taken_after_the_recovery_pass():
    """A module that recovers during the build must be in the snapshot.

    The import-state snapshot used to be recorded before
    ``_registry_kernel_classes()`` ran its own discovery pass. A module that
    failed once and succeeded on that pass appeared in the returned map but not
    in the snapshot, so the next lookup rebuilt and discarded the map object.
    """

    _assert_ok(
        _run(
            """
            import importlib

            import gptqmodel.utils.importer as importer
            from gptqmodel.quantization import FORMAT, METHOD
            from gptqmodel.utils.backend import BACKEND

            victim = "qlinear.swordfish"
            calls = {"n": 0}
            real_import_module = importlib.import_module


            def fail_once_import_module(name, *args):
                if name.endswith(victim):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise OSError("transient failure")
                return real_import_module(name, *args)


            importlib.import_module = fail_once_import_module
            try:
                auto, supports = importer.build_kernel_support_maps()
            finally:
                importlib.import_module = real_import_module

            assert calls["n"] == 2, calls
            assert "swordfish" in importer._BUILT_MODULES
            assert BACKEND.SWORDFISH in supports[METHOD.GPTQ][FORMAT.GPTQ]
            # The snapshot matches the map, so the next lookup must reuse it.
            assert importer.build_kernel_support_maps()[0] is auto
            """
        )
    )
