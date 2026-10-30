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
    """Patching a legacy name must not pin the probe for the rest of the process.

    ``monkeypatch.undo()`` writes the pre-patch value back as a real module
    attribute. If the accessor prefers ``globals()``, that stale value wins
    forever and defeats both ``cache_clear()`` and the disable env flag.
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
        patcher = MonkeyPatch()
        try:
            patcher.setattr(module, name, not baseline, raising=False)
            assert accessor() == (not baseline), module.__name__
        finally:
            patcher.undo()

        assert name not in module.__dict__, f"{module.__name__} leaked a real attribute"
        # ``undo()`` writes the probe's own value back, which is
        # indistinguishable from an explicit override of that value. The bug was
        # that the stale value then survived ``cache_clear()`` and hid every
        # later environment change. Clearing must now re-arm the probe.
        probe.cache_clear()
        assert module._AVAILABILITY_OVERRIDES == {}, (
            f"{module.__name__} kept {module._AVAILABILITY_OVERRIDES!r} after cache_clear()"
        )
        assert accessor() == probe(), f"{module.__name__} did not fall back to the probe"


def test_overrides_stay_out_of_module_dict_and_delete_cleanly():
    """Overrides must not become permanent module attributes.

    An ordinary ``setattr`` used to land in ``__dict__``, where the accessor
    preferred it over the probe forever. Overrides now live in a private dict,
    so ``__dict__`` stays clean and ``del`` re-arms the real probe.
    """

    import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module

    baseline = bitblas_module.bitblas_available()
    bitblas_module.BITBLAS_AVAILABLE = not baseline
    try:
        assert bitblas_module.bitblas_available() == (not baseline)
        assert "BITBLAS_AVAILABLE" not in bitblas_module.__dict__, (
            "override leaked into the module __dict__ and would outlive a delete"
        )
    finally:
        del bitblas_module.BITBLAS_AVAILABLE

    assert bitblas_module.bitblas_available() == baseline
    assert "BITBLAS_AVAILABLE" not in bitblas_module.__dict__
    assert bitblas_module.__dict__.get("_AVAILABILITY_DELETED") is True


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
                    importer.get_kernel_for_backend(BACKEND.GPTQ_SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ)
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
                BACKEND.GPTQ_SWORDFISH, METHOD.GPTQ, FORMAT.GPTQ
            )
            assert kernel.__name__ == "SwordfishLinear", kernel
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


def test_deleting_a_lazy_availability_name_restores_the_probe():
    """``del mod.X_AVAILABLE`` must work and re-arm the probe, like a real attribute.

    ``hasattr()`` is always True for the lazy name, so ``delattr`` used to raise
    ``AttributeError`` even though the attribute was readable. It now drops the
    override (or the real attribute) and the accessor falls back to the probe.
    """

    _assert_ok(
        _run(
            """
            import gptqmodel.nn_modules.qlinear.bitblas as module

            baseline = module.bitblas_available()
            module.BITBLAS_AVAILABLE = not baseline
            assert module.bitblas_available() == (not baseline)

            delattr(module, "BITBLAS_AVAILABLE")
            assert module.bitblas_available() == baseline
            assert module._AVAILABILITY_OVERRIDES == {}

            # Before the lazy change the name was a real attribute, so a delete
            # really removed it and reading it raised.
            try:
                module.BITBLAS_AVAILABLE
            except AttributeError:
                pass
            else:
                raise AssertionError("the deleted name is still readable")

            # Reassigning re-arms it.
            module.BITBLAS_AVAILABLE = not baseline
            assert module.BITBLAS_AVAILABLE == (not baseline)
            del module.BITBLAS_AVAILABLE
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
