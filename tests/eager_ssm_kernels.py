# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pin the Torch SSM/DeltaNet references for CPU forwards inside one test.

`transformers.use_kernel_func_from_hub_with_fallback` resolves
`causal_conv1d`/`mamba_ssm`/`fla` at import time and prefers the installed
package over its own Torch reference. Those implementations are CUDA-only, so a
host that merely happens to have the package installed hands CPU tensors to a
CUDA kernel and fails on the device instead of on the invariant under test.

`functools.wraps` keeps the undecorated Torch reference on `__wrapped__`. This
context manager swaps it in for the duration of the block and always restores
the installed wrapper, so a test that does verify a CUDA kernel is unaffected.
"""

import builtins
import importlib
import importlib.util
import sys
from contextlib import ContextDecorator


# Every package the decorator can resolve. A wrapper's closure names the
# package it resolved, so this identifies them without a hand-kept function
# list (which silently missed `mamba_selective_scan` and friends).
_KERNEL_PACKAGES = frozenset({"causal_conv1d", "mamba_ssm", "fla"})

# Only modeling modules own these globals; other imported packages may reuse
# the same names for an unrelated API.
_MODULE_PREFIXES = (
    "transformers.models.",
    "gptqmodel.models.ports.",
)


def _resolved_package(wrapper):
    """The kernel package a decorated function resolved, or None."""

    closure = getattr(wrapper, "__closure__", None)
    if closure is None:
        return None
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:  # empty cell
            continue
        if isinstance(value, str) and value in _KERNEL_PACKAGES:
            return value
    return None


def _candidate_modules(families):
    """Imported modeling modules, optionally narrowed to `modeling_<family>`."""

    if families:
        wanted = tuple(f"modeling_{family}" for family in families)
    else:
        wanted = None
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(_MODULE_PREFIXES):
            continue
        if wanted is not None and not name.endswith(wanted):
            continue
        yield module


def _is_modeling_name(name, package=None):
    """True when `name` (absolute or relative to `package`) is a modeling module."""

    if not isinstance(name, str) or not name:
        return False
    if name.startswith("."):
        if not isinstance(package, str) or not package:
            return False
        try:
            name = importlib.util.resolve_name(name, package)
        except (ImportError, ValueError):
            return False
    return name.startswith(_MODULE_PREFIXES)


def _import_package(globals_arg):
    """Package a `__import__` relative import resolves against."""

    if not isinstance(globals_arg, dict):
        return None
    package = globals_arg.get("__package__")
    if package:
        return package
    module_name = globals_arg.get("__name__")
    if isinstance(module_name, str) and "." in module_name:
        return module_name.rpartition(".")[0]
    return None


def _import_module_aliases(original):
    """`(module, name)` bindings that hold a direct reference to `original`.

    A module that does ``from importlib import import_module`` captures the
    function object, so replacing ``importlib.import_module`` alone never
    reaches it. The vendored port registry does exactly that, which left every
    port it resolves lazily unpatched (``minicpmv4_6`` pulls ``qwen3_5`` in that
    way). Only GPT-QModel's own modules are rewritten: a third-party alias is
    none of this helper's business.
    """

    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("gptqmodel."):
            continue
        for attribute, value in list(vars(module).items()):
            if value is original:
                yield module, attribute


class eager_torch_ssm_kernels(ContextDecorator):
    """Swap installed CUDA-only SSM/DeltaNet kernels for their Torch reference.

    `families` narrows the swap to `modeling_<family>` modules (e.g. `"jamba"`).
    Omit it to pin every already-imported modeling module.

    A modeling module imported inside the block is patched as soon as its import
    finishes, so a module-level `autouse` fixture also covers ports that a test
    only imports lazily. Both import entry points are hooked because
    ``importlib.import_module`` does not go through ``builtins.__import__``.
    Those hooks are process-global, so the block must not be entered from more
    than one thread.
    """

    def __init__(self, *families):
        self.families = families
        self._saved = {}
        self._original_import = None
        self._original_import_module = None
        self._rebound_aliases = []
        self._refreshing = False

    @property
    def patched(self):
        """`module.name` for every kernel global swapped in this block."""

        return [
            f"{module.__name__}.{name}"
            for module, patched in self._saved.values()
            for name in patched
        ]

    def _refresh(self):
        for module in _candidate_modules(self.families):
            patched = self._saved.setdefault(id(module), (module, {}))[1]
            for name, value in list(vars(module).items()):
                if name in patched:
                    continue
                if getattr(value, "__wrapped__", None) is None:
                    continue
                if _resolved_package(value) is None:
                    continue
                patched[name] = value
                setattr(module, name, value.__wrapped__)

    def __enter__(self):
        self._saved = {}
        self._original_import = builtins.__import__
        self._original_import_module = importlib.import_module
        self._refreshing = True
        try:
            self._refresh()
        finally:
            self._refreshing = False
        try:
            original_import = self._original_import
            original_import_module = self._original_import_module

            def refresh_if_modeling(is_modeling):
                # Only a modeling import can add a kernel global; refreshing on
                # every import would put this scan on the hot path of unrelated
                # imports. The guard keeps `_refresh`'s own imports unpatched.
                if is_modeling and not self._refreshing:
                    self._refreshing = True
                    try:
                        self._refresh()
                    finally:
                        self._refreshing = False

            def patched_import(name, globals_arg=None, locals_arg=None, fromlist=(), level=0):
                # `__import__` receives the relative form (and the caller's
                # `__package__`), so resolve it before deciding whether to scan.
                if level:
                    is_modeling = _is_modeling_name("." * level + name, _import_package(globals_arg))
                else:
                    is_modeling = _is_modeling_name(name)
                # Hand the original the exact arguments so its own resolution and
                # `__package__` handling stay untouched.
                module = original_import(name, globals_arg, locals_arg, fromlist, level)
                refresh_if_modeling(is_modeling)
                return module

            def patched_import_module(name, package=None):
                is_modeling = _is_modeling_name(name, package)
                module = original_import_module(name, package)
                refresh_if_modeling(is_modeling)
                return module

            builtins.__import__ = patched_import
            importlib.import_module = patched_import_module
            # Rebind captured `from importlib import import_module` aliases so a
            # lazily resolving registry routes through `patched_import_module`.
            for module, attribute in _import_module_aliases(original_import_module):
                setattr(module, attribute, patched_import_module)
                self._rebound_aliases.append((module, attribute, original_import_module))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original_import is not None:
            builtins.__import__ = self._original_import
            self._original_import = None
        if self._original_import_module is not None:
            importlib.import_module = self._original_import_module
            self._original_import_module = None
        for module, attribute, original in reversed(self._rebound_aliases):
            setattr(module, attribute, original)
        self._rebound_aliases = []
        for module, patched in reversed(list(self._saved.values())):
            for name, wrapper in reversed(list(patched.items())):
                setattr(module, name, wrapper)
        self._saved = {}
        return False
