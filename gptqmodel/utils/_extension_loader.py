# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional


_TORCH_SHARED_LIBS_PRELOADED = False


def _ensure_torch_shared_libraries_loaded() -> None:
    """Load torch's shared libraries with RTLD_GLOBAL so extensions can resolve symbols."""

    global _TORCH_SHARED_LIBS_PRELOADED
    if _TORCH_SHARED_LIBS_PRELOADED:
        return

    try:
        import torch  # local import to avoid hard dependency if torch is absent
    except Exception:
        return

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib_dir.is_dir():
        return

    # Load core torch libraries first so subsequent extension loads can resolve symbols.
    load_order = (
        "libtorch_python.so",
        "libtorch_cuda.so",
        "libtorch_cpu.so",
        "libtorch.so",
        "libc10_cuda.so",
        "libc10.so",
        "libtorch_global_deps.so",
    )

    for name in load_order:
        candidate = torch_lib_dir / name
        if not candidate.is_file():
            continue
        try:
            mode = getattr(ctypes, "RTLD_GLOBAL", None)
            if mode is None:
                ctypes.CDLL(str(candidate))
            else:
                ctypes.CDLL(str(candidate), mode=mode)
        except OSError:
            # Silently ignore individual load failures; later loads may still succeed
            continue

    _TORCH_SHARED_LIBS_PRELOADED = True


def load_extension_module(module_name: str,
                          package: Optional[str] = "gptqmodel") -> ModuleType:
    """Import a compiled extension, with fallbacks for editable installs.

    Args:
        module_name: The qualified module name to import.
        package: Package hint used to derive search paths.

    Returns:
        The loaded module.

    Raises:
        ImportError: If the module cannot be located or loaded.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]

    try:
        return importlib.import_module(module_name)
    except ImportError as primary_error:
        ext_path = _resolve_extension_path(module_name, package)
        if ext_path is None:
            raise primary_error

        loader = importlib.machinery.ExtensionFileLoader(module_name,
                                                         str(ext_path))
        spec = importlib.util.spec_from_loader(module_name, loader)
        if spec is None:
            raise primary_error

        module = importlib.util.module_from_spec(spec)
        try:
            _ensure_torch_shared_libraries_loaded()
            loader.exec_module(module)
        except Exception as load_error:  # pragma: no cover - surface exact failure
            raise ImportError(
                f"Failed to load extension module {module_name} from {ext_path}: {load_error}"
            ) from load_error

        sys.modules[module_name] = module
        return module


def _resolve_extension_path(module_name: str,
                            package: Optional[str]) -> Optional[Path]:
    for directory in _candidate_directories(package):
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = directory / f"{module_name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _candidate_directories(package: Optional[str]) -> Iterable[Path]:
    seen = set()

    def _add(path: Path):
        try:
            resolved = path.resolve()
        except (FileNotFoundError, RuntimeError):
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            yield resolved

    if package:
        spec = importlib.util.find_spec(package)
        if spec:
            locations = spec.submodule_search_locations or []
            if not locations and spec.origin:
                locations = [Path(spec.origin).parent]
            for location in locations:
                location_path = Path(location)
                yield from _add(location_path)
                yield from _add(location_path.parent)

    # Fallbacks cover source checkout and editable installs.
    current = Path(__file__).resolve()
    base = current.parent.parent  # gptqmodel/
    for candidate in (
        base,
        base.parent,
        base / "lib",
        base.parent / "lib",
        base.parent / "build",
    ):
        yield from _add(candidate)
