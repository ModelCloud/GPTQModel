"""Command line interface for GPTQModel utility commands."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Dict, Iterable, List, Sequence, Tuple

from gptqmodel.version import __version__


def _format_header(title: str) -> str:
    return f"{title}\n{'-' * len(title)}"


def _collect_runtime_info() -> List[Tuple[str, str]]:
    info: List[Tuple[str, str]] = []

    info.append(("Python", sys.version.split()[0]))
    info.append(("Platform", platform.platform()))
    info.append(("GPTQModel", __version__))

    try:
        import torch

        info.append(("PyTorch", torch.__version__))

        cuda_version = torch.version.cuda or "cpu build"
        info.append(("CUDA version", cuda_version))

        cuda_available = torch.cuda.is_available()
        info.append(("CUDA available", "yes" if cuda_available else "no"))

        if cuda_available:
            device_count = torch.cuda.device_count()
            info.append(("CUDA device count", str(device_count)))

            gpu_descriptions: List[str] = []
            for idx in range(device_count):
                try:
                    name = torch.cuda.get_device_name(idx)
                    major, minor = torch.cuda.get_device_capability(idx)
                    gpu_descriptions.append(f"{idx}: {name} (cc {major}.{minor})")
                except RuntimeError as exc:  # pragma: no cover - defensive guard
                    gpu_descriptions.append(f"{idx}: <error retrieving info> ({exc})")

            if gpu_descriptions:
                info.append(("CUDA devices", "; ".join(gpu_descriptions)))
        else:
            info.append(("CUDA device count", "0"))

    except Exception as exc:  # pragma: no cover - torch should normally be installed
        info.append(("PyTorch", f"not available ({exc.__class__.__name__}: {exc})"))

    return info


def _collect_optional_dependencies() -> List[Tuple[str, str]]:
    optional_modules: Dict[str, str] = {
        "auto_round": "auto_round",
        "bitblas": "bitblas",
        "flashinfer": "flashinfer",
        "optimum": "optimum",
        "sglang": "sglang",
        "vllm": "vLLM",
    }

    results: List[Tuple[str, str]] = []
    for module_name, display_name in optional_modules.items():
        spec = find_spec(module_name)
        if spec is None:
            results.append((display_name, "not installed"))
            continue

        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive fallback
            results.append((display_name, f"installed but import failed ({exc})"))
            continue

        version = getattr(module, "__version__", None) or getattr(module, "VERSION", None)
        results.append((display_name, version or "installed"))

    return results


def _print_table(rows: Sequence[Tuple[str, str]]) -> None:
    if not rows:
        return

    padding = max(len(key) for key, _ in rows) + 2
    for key, value in rows:
        print(f"{key:<{padding}}{value}")


def _handle_env_command(_args: argparse.Namespace) -> None:
    print(_format_header("GPTQModel environment"))
    _print_table(_collect_runtime_info())

    optional = _collect_optional_dependencies()
    if optional:
        print()
        print(_format_header("Optional dependencies"))
        _print_table(optional)