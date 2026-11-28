# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Environment checks a user can paste into an issue.

Every check answers one question with OK/WARN/FAIL/INFO and a detail string. The
point is triage: a support thread should not start with three round trips asking
for the torch build, the GPU capability and whether an optional kernel is
installed.
"""

import importlib.util
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


__all__ = ["Check", "collect", "exit_code", "render", "to_json"]

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"

#: Dependencies the package cannot run without.
_REQUIRED = ("transformers", "tokenicer", "numpy", "safetensors")
#: Backends that are optional by design; absence is information, not a problem.
_OPTIONAL = ("triton", "bitsandbytes", "bitblas", "torchao", "accelerate", "humming")
_ENV_FLAGS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "HF_HUB_OFFLINE",
    "GPTQ_BYPASS_SAFETY_CHECKS",
    "PYTHON_GIL",
)


@dataclass(frozen=True)
class Check:
    area: str
    name: str
    status: str
    detail: str


def _normalized(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _installed_version(distribution: str) -> Optional[str]:
    """Installed version from metadata, without importing the module.

    ``tokenicer`` (and some optional backends) set no ``__version__``, and
    importing a backend only to read a version can initialize it. Metadata is
    the only source; an importable module without metadata reports "unknown"
    through :func:`_module_installed`.
    """

    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a broken metadata entry is "not installed"
        return None


def _module_installed(module: str) -> bool:
    return _installed_version(module) is not None or _has_module(module)


def _module_version(module: str) -> Optional[str]:
    return _installed_version(module)


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _requirements_file() -> Optional[Path]:
    """The supported-range source: the checkout's ``requirements.txt`` if present.

    Wheels cannot ship the file reliably, so the installed metadata is the
    fallback; a source checkout (where this command is most useful) always has
    the authoritative list.
    """

    for candidate in (
        Path(__file__).resolve().parents[2] / "requirements.txt",
        Path(__file__).resolve().parents[1] / "requirements.txt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _supported_specs() -> Dict[str, List[Any]]:
    """Supported requirement specifiers, by normalized package name."""

    from packaging.requirements import Requirement

    specs: Dict[str, List[Any]] = {}
    path = _requirements_file()
    lines: List[str] = []
    if path is not None:
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        try:
            lines = list(importlib_metadata.requires("gptqmodel") or [])
        except importlib_metadata.PackageNotFoundError:
            lines = []

    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            requirement = Requirement(line)
        except Exception:  # noqa: BLE001 - a comment/URL line is not a version pin
            continue
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        specs.setdefault(_normalized(requirement.name), []).append(requirement)
    return specs


def _out_of_range(module: str, version: str, specs: Dict[str, List[Any]]) -> Optional[str]:
    """The requirement violated by ``version``, or ``None`` when it is supported."""

    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion:
        return None
    for requirement in specs.get(_normalized(module), []):
        try:
            if parsed not in requirement.specifier:
                return str(requirement)
        except Exception:  # noqa: BLE001 - an unparsable spec is not a verdict
            continue
    return None


def _runtime_checks() -> List[Check]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    implementation = platform.python_implementation()
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    gil = "" if gil_probe is None else (" (no GIL)" if gil_probe() is False else " (GIL)")
    platform_name = platform.platform()
    wsl = "wsl" in platform.release().lower() or "microsoft" in platform_name.lower()
    return [
        Check("runtime", "python", OK, f"{implementation} {version}{gil}"),
        Check("runtime", "platform", INFO, f"{platform.system()} {platform.machine()}" + (" (WSL)" if wsl else "")),
    ]


def _torch_checks() -> List[Check]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - the whole report is about why this failed
        return [Check("torch", "import", FAIL, f"{type(exc).__name__}: {exc}")]

    checks = [Check("torch", "version", OK, str(torch.__version__))]
    if torch.version.cuda:
        checks.append(Check("torch", "cuda build", OK, f"CUDA {torch.version.cuda}"))
    elif torch.version.hip:
        checks.append(Check("torch", "hip build", OK, f"ROCm {torch.version.hip}"))
    else:
        checks.append(Check("torch", "accelerator build", INFO, "CPU-only torch build"))

    try:
        from .._torch_safety import (  # noqa: PLC0415 - keeps the import optional
            matching_torch_advisories,
            safety_checks_bypassed,
        )

        advisories = matching_torch_advisories()
        if not advisories:
            checks.append(Check("torch", "safety advisories", OK, "none"))
        else:
            ids = ", ".join(
                advisory.cve_id or advisory.advisory_id for advisory in advisories
            )
            # ``gptqmodel/__init__`` refuses an affected runtime before this
            # command can run, so the only reachable state is the bypassed one.
            bypassed = safety_checks_bypassed()
            checks.append(
                Check(
                    "torch",
                    "safety advisories",
                    INFO,
                    f"{ids} (checks bypassed)" if bypassed else ids,
                )
            )
    except Exception as exc:  # noqa: BLE001 - advisory data must never break the report
        checks.append(Check("torch", "safety advisories", INFO, f"unavailable: {exc}"))
    return checks


def _gpu_checks() -> List[Check]:
    try:
        import torch
    except Exception:  # noqa: BLE001 - already reported by _torch_checks
        return []

    if not torch.cuda.is_available():
        return [Check("gpu", "devices", INFO, "none available (CPU execution)")]

    is_hip = getattr(torch.version, "hip", None) is not None
    prefix = "hip" if is_hip else "cuda"
    checks = []
    for index in range(torch.cuda.device_count()):
        try:
            name = torch.cuda.get_device_name(index)
            free, total = torch.cuda.mem_get_info(index)
            if is_hip:
                # ``get_device_capability`` reports a CUDA-shaped tuple even on
                # ROCm; the architecture string is what AMD tooling names.
                arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "unknown")
            else:
                major, minor = torch.cuda.get_device_capability(index)
                arch = f"sm_{major}{minor}"
            detail = f"{name} {arch}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB free"
            checks.append(Check("gpu", f"{prefix}:{index}", OK, detail))
        except Exception as exc:  # noqa: BLE001 - one bad device must not hide the others
            checks.append(Check("gpu", f"{prefix}:{index}", WARN, f"{type(exc).__name__}: {exc}"))
    return checks


def _dependency_checks() -> List[Check]:
    specs = _supported_specs()
    checks = []
    for module in (*_REQUIRED, "torch"):
        version = _module_version(module)
        if not version:
            status = FAIL if not _module_installed(module) else WARN
            checks.append(
                Check("deps", module, status, "not installed" if status == FAIL else "version unknown")
            )
            continue
        violated = _out_of_range(module, version, specs)
        if violated is None:
            checks.append(Check("deps", module, OK, str(version)))
        else:
            checks.append(
                Check(
                    "deps",
                    module,
                    WARN,
                    f"{version} (requires {violated.specifier if hasattr(violated, 'specifier') else violated})",
                )
            )
    for module in _OPTIONAL:
        version = _module_version(module)
        if version:
            checks.append(Check("deps", module, OK, str(version)))
        elif _module_installed(module):
            checks.append(Check("deps", module, OK, "version unknown"))
        else:
            checks.append(Check("deps", module, INFO, "not installed"))
    return checks


def _torch_extension_artifacts() -> List[Path]:
    torch_extensions = Path(
        os.environ.get("TORCH_EXTENSIONS_DIR") or Path.home() / ".cache" / "torch_extensions"
    )
    try:
        return sorted(torch_extensions.glob("**/*.so"))
    except OSError:
        return []


def _extension_checks() -> List[Check]:
    """Report compiled kernels wherever this install actually puts them.

    Kernels are either shipped next to the package, installed under
    ``gptqmodel_ext``, JIT-built into torch's extension cache, or built into the
    NF12 source-keyed cache. Globbing only the package directory reported "none"
    on every JIT install.
    """

    package_dir = Path(__file__).resolve().parent.parent
    bundled = sorted(path.name for path in package_dir.glob("*.so"))

    ext_spec = importlib.util.find_spec("gptqmodel_ext")
    ext_dir = None
    if ext_spec is not None and ext_spec.origin:
        ext_dir = Path(ext_spec.origin).parent
    ext_built = sorted(path.name for path in ext_dir.glob("*.so")) if ext_dir is not None else []

    torch_built = _torch_extension_artifacts()

    nf12_ready = False
    nf12_dir = None
    try:
        from gptqmodel_nf12 import _jit  # noqa: PLC0415 - optional native package

        nf12_dir = _jit.cache_directory()
        nf12_ready = _jit.cache_ready()
    except Exception:  # noqa: BLE001 - no CUDA/tools is not an extension failure
        pass

    if bundled or ext_built:
        listed = bundled + ext_built
        shown = ", ".join(listed[:4]) + (" ..." if len(listed) > 4 else "")
        return [Check("extension", "native kernels", OK, f"{len(listed)} built ({shown})")]
    if torch_built:
        shown = ", ".join(path.parent.name for path in torch_built[:4])
        more = " ..." if len(torch_built) > 4 else ""
        return [
            Check(
                "extension",
                "native kernels",
                OK,
                f"{len(torch_built)} JIT artifact(s) in the torch extension cache ({shown}{more})",
            )
        ]
    if nf12_ready and nf12_dir is not None:
        return [Check("extension", "native kernels", OK, f"NF12 JIT cache ready ({nf12_dir})")]
    return [
        Check(
            "extension",
            "native kernels",
            INFO,
            "none built yet; torch/triton kernels still work and JIT builds on first use"
            + (f" (NF12 cache: {nf12_dir}, not built)" if nf12_dir is not None else ""),
        )
    ]


def _env_checks() -> List[Check]:
    checks = []
    for name in _ENV_FLAGS:
        value = os.environ.get(name)
        checks.append(Check("env", name, INFO, value if value is not None else "(unset)"))
    return checks


def _model_checks(model_path: str, trust_remote_code: bool = False) -> List[Check]:
    from ..utils.security import (  # noqa: PLC0415 - keeps the CLI import light
        RemoteCodeSecurityError,
        require_remote_code_permission,
    )

    try:
        require_remote_code_permission(trust_remote_code)
    except RemoteCodeSecurityError as exc:
        return [
            Check("model", "remote code", FAIL, str(exc)),
        ]
    except Exception as exc:  # noqa: BLE001 - the security gate must not crash the report
        return [Check("model", "remote code", WARN, f"{type(exc).__name__}: {exc}")]

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - a bad path is a FAIL the user can act on
        message = str(exc)
        hint = ""
        if "trust_remote_code" in message.lower() or "auto_map" in message.lower():
            hint = (
                "; this checkpoint ships custom Python. Re-run with --trust-remote-code "
                "and GPTQ_BYPASS_SAFETY_CHECKS=1 after reviewing that code"
            )
        return [Check("model", "config", FAIL, f"{type(exc).__name__}: {message}{hint}")]

    checks = [
        Check("model", "architectures", INFO, ", ".join(getattr(config, "architectures", []) or ["(unknown)"])),
        Check("model", "model_type", INFO, str(getattr(config, "model_type", "(unknown)"))),
    ]
    quantization = getattr(config, "quantization_config", None) or {}
    if quantization:
        method = quantization.get("quant_method") or quantization.get("method") or "unknown"
        bits = quantization.get("bits", "?")
        group = quantization.get("group_size", "?")
        checks.append(Check("model", "quantization", OK, f"{method}, {bits} bit, group {group}"))
    else:
        checks.append(Check("model", "quantization", INFO, "unquantized checkpoint"))

    try:
        from ..utils.kv_cache import estimate_kv_cache

        estimate = estimate_kv_cache(config)
        if estimate is None:
            checks.append(Check("model", "kv cache", INFO, "config does not describe it"))
        else:
            per_token = estimate.kv_bytes_per_token
            checks.append(
                Check(
                    "model",
                    "kv cache",
                    INFO,
                    f"{estimate.layer_type}, {per_token / 1024:.1f} KiB/token, "
                    f"8k {per_token * 8192 / 2**20:.0f} MiB",
                )
            )
    except Exception as exc:  # noqa: BLE001 - the estimate is a convenience, not a gate
        checks.append(Check("model", "kv cache", INFO, f"estimate unavailable: {exc}"))
    return checks


def collect(model_path: Optional[str] = None, *, trust_remote_code: bool = False) -> List[Check]:
    """Run every check this machine can answer."""

    checks = []
    for part in (_runtime_checks, _torch_checks, _gpu_checks, _dependency_checks, _extension_checks, _env_checks):
        try:
            checks.extend(part())
        except Exception as exc:  # noqa: BLE001 - one broken area must not hide the rest
            checks.append(Check("report", part.__name__, WARN, f"{type(exc).__name__}: {exc}"))
    if model_path:
        checks.extend(_model_checks(model_path, trust_remote_code=trust_remote_code))
    return checks


def render(checks: Iterable[Check], logger: Any = None) -> str:
    from ..utils.logger import render_table  # noqa: PLC0415 - table renderer is LogBar

    rows = [[check.area, check.name, check.status, check.detail] for check in checks]
    table = render_table(rows, headers=["area", "check", "status", "detail"], tablefmt="simple", logger=logger)
    failing = sum(1 for check in checks if check.status == FAIL)
    warning = sum(1 for check in checks if check.status == WARN)
    verdict = "ok" if not failing and not warning else f"{failing} fail, {warning} warn"
    return f"{table}\n\nresult: {verdict}"


def to_json(checks: Iterable[Check]) -> Dict[str, Any]:
    materialized = list(checks)
    return {
        "checks": [asdict(check) for check in materialized],
        "failing": sum(1 for check in materialized if check.status == FAIL),
        "warning": sum(1 for check in materialized if check.status == WARN),
    }


def exit_code(checks: Iterable[Check], fail_on: str = "fail") -> int:
    materialized = list(checks)
    if any(check.status == FAIL for check in materialized):
        return 1
    if fail_on == "warn" and any(check.status == WARN for check in materialized):
        return 1
    return 0
