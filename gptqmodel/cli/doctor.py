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


def _module_version(module: str) -> Optional[str]:
    try:
        package = __import__(module)
    except Exception:  # noqa: BLE001 - a broken install reports as a version of None
        return None
    return getattr(package, "__version__", "unknown")


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


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
            torch_version_is_vulnerable,
        )

        advisories = matching_torch_advisories()
        if advisories:
            ids = ", ".join(
                advisory.cve_id or advisory.advisory_id for advisory in advisories
            )
            if safety_checks_bypassed():
                # The user opted out, so this is information, not a warning.
                checks.append(Check("torch", "safety advisories", INFO, f"{ids} (checks bypassed)"))
            else:
                status = WARN if torch_version_is_vulnerable() else INFO
                checks.append(Check("torch", "safety advisories", status, ids))
        else:
            checks.append(Check("torch", "safety advisories", OK, "none"))
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

    checks = []
    for index in range(torch.cuda.device_count()):
        try:
            name = torch.cuda.get_device_name(index)
            major, minor = torch.cuda.get_device_capability(index)
            free, total = torch.cuda.mem_get_info(index)
            detail = f"{name} sm_{major}{minor}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB free"
            checks.append(Check("gpu", f"cuda:{index}", OK, detail))
        except Exception as exc:  # noqa: BLE001 - one bad device must not hide the others
            checks.append(Check("gpu", f"cuda:{index}", WARN, f"{type(exc).__name__}: {exc}"))
    return checks


def _dependency_checks() -> List[Check]:
    checks = []
    for module in _REQUIRED:
        version = _module_version(module)
        checks.append(
            Check("deps", module, OK if version else FAIL, "not importable" if not version else str(version))
        )
    for module in _OPTIONAL:
        version = _module_version(module)
        checks.append(
            Check("deps", module, INFO if not version else OK, "not installed" if not version else str(version))
        )

    # transformers 5.24 moved Ernie-VL tokenizer support into `tokenicer`; without
    # it the import of `transformers` succeeds and the first model load fails.
    transformers = _module_version("transformers")
    if transformers and transformers >= "5.24":
        if _has_module("tokenicer.ernie_vl"):
            checks.append(Check("deps", "tokenicer.ernie_vl", OK, "present"))
        else:
            checks.append(
                Check(
                    "deps",
                    "tokenicer.ernie_vl",
                    WARN,
                    "transformers >= 5.24 expects it; `pip install -U tokenicer`",
                )
            )
    return checks


def _extension_checks() -> List[Check]:
    """Report the compiled kernels shipped next to the package."""

    from pathlib import Path

    package_dir = Path(__file__).resolve().parent.parent
    artifacts = sorted(path.name for path in package_dir.glob("*.so"))
    if artifacts:
        listed = ", ".join(artifacts[:4]) + (" ..." if len(artifacts) > 4 else "")
        return [Check("extension", "native kernels", OK, f"{len(artifacts)} built ({listed})")]
    return [
        Check(
            "extension",
            "native kernels",
            INFO,
            "none built next to the package; torch/triton kernels still work",
        )
    ]


def _env_checks() -> List[Check]:
    checks = []
    for name in _ENV_FLAGS:
        value = os.environ.get(name)
        checks.append(Check("env", name, INFO, value if value is not None else "(unset)"))
    return checks


def _model_checks(model_path: str) -> List[Check]:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - a bad path is a FAIL the user can act on
        return [Check("model", "config", FAIL, f"{type(exc).__name__}: {exc}")]

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


def collect(model_path: Optional[str] = None) -> List[Check]:
    """Run every check this machine can answer."""

    checks = []
    for part in (_runtime_checks, _torch_checks, _gpu_checks, _dependency_checks, _extension_checks, _env_checks):
        try:
            checks.extend(part())
        except Exception as exc:  # noqa: BLE001 - one broken area must not hide the rest
            checks.append(Check("report", part.__name__, WARN, f"{type(exc).__name__}: {exc}"))
    if model_path:
        checks.extend(_model_checks(model_path))
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
