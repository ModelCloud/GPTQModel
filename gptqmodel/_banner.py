# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Iterable

TRITON_PACKAGE_CANDIDATES = (
    "triton",
    "triton-windows",
    "pytorch_triton_xpu",
    "pytorch-triton-xpu",
)


def resolve_installed_package_version(package_names: Iterable[str]) -> str | None:
    for package_name in package_names:
        try:
            resolved_version = package_version(package_name)
        except PackageNotFoundError:
            continue

        if resolved_version:
            return resolved_version

    return None


def build_startup_banner(
    ascii_logo: str,
    *,
    gptqmodel_version: str,
    transformers_version: str,
    torch_version: str,
    triton_version: str | None = None,
) -> str:
    version_rows = [
        ("GPT-QModel", gptqmodel_version),
        ("Transformers", transformers_version),
        ("Torch", torch_version),
    ]

    if triton_version:
        version_rows.append(("Triton", triton_version))

    label_width = max(len(label) for label, _ in version_rows)
    formatted_rows = [
        f"{label:<{label_width}} : {value}" for label, value in version_rows
    ]

    return "\n".join([ascii_logo.rstrip("\n"), *formatted_rows])


def _get_git_commit():
    import subprocess
    try:
        hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
        return f"+{hash}"
    except Exception:
        return ""

def get_startup_banner(
    ascii_logo: str,
    *,
    gptqmodel_version: str,
    transformers_version: str,
    torch_version: str,
) -> str:
    return build_startup_banner(
        ascii_logo,
        gptqmodel_version=f"{gptqmodel_version}{_get_git_commit()}",
        transformers_version=transformers_version,
        torch_version=torch_version,
        triton_version=resolve_installed_package_version(TRITON_PACKAGE_CANDIDATES),
    )
