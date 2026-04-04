# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup


def _package_version() -> str:
    version_vars: dict[str, str] = {}
    exec(Path("gptqmodel/version.py").read_text(encoding="utf-8"), {}, version_vars)
    return version_vars["__version__"]


packages = find_packages(exclude=("tests", "tests.*"))
for package_name in find_namespace_packages(include=("gptqmodel_ext.*",)):
    if package_name not in packages:
        packages.append(package_name)


setup(
    version=_package_version(),
    packages=packages,
    include_package_data=True,
)
