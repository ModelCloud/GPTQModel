# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# odd minor versions are dev (main) branch
# even minor versions are release
# 5.2.0 => release, 5.1.0 => devel
# micro version (5.2.x) denotes patch fix, i.e. 5.2.1 is a patch fix release
__version__ = "7.3.3"
__version__ += "+ultra"

import subprocess
import sys
from pathlib import Path


def _get_git_commit_short(base_version: str = __version__) -> str:
    """Return a version string that appends the current git short hash for local/editable checkouts."""
    git_hash = ""
    try:
        version_file = Path(__file__).resolve()
        for parent in [version_file.parent, *version_file.parents]:
            if not (parent / ".git").exists():
                continue

            top_result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(parent),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if top_result.returncode != 0:
                continue

            repo_root = Path(top_result.stdout.strip())
            try:
                rel = version_file.relative_to(repo_root)
            except ValueError:
                continue

            ls_result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(rel)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if ls_result.returncode != 0:
                continue

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
                break
    except Exception:
        pass

    if not git_hash:
        return base_version

    local_segment = f"local-git-{git_hash}"
    if "+" in base_version:
        return f"{base_version}-{local_segment}"
    return f"{base_version}+{local_segment}"


def _compute_local_version() -> str:
    return _get_git_commit_short(__version__)


def __getattr__(name: str):
    """Lazily compute local-git version so setup.py and package imports don't pay the git cost."""
    if name == "__local_version__":
        local = _compute_local_version()
        sys.modules[__name__].__dict__["__local_version__"] = local
        return local
    if name == "__git_hash__":
        local = __getattr__("__local_version__")
        gh = local.split("-local-git-")[-1] if "-local-git-" in local else ""
        sys.modules[__name__].__dict__["__git_hash__"] = gh
        return gh
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
