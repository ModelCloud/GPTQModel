#!/usr/bin/env python3
"""Execute a repository script with that worktree's package imported first.

The development environment has an editable ``gptqmodel`` installation.  A
normal ``PYTHONPATH`` is not sufficient to override its meta-path finder, so
historical experiment worktrees can accidentally execute the current checkout.
This tiny bootstrap explicitly loads the requested worktree package before
running the target script.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--script", required=True, help="Path relative to --worktree")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    script = (worktree / args.script).resolve()
    package_root = worktree / "gptqmodel"
    if not package_root.is_dir() or not script.is_file():
        raise FileNotFoundError(f"invalid worktree script/package: {worktree} {args.script}")

    # Remove any package loaded by the editable install and eagerly bind the
    # worktree package so all subsequent relative imports stay in that tree.
    for name in tuple(sys.modules):
        if name == "gptqmodel" or name.startswith("gptqmodel."):
            del sys.modules[name]
    sys.path.insert(0, str(worktree))
    spec = importlib.util.spec_from_file_location(
        "gptqmodel",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gptqmodel from {package_root}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["gptqmodel"] = package
    spec.loader.exec_module(package)

    os.chdir(worktree)
    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args.pop(0)
    sys.argv = [str(script), *script_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
