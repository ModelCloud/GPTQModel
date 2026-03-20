import os
import subprocess
import sys
from pathlib import Path

import yaml

base_dir = os.path.dirname(os.path.abspath(__file__))


def resolve_test_path(raw_name: str) -> Path:
    return Path("tests") / f"{raw_name}.py"


def collect_pkgs(test_path: Path, deps: dict):
    specific_pkgs = set()

    common_pkgs = set(deps.get("common") or [])

    specific_pkgs.update(deps.get("tests", {}).get(test_path.name) or [])

    test_path_str = test_path.as_posix()
    for key, value in deps.items():
        if not (isinstance(key, str) and key.startswith("tests/")):
            continue
        if not test_path_str.startswith(key + "/"):
            continue

        if isinstance(value, list):
            specific_pkgs.update(value)

        elif isinstance(value, dict):
            specific_pkgs.update(value.get(test_path.name) or [])

        else:
            pass

    return specific_pkgs, common_pkgs


def uv_uninstall(pkgs):
    if not pkgs:
        return

    print("--- Uninstalling deps with uv:")
    for p in pkgs:
        print("  -", p)

    for p in pkgs:
        cmd = ["uv", "pip", "uninstall", p]
        try:
            subprocess.check_call(cmd, shell=False)
        except Exception as e:
            print(f"--- Unnstall failed: {e}")


if __name__ == "__main__":
    raw_name = sys.argv[1].removeprefix("tests/").removesuffix(".py")
    test_path = resolve_test_path(raw_name)

    with open(os.path.join(base_dir, "blacklist.yaml")) as f:
        deps = yaml.safe_load(f)

    specific_pkgs, common_pkgs = collect_pkgs(test_path, deps)

    uv_uninstall(sorted(specific_pkgs))

    uv_uninstall(sorted(common_pkgs))
