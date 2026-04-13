import os
import subprocess
import sys

import yaml
from deps_utils import collect_pkgs, resolve_test_path

base_dir = os.path.dirname(os.path.abspath(__file__))


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
