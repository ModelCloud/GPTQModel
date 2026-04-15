import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

base_dir = os.path.dirname(os.path.abspath(__file__))
_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")


def resolve_test_path(raw_name: str) -> Path:
    return Path("tests") / f"{raw_name}.py"


def normalize_pkg_spec(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    if s.startswith("git+"):
        return s

    if s.startswith("https://github.com/"):
        s = s.rstrip("/")
        if not s.endswith(".git"):
            s += ".git"
        return "git+" + s

    return s


def pkg_key(spec: str) -> str:
    spec = normalize_pkg_spec(spec)
    if not spec:
        return spec

    if spec.startswith("git+"):
        repo = spec.rsplit("/", 1)[-1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return repo.split("@", 1)[0].lower().replace("_", "-")

    if "://" in spec:
        return spec

    spec = spec.split(";", 1)[0].strip()
    if " @" in spec:
        spec = spec.split(" @", 1)[0].strip()

    match = _PKG_NAME_RE.match(spec)
    if not match:
        return spec.lower()

    return match.group(0).lower().replace("_", "-")


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

    specific_pkg_keys = {pkg_key(pkg) for pkg in specific_pkgs}
    common_pkgs = {pkg for pkg in common_pkgs if pkg_key(pkg) not in specific_pkg_keys}

    return specific_pkgs, common_pkgs


def pip_install(pkgs):
    if not pkgs:
        return

    print("--- Installing deps:")
    for p in pkgs:
        print("  -", p)

    cmd = [
        sys.executable,
        "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
    ]
    pkgs = [normalize_pkg_spec(p) for p in pkgs]
    cmd.extend(pkgs)

    subprocess.check_call(cmd, shell=False)


def uv_install(pkgs):
    if not pkgs:
        return

    pkgs = [normalize_pkg_spec(p) for p in pkgs]

    print("--- Installing deps with uv:")
    for p in pkgs:
        print("  -", p)

    for p in pkgs:
        cmd = ["uv", "pip", "install", "--no-cache", p]
        print("installing: ", cmd)
        try:
            subprocess.check_call(cmd, shell=False)
        except Exception as e:
            print(f"Install failed: {e}")


if __name__ == "__main__":
    raw_name = sys.argv[1].removeprefix("tests/").removesuffix(".py")
    test_path = resolve_test_path(raw_name)

    with open(os.path.join(base_dir, "deps.yaml")) as f:
        deps = yaml.safe_load(f)

    specific_pkgs, common_pkgs = collect_pkgs(test_path, deps)

    uv_install(specific_pkgs)

    uv_install(common_pkgs)
