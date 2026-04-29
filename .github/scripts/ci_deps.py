import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent
_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")


def resolve_test_path(raw_name: str) -> Path:
    return Path("tests") / f"{raw_name}.py"


def load_yaml(filename: str) -> dict[str, Any]:
    with (BASE_DIR / filename).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def normalize_pkg_spec(spec: str) -> str:
    spec = (spec or "").strip()
    if not spec:
        return spec

    if spec.startswith("git+"):
        return spec

    if spec.startswith("https://github.com/"):
        spec = spec.rstrip("/")
        if not spec.endswith(".git"):
            spec += ".git"
        return "git+" + spec

    return spec


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


def collect_pkgs(test_path: Path, deps: dict[str, Any], *, dedupe_common: bool) -> tuple[list[str], list[str]]:
    specific_pkgs: set[str] = set()
    common_pkgs: set[str] = set(deps.get("common") or [])

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

    if dedupe_common:
        specific_keys = {pkg_key(pkg) for pkg in specific_pkgs}
        common_pkgs = {pkg for pkg in common_pkgs if pkg_key(pkg) not in specific_keys}

    return sorted(specific_pkgs), sorted(common_pkgs)


def dedupe_pkg_specs(pkgs: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for pkg in pkgs:
        normalized = normalize_pkg_spec(pkg)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def run_uv_pip(
        action: str,
        pkgs: list[str],
        *,
        extra_args: list[str] | None = None,
        ignore_errors: bool = False,
) -> None:
    if not pkgs:
        return

    normalized = dedupe_pkg_specs(pkgs)
    print(f"--- {action.title()} deps with uv:")
    for pkg in normalized:
        print("  -", pkg)

    cmd = ["uv", "pip", action]
    if action == "uninstall":
        cmd.append("-y")
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(normalized)
    print("+", " ".join(cmd))

    if ignore_errors:
        completed = subprocess.run(cmd, shell=False, check=False)
        if completed.returncode != 0:
            print(f"{action.title()} failed with exit code {completed.returncode}")
        return

    subprocess.check_call(cmd, shell=False)


def install_deps(raw_name: str) -> int:
    test_path = resolve_test_path(raw_name.removeprefix("tests/").removesuffix(".py"))
    deps = load_yaml("deps.yaml")
    specific_pkgs, common_pkgs = collect_pkgs(test_path, deps, dedupe_common=True)
    run_uv_pip("install", specific_pkgs + common_pkgs, extra_args=["--no-cache"])
    return 0


def uninstall_deps(raw_name: str) -> int:
    test_path = resolve_test_path(raw_name.removeprefix("tests/").removesuffix(".py"))
    deps = load_yaml("blacklist.yaml")
    specific_pkgs, common_pkgs = collect_pkgs(test_path, deps, dedupe_common=False)
    run_uv_pip("uninstall", specific_pkgs + common_pkgs, ignore_errors=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("test_name")

    uninstall_parser = subparsers.add_parser("uninstall")
    uninstall_parser.add_argument("test_name")

    args = parser.parse_args()

    if args.command == "install":
        return install_deps(args.test_name)
    if args.command == "uninstall":
        return uninstall_deps(args.test_name)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
