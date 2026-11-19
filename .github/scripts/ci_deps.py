import argparse
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import yaml
from packaging.tags import compatible_tags, cpython_tags, platform_tags, sys_tags
from packaging.utils import parse_wheel_filename


BASE_DIR = Path(__file__).resolve().parent
_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")
_PLAN_GROUPS = ("cpu", "torch", "model")


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


def python_lane(value: Any) -> tuple[int, int, bool] | None:
    """``(major, minor, free_threaded)`` for a ``py`` value, or ``None``.

    ``test.yaml`` spells the lane as an exact managed build (``3.14.10t``).
    Only the major/minor and the free-threaded flag change which wheel tags
    install, so the patch version is ignored here.
    """

    text = str(value or "").strip().lower()
    if not text:
        return None
    free_threaded = text.endswith("t")
    if free_threaded:
        text = text[:-1]
    parts = text.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return int(parts[0]), int(parts[1]), free_threaded


def _is_free_threaded_build() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def interpreter_tags(python_version: str | None = None) -> set[str]:
    """Wheel tags installable on ``python_version`` (default: this interpreter).

    The CI planner runs on the lane interpreter, so the default path is an
    exact :func:`packaging.tags.sys_tags` answer. An explicit version builds
    the same tag set for another CPython so the check can be exercised on a
    developer host without that interpreter.
    """

    requested = python_lane(python_version)
    current = (sys.version_info.major, sys.version_info.minor, _is_free_threaded_build())
    if requested is None or requested == current:
        return {str(tag) for tag in sys_tags()}

    major, minor, free_threaded = requested
    platforms = list(platform_tags())
    abis = [f"cp{major}{minor}t"] if free_threaded else None
    tags = {str(tag) for tag in cpython_tags(python_version=(major, minor), abis=abis, platforms=platforms)}
    tags |= {str(tag) for tag in compatible_tags(python_version=(major, minor), platforms=platforms)}
    if free_threaded:
        # A free-threaded interpreter advertises its own `t` ABI; packaging
        # spells that interpreter/ABI pair `cp314t-cp314t-<platform>`.
        tags |= {f"cp{major}{minor}t-cp{major}{minor}t-{platform}" for platform in platforms}
    return tags


def is_direct_wheel(spec: str) -> bool:
    """Whether ``spec`` names a wheel file rather than a package requirement."""

    path = spec.split("://", 1)[-1].split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(".whl")


def wheel_tags(spec: str) -> set[str]:
    """The tags a direct wheel URL declares, e.g. ``{"cp313-cp313-linux_x86_64"}``."""

    filename = spec.split("://", 1)[-1].split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]
    _, _, _, tags = parse_wheel_filename(filename)
    return {str(tag) for tag in tags}


def incompatible_direct_wheels(
        pkgs: list[str],
        python_version: str | None = None,
) -> list[tuple[str, list[str]]]:
    """Direct wheels in ``pkgs`` that ``python_version`` cannot install.

    Returns ``(spec, declared_tags)`` for each incompatible entry. Package
    requirements are resolved by uv and are not checked here: only the pinned
    wheel URLs can select an ABI the job interpreter does not have.
    """

    available = interpreter_tags(python_version)
    incompatible: list[tuple[str, list[str]]] = []
    for spec in pkgs:
        if not is_direct_wheel(spec):
            continue
        declared = wheel_tags(spec)
        if not declared & available:
            incompatible.append((spec, sorted(declared)))
    return incompatible


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
    run_uv_pip("install", specific_pkgs + common_pkgs, extra_args=["--no-cache", "--torch-backend", "cu132"])
    return 0


def uninstall_deps(raw_name: str) -> int:
    test_path = resolve_test_path(raw_name.removeprefix("tests/").removesuffix(".py"))
    deps = load_yaml("blacklist.yaml")
    specific_pkgs, common_pkgs = collect_pkgs(test_path, deps, dedupe_common=False)
    run_uv_pip("uninstall", specific_pkgs + common_pkgs, ignore_errors=True)
    return 0


def _plan_test_names(plan_path: str | Path) -> list[str]:
    """Test names the planner scheduled, from a ``ci_workflow.py list-tests`` plan."""

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for group in _PLAN_GROUPS:
        for entry in plan.get(f"{group}_matrix") or []:
            script = entry.get("test_script")
            if script:
                names.add(str(script))
    return sorted(names)


def _manifest_test_names(deps: dict[str, Any]) -> list[str]:
    """Every test named by ``deps.yaml``, in ``tests/``-relative form."""

    names: set[str] = set()
    for name in deps.get("tests") or {}:
        names.add(str(name).removesuffix(".py"))
    for key, value in deps.items():
        if not (isinstance(key, str) and key.startswith("tests/")) or not isinstance(value, dict):
            continue
        section = key.removeprefix("tests/")
        for name in value:
            names.add(f"{section}/{str(name).removesuffix('.py')}")
    return sorted(names)


def check_direct_wheels(
        names: list[str],
        *,
        python_version: str | None = None,
        deps: dict[str, Any] | None = None,
) -> list[str]:
    """Direct wheels among ``names`` that the target interpreter cannot install.

    The planner runs this on the lane interpreter before any test job is
    scheduled, so a pinned wheel with a foreign ABI fails the plan instead of
    the install step (#5420).
    """

    deps = load_yaml("deps.yaml") if deps is None else deps
    failures: list[str] = []
    for name in names:
        specific_pkgs, _common_pkgs = collect_pkgs(
            resolve_test_path(name), deps, dedupe_common=False
        )
        for spec, tags in incompatible_direct_wheels(specific_pkgs, python_version):
            failures.append(f"{name}: {spec} declares {', '.join(tags)}")

    lane = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    if failures:
        print(f"incompatible direct wheels for the {lane} lane:")
        for failure in failures:
            print("  -", failure)
    else:
        print(f"ok: every direct wheel for {len(names)} test(s) installs on {lane}")
    return failures


def check_tags(args: argparse.Namespace) -> int:
    selected = [bool(args.test_name), bool(args.plan), bool(args.all)]
    if sum(selected) != 1:
        raise SystemExit("check-tags needs exactly one of --test-name, --plan, or --all")
    if args.test_name:
        names = [args.test_name]
    elif args.plan:
        names = _plan_test_names(args.plan)
    else:
        names = _manifest_test_names(load_yaml("deps.yaml"))
    failures = check_direct_wheels(names, python_version=args.python_version)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("test_name")

    uninstall_parser = subparsers.add_parser("uninstall")
    uninstall_parser.add_argument("test_name")

    tags_parser = subparsers.add_parser("check-tags")
    tags_parser.add_argument("--test-name")
    tags_parser.add_argument("--plan", help="plan JSON written by `ci_workflow.py list-tests`")
    tags_parser.add_argument(
        "--all",
        action="store_true",
        help="check every test in deps.yaml instead of a single test or plan",
    )
    tags_parser.add_argument(
        "--python-version",
        default=None,
        help="target CPython (e.g. 3.14.10t); defaults to the running interpreter",
    )

    args = parser.parse_args()

    if args.command == "install":
        return install_deps(args.test_name)
    if args.command == "uninstall":
        return uninstall_deps(args.test_name)
    if args.command == "check-tags":
        return check_tags(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
