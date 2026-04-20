import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import requests
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def strip_py_suffix(name: str) -> str:
    return name.removesuffix(".py")


def append_github_env(name: str, value: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def parse_test_config(
        yaml_file: str | Path,
        group: str,
        test_name: str | None = None,
) -> dict[str, Any]:
    yaml_path = Path(yaml_file)
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    result: dict[str, Any] = {}
    common_data = data.get("common") or {}
    if not isinstance(common_data, dict):
        raise ValueError("group must be a mapping: common")

    for key, value in common_data.items():
        if not isinstance(value, dict):
            result[key] = value

    if group not in data:
        raise KeyError(f"group not found: {group}")

    group_data = data[group] or {}
    if not isinstance(group_data, dict):
        raise ValueError(f"group must be a mapping: {group}")

    for key, value in group_data.items():
        if not isinstance(value, dict):
            result[key] = value

    if test_name is not None:
        test_config = group_data.get(test_name)
        if test_config is not None:
            if not isinstance(test_config, dict):
                raise ValueError(f"test config must be a mapping: {test_name}")
            result.update(test_config)

    return result


def parse_test_config_for_path(
        yaml_file: str | Path,
        group: str,
        test_name: str,
) -> dict[str, Any]:
    test_path = PurePosixPath(test_name)
    candidate_groups = [group]
    if len(test_path.parts) > 1:
        prefix = PurePosixPath(group)
        for depth in range(1, len(test_path.parts)):
            candidate_groups.append(str(prefix.joinpath(*test_path.parts[:depth])))

    result: dict[str, Any] = {}
    leaf_name = test_path.name
    for index, candidate_group in enumerate(candidate_groups):
        scoped_test_name = None
        if index == len(candidate_groups) - 1:
            scoped_test_name = leaf_name
        scoped = parse_test_config(yaml_file, candidate_group, scoped_test_name)
        result.update(scoped)
    return result


def has_no_gpu_marker(file_path: Path) -> bool:
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == "# GPU=-1":
                    return True
                if stripped.startswith("import ") or stripped.startswith("from "):
                    return False
    except OSError:
        return False
    return False


def sort_key(path: str, file_path: Path) -> tuple[bool, bool, bool, str]:
    return ("moe" in path, not has_no_gpu_marker(file_path), "/" in path, path)


def is_model_compat_test(rel_path: str, file_path: Path) -> bool:
    if not rel_path.startswith("models/"):
        return False
    try:
        contents = file_path.read_text(encoding="utf-8")
    except OSError:
        return False
    markers = ("quantize_and_evaluate(", "self.evaluate_model(", "check_results(")
    return any(marker in contents for marker in markers)


def matches_test_regex(test_regex: str, rel_path: str) -> bool:
    if re.match(test_regex, rel_path):
        return True
    return re.match(test_regex, PurePosixPath(rel_path).name) is not None


def should_skip_test(yaml_file: str | Path, rel_path: str) -> bool:
    try:
        config = parse_test_config_for_path(yaml_file, "tests", rel_path)
    except KeyError:
        return False
    return bool(config.get("skip", False))


def list_tests(ignored_test_files: str | list[str], test_names: str, test_regex: str, tests_root: str | Path) -> tuple[list[str], list[str], list[str]]:
    tests_root = Path(tests_root)
    input_tests = [strip_py_suffix(name) for name in split_csv(test_names)]
    ignored_raw = ignored_test_files if isinstance(ignored_test_files, list) else split_csv(ignored_test_files)
    ignored_set = {strip_py_suffix(name) for name in ignored_raw}
    yaml_file = Path(__file__).with_name("test.yaml")

    all_tests = {
        rel: path
        for path in tests_root.rglob("test_*.py")
        for rel in [str(path.relative_to(tests_root).with_suffix(""))]
        if rel not in ignored_set and path.stem not in ignored_set
        if not should_skip_test(yaml_file, rel)
    }

    model_tests = {
        rel
        for rel, path in all_tests.items()
        if (not input_tests or rel in input_tests)
           and "mlx" not in rel
           and "ipex" not in rel
           and "xpu" not in rel
           and matches_test_regex(test_regex, rel)
           and is_model_compat_test(rel, path)
    }

    torch_tests = {
        rel
        for rel in all_tests
        if (not input_tests or rel in input_tests)
           and rel not in model_tests
           and "mlx" not in rel
           and "ipex" not in rel
           and "xpu" not in rel
           and matches_test_regex(test_regex, rel)
    }

    mlx_tests = {
        rel
        for rel in all_tests
        if ("mlx" in rel or "apple" in rel)
           and ((rel in input_tests) if input_tests else True)
           and matches_test_regex(test_regex, rel)
    }

    return (
        sorted(torch_tests, key=lambda rel: sort_key(rel, all_tests[rel])),
        sorted(model_tests, key=lambda rel: sort_key(rel, all_tests[rel])),
        sorted(mlx_tests, key=lambda rel: sort_key(rel, all_tests[rel])),
    )


def list_matching_versions(package: str, version_spec: str) -> list[str]:
    specifier = SpecifierSet(version_spec)
    response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=30)
    response.raise_for_status()
    data = response.json()
    matched = sorted(
        (Version(version) for version in data["releases"].keys() if Version(version) in specifier),
        reverse=True,
    )
    return [str(version) for version in matched]


def cmd_list_tests(args: argparse.Namespace) -> int:
    torch_files, model_files, mlx_files = list_tests(
        ignored_test_files=args.ignored_test_files,
        test_names=args.test_names,
        test_regex=args.test_regex,
        tests_root=args.tests_root,
    )
    print(f"{json.dumps(torch_files)}|{json.dumps(model_files)}|{json.dumps(mlx_files)}")
    return 0


def cmd_resolve_env(args: argparse.Namespace) -> int:
    config = parse_test_config_for_path(args.yaml_file, args.group, args.test_name)
    py = str(config["py"])
    gpu = str(config["gpu"])
    sm = str(config.get("sm", 0))
    env_name = (
        f"{args.env_prefix}_cu{args.cuda_version}_torch{args.torch_version}"
        f"_py{py}_{args.test_name}"
    )

    append_github_env("PYTHON_VERSION", py)
    append_github_env("GPU_COUNT", gpu)
    append_github_env("SM", sm)
    append_github_env("ENV_NAME", env_name)

    if args.shell:
        print(f"PYTHON_VERSION={shlex.quote(py)}")
        print(f"GPU_COUNT={shlex.quote(gpu)}")
        print(f"SM={shlex.quote(sm)}")
        print(f"ENV_NAME={shlex.quote(env_name)}")
    else:
        print(f"using py={py} gpu={gpu} sm={sm} env={env_name} for test {args.test_name}")
    return 0


def cmd_loop_versions(args: argparse.Namespace) -> int:
    print(json.dumps(list_matching_versions(args.package, args.version)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tests")
    list_parser.add_argument("--ignored-test-files", default=os.getenv("IGNORED_TEST_FILES", ""))
    list_parser.add_argument("--test-names", default=os.getenv("TEST_NAMES", ""))
    list_parser.add_argument("--test-regex", default=os.getenv("TEST_REGEX", ".*"))
    list_parser.add_argument("--tests-root", default="tests")

    resolve_parser = subparsers.add_parser("resolve-env")
    resolve_parser.add_argument("--yaml-file", default=Path(__file__).with_name("test.yaml"))
    resolve_parser.add_argument("--group", required=True)
    resolve_parser.add_argument("--test-name", required=True)
    resolve_parser.add_argument("--cuda-version", required=True)
    resolve_parser.add_argument("--torch-version", required=True)
    resolve_parser.add_argument("--env-prefix", default="gptqmodel_test")
    resolve_parser.add_argument("--shell", action="store_true")

    versions_parser = subparsers.add_parser("loop-versions")
    versions_parser.add_argument("package")
    versions_parser.add_argument("version")

    args = parser.parse_args()
    if args.command == "list-tests":
        return cmd_list_tests(args)
    if args.command == "resolve-env":
        return cmd_resolve_env(args)
    if args.command == "loop-versions":
        return cmd_loop_versions(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
