import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ci_common import append_github_env


def split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def strip_py_suffix(name: str) -> str:
    return name.removesuffix(".py")


@dataclass(frozen=True)
class TestRuntime:
    test_name: str
    test_path: str
    skip_gpu_allocation: bool
    xpu_mode: bool


@dataclass(frozen=True)
class TestMatrixEntry:
    test_script: str
    test_group: str
    alloc_gpu_count: str
    require_single_gpu: str
    include_model_test_mode: str

    def as_dict(self) -> dict[str, str]:
        return {
            "test_script": self.test_script,
            "test_group": self.test_group,
            "alloc_gpu_count": self.alloc_gpu_count,
            "require_single_gpu": self.require_single_gpu,
            "include_model_test_mode": self.include_model_test_mode,
        }


@lru_cache(maxsize=None)
def _load_yaml_cached(yaml_path: str) -> dict[str, Any]:
    import yaml

    with Path(yaml_path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"yaml file must contain a mapping: {yaml_path}")
    return data


def load_yaml(yaml_file: str | Path) -> dict[str, Any]:
    return _load_yaml_cached(str(Path(yaml_file).resolve()))


def parse_test_config_data(
        data: dict[str, Any],
        group: str,
        test_name: str | None = None,
) -> dict[str, Any]:
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


def parse_test_config(
        yaml_file: str | Path,
        group: str,
        test_name: str | None = None,
) -> dict[str, Any]:
    return parse_test_config_data(load_yaml(yaml_file), group, test_name)


def normalize_test_name(name: str) -> str:
    return strip_py_suffix(name.removeprefix("tests/"))


def test_path_from_name(test_name: str, tests_root: str | Path = "tests") -> Path:
    normalized = normalize_test_name(test_name)
    return Path(tests_root) / f"{normalized}.py"


def parse_test_config_for_path(
        yaml_file: str | Path,
        group: str,
        test_name: str,
) -> dict[str, Any]:
    data = load_yaml(yaml_file)
    return parse_test_config_data_for_path(data, group, test_name)


def parse_test_config_data_for_path(
        data: dict[str, Any],
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
        try:
            scoped = parse_test_config_data(data, candidate_group, scoped_test_name)
        except KeyError:
            continue
        result.update(scoped)
    return result


@lru_cache(maxsize=None)
def _read_text_cached(file_path: str) -> str | None:
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return None


def has_no_gpu_marker(file_path: Path) -> bool:
    contents = _read_text_cached(str(file_path.resolve()))
    if contents is None:
        return False
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "# GPU=-1":
            return True
        if stripped.startswith("import ") or stripped.startswith("from "):
            return False
    return False


def sort_key(path: str, file_path: Path) -> tuple[bool, bool, bool, str]:
    return ("moe" in path, not has_no_gpu_marker(file_path), "/" in path, path)


def is_model_compat_test(rel_path: str, file_path: Path) -> bool:
    if not rel_path.startswith("models/"):
        return False
    contents = _read_text_cached(str(file_path.resolve()))
    if contents is None:
        return False
    markers = ("quantize_and_evaluate(", "self.evaluate_model(", "check_results(")
    return any(marker in contents for marker in markers)


def matches_test_regex(test_regex: Any, rel_path: str) -> bool:
    rel_posix = PurePosixPath(rel_path)
    candidates = (
        rel_path,
        f"{rel_path}.py",
        str(PurePosixPath("tests") / rel_posix),
        str(PurePosixPath("tests") / rel_posix.with_suffix(".py")),
        rel_posix.name,
        rel_posix.with_suffix(".py").name,
    )
    return any(test_regex.match(candidate) for candidate in candidates)


def should_skip_test(config_data: dict[str, Any], rel_path: str) -> bool:
    try:
        config = parse_test_config_data_for_path(config_data, "tests", rel_path)
    except KeyError:
        return False
    return bool(config.get("skip", False))


def list_tests(
        ignored_test_files: str | list[str],
        test_names: str,
        test_regex: str,
        tests_root: str | Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    tests_root = Path(tests_root)
    input_tests = [strip_py_suffix(name) for name in split_csv(test_names)]
    ignored_raw = ignored_test_files if isinstance(ignored_test_files, list) else split_csv(ignored_test_files)
    ignored_set = {strip_py_suffix(name) for name in ignored_raw}
    config_data = load_yaml(Path(__file__).with_name("test.yaml"))
    compiled_test_regex = re.compile(test_regex)

    all_tests = {
        rel: path
        for path in tests_root.rglob("test_*.py")
        for rel in [str(path.relative_to(tests_root).with_suffix(""))]
        if rel not in ignored_set and path.stem not in ignored_set
        if not should_skip_test(config_data, rel)
    }

    model_tests = {
        rel
        for rel, path in all_tests.items()
        if (not input_tests or rel in input_tests)
           and "mlx" not in rel
           and "ipex" not in rel
           and "xpu" not in rel
           and matches_test_regex(compiled_test_regex, rel)
           and is_model_compat_test(rel, path)
    }

    cpu_tests = {
        rel
        for rel, path in all_tests.items()
        if (not input_tests or rel in input_tests)
           and rel not in model_tests
           and "mlx" not in rel
           and "ipex" not in rel
           and "xpu" not in rel
           and matches_test_regex(compiled_test_regex, rel)
           and has_no_gpu_marker(path)
    }

    torch_tests = {
        rel
        for rel in all_tests
        if (not input_tests or rel in input_tests)
           and rel not in model_tests
           and rel not in cpu_tests
           and "mlx" not in rel
           and "ipex" not in rel
           and "xpu" not in rel
           and matches_test_regex(compiled_test_regex, rel)
    }

    mlx_tests = {
        rel
        for rel in all_tests
        if ("mlx" in rel or "apple" in rel)
           and ((rel in input_tests) if input_tests else True)
           and matches_test_regex(compiled_test_regex, rel)
    }

    return (
        sorted(cpu_tests, key=lambda rel: sort_key(rel, all_tests[rel])),
        sorted(torch_tests, key=lambda rel: sort_key(rel, all_tests[rel])),
        sorted(model_tests, key=lambda rel: sort_key(rel, all_tests[rel])),
        sorted(mlx_tests, key=lambda rel: sort_key(rel, all_tests[rel])),
    )


def build_group_matrix(group: str, tests: list[str]) -> list[dict[str, str]]:
    if group == "cpu":
        return [
            TestMatrixEntry(
                test_script=test_script,
                test_group="cpu",
                alloc_gpu_count="0",
                require_single_gpu="false",
                include_model_test_mode="false",
            ).as_dict()
            for test_script in tests
        ]
    if group == "torch":
        return [
            TestMatrixEntry(
                test_script=test_script,
                test_group="torch",
                alloc_gpu_count="resolved",
                require_single_gpu="false",
                include_model_test_mode="false",
            ).as_dict()
            for test_script in tests
        ]
    if group == "model":
        return [
            TestMatrixEntry(
                test_script=test_script,
                test_group="model",
                alloc_gpu_count="1",
                require_single_gpu="true",
                include_model_test_mode="true",
            ).as_dict()
            for test_script in tests
        ]
    raise ValueError(f"unsupported test group: {group}")


def build_test_matrices(
        *,
        cpu_tests: list[str],
        torch_tests: list[str],
        model_tests: list[str],
) -> dict[str, list[dict[str, str]]]:
    return {
        "cpu_matrix": build_group_matrix("cpu", cpu_tests),
        "torch_matrix": build_group_matrix("torch", torch_tests),
        "model_matrix": build_group_matrix("model", model_tests),
    }


def build_test_plan(
        *,
        ignored_test_files: str | list[str],
        test_names: str,
        test_regex: str,
        tests_root: str | Path,
) -> dict[str, list[dict[str, str]] | list[str]]:
    cpu_tests, torch_tests, model_tests, mlx_tests = list_tests(
        ignored_test_files=ignored_test_files,
        test_names=test_names,
        test_regex=test_regex,
        tests_root=tests_root,
    )
    matrices = build_test_matrices(
        cpu_tests=cpu_tests,
        torch_tests=torch_tests,
        model_tests=model_tests,
    )
    return {
        "cpu_files": cpu_tests,
        "torch_files": torch_tests,
        "model_files": model_tests,
        "mlx_files": mlx_tests,
        **matrices,
    }


def resolve_test_runtime(test_name: str, tests_root: str | Path = "tests") -> TestRuntime:
    normalized = normalize_test_name(test_name)
    test_path = test_path_from_name(normalized, tests_root=tests_root)
    xpu_mode = "xpu" in normalized
    skip_gpu_allocation = xpu_mode or has_no_gpu_marker(test_path)
    return TestRuntime(
        test_name=normalized,
        test_path=str(test_path),
        skip_gpu_allocation=skip_gpu_allocation,
        xpu_mode=xpu_mode,
    )


def list_matching_versions(package: str, version_spec: str) -> list[str]:
    import requests
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

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
    test_plan = build_test_plan(
        ignored_test_files=args.ignored_test_files,
        test_names=args.test_names,
        test_regex=args.test_regex,
        tests_root=args.tests_root,
    )
    print(json.dumps(test_plan))
    return 0


def cmd_resolve_env(args: argparse.Namespace) -> int:
    config = parse_test_config_for_path(args.yaml_file, args.group, args.test_name)
    py = str(config["py"])
    gpu = str(config["gpu"])
    sm = str(config.get("sm", 0))
    test_name_for_env = args.test_name.replace("/", "_")
    env_name = (
        f"{args.env_prefix}_cu{args.cuda_version}_torch{args.torch_version}"
        f"_py{py}_{test_name_for_env}"
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


def cmd_resolve_test_runtime(args: argparse.Namespace) -> int:
    runtime = resolve_test_runtime(args.test_name, tests_root=args.tests_root)
    skip_gpu = str(runtime.skip_gpu_allocation).lower()
    xpu_mode = str(runtime.xpu_mode).lower()

    append_github_env("TEST_FILE", runtime.test_path)
    append_github_env("SKIP_GPU_ALLOCATION", skip_gpu)
    append_github_env("XPU_MODE", xpu_mode)

    if args.shell:
        print(f"TEST_FILE={shlex.quote(runtime.test_path)}")
        print(f"SKIP_GPU_ALLOCATION={shlex.quote(skip_gpu)}")
        print(f"XPU_MODE={shlex.quote(xpu_mode)}")
    else:
        print(
            f"using test_file={runtime.test_path} skip_gpu_allocation={skip_gpu} "
            f"xpu_mode={xpu_mode} for test {runtime.test_name}"
        )
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

    runtime_parser = subparsers.add_parser("resolve-test-runtime")
    runtime_parser.add_argument("--test-name", required=True)
    runtime_parser.add_argument("--tests-root", default="tests")
    runtime_parser.add_argument("--shell", action="store_true")

    versions_parser = subparsers.add_parser("loop-versions")
    versions_parser.add_argument("package")
    versions_parser.add_argument("version")

    args = parser.parse_args()
    if args.command == "list-tests":
        return cmd_list_tests(args)
    if args.command == "resolve-env":
        return cmd_resolve_env(args)
    if args.command == "resolve-test-runtime":
        return cmd_resolve_test_runtime(args)
    if args.command == "loop-versions":
        return cmd_loop_versions(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
