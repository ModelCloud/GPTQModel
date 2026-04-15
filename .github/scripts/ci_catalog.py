import re
from hashlib import sha1
from pathlib import Path
from typing import Any

from ci_common import load_yaml

DEFAULT_DEPS_FILE = Path(__file__).with_name("deps.yaml")
DEFAULT_TEST_CONFIG_FILE = Path(__file__).with_name("test.yaml")

SHARED_COMMON_ENVS = {
    "gptqmodel_unit_tests_common",
    "gptqmodel_unit_tests_models_common",
}

_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")


def resolve_test_path(raw_name: str) -> Path:
    normalized = raw_name.removeprefix("tests/").removesuffix(".py")
    return Path("tests") / f"{normalized}.py"


def lookup_test_entry(mapping: dict[str, Any] | None, test_path: Path) -> list[Any]:
    if not isinstance(mapping, dict):
        return []

    value = mapping.get(test_path.name)
    if value is None:
        value = mapping.get(test_path.stem)

    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def collect_packages(test_path: Path, deps: dict[str, Any]) -> tuple[set[Any], set[Any]]:
    specific_packages: set[Any] = set()
    common_packages = set(deps.get("common") or [])

    specific_packages.update(lookup_test_entry(deps.get("tests"), test_path))

    test_path_str = test_path.as_posix()
    for key, value in deps.items():
        if not (isinstance(key, str) and key.startswith("tests/")):
            continue
        if not test_path_str.startswith(f"{key}/"):
            continue

        if isinstance(value, list):
            specific_packages.update(value)
        elif isinstance(value, dict):
            specific_packages.update(lookup_test_entry(value, test_path))

    return specific_packages, common_packages


def has_specific_dependencies(test_name: str, deps: dict[str, Any]) -> bool:
    test_path = resolve_test_path(test_name)
    specific_packages, _ = collect_packages(test_path, deps)
    return bool(specific_packages)


def flatten_test_name(test_name: str) -> str:
    return test_name.removeprefix("tests/").removesuffix(".py").replace("/", "_")


def is_model_test(test_name: str) -> bool:
    normalized = test_name.removeprefix("tests/").removesuffix(".py")
    return normalized.startswith("models/")


def normalize_package_spec(spec: str) -> str:
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


def package_env_key(spec: str) -> str:
    spec = normalize_package_spec(spec)
    if not spec:
        return "deps"

    if spec.startswith("git+"):
        repo = spec.rsplit("/", 1)[-1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return sanitize_env_token(repo.split("@", 1)[0])

    if "://" in spec:
        filename = spec.rstrip("/").rsplit("/", 1)[-1]
        filename = filename.removesuffix(".whl").removesuffix(".tar.gz").removesuffix(".zip")
        match = re.match(r"([A-Za-z0-9_.-]+?)-\d", filename)
        if match:
            return sanitize_env_token(match.group(1))
        return sanitize_env_token(filename)

    base = spec.split(";", 1)[0].strip()
    if " @" in base:
        base = base.split(" @", 1)[0].strip()

    match = _PKG_NAME_RE.match(base)
    if not match:
        return "deps"

    return sanitize_env_token(match.group(0))


def package_install_key(spec: str) -> str:
    spec = normalize_package_spec(spec)
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


def sanitize_env_token(value: str) -> str:
    token = value.lower().replace(".", "_")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "deps"


def build_specific_deps_env_suffix(packages: set[Any]) -> str:
    tokens = sorted(
        {package_env_key(str(package)) for package in packages if str(package).strip()}
    )
    if not tokens:
        return "deps"
    if len(tokens) == 1:
        return tokens[0]

    joined = "_".join(tokens)
    if len(joined) <= 48:
        return joined

    digest = sha1("|".join(tokens).encode("utf-8")).hexdigest()[:10]
    return f"deps_{digest}"


def build_env_name(test_name: str, deps: dict[str, Any]) -> str:
    test_path = resolve_test_path(test_name)
    specific_packages, _ = collect_packages(test_path, deps)
    prefix = "gptqmodel_unit_tests_models" if is_model_test(test_name) else "gptqmodel_unit_tests"

    if specific_packages:
        return f"{prefix}_{build_specific_deps_env_suffix(specific_packages)}_common"

    if is_model_test(test_name):
        return "gptqmodel_unit_tests_models_common"

    return "gptqmodel_unit_tests_common"


def parse_test_config(
    yaml_file: str | Path,
    group: str,
    test_name: str | None = None,
) -> dict[str, Any]:
    data = load_yaml(yaml_file)
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
        if test_config is None:
            return result
        if not isinstance(test_config, dict):
            raise ValueError(f"test config must be a mapping: {test_name}")
        result.update(test_config)

    return result


def parse_test_config_flags(
    yaml_file: str | Path,
    group: str,
    test_name: str | None = None,
) -> dict[str, int]:
    config = parse_test_config(yaml_file, group, test_name)
    return {key: 1 for key in config}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _strip_py_suffix(name: str) -> str:
    return name.removesuffix(".py")


def _sort_key(path: str) -> tuple[bool, bool, str]:
    return ("moe" in path, "/" in path, path)


def _is_model_compat_test(rel_path: str, file_path: Path) -> bool:
    if not rel_path.startswith("models/"):
        return False

    try:
        contents = file_path.read_text(encoding="utf-8")
    except OSError:
        return False

    compat_markers = (
        "quantize_and_evaluate(",
        "self.evaluate_model(",
        "check_results(",
    )
    return any(marker in contents for marker in compat_markers)


def list_test_files(
    *,
    ignored_test_files: str | list[str],
    test_names: str = "",
    test_regex: str = ".*",
    tests_root: str | Path = "tests",
) -> tuple[list[str], list[str], list[str]]:
    tests_root = Path(tests_root)
    requested_tests = [_strip_py_suffix(name) for name in _split_csv(test_names)]
    ignored = ignored_test_files if isinstance(ignored_test_files, list) else _split_csv(ignored_test_files)
    ignored_set = {_strip_py_suffix(item) for item in ignored}

    all_tests: dict[str, Path] = {
        rel_path: path
        for path in tests_root.rglob("test_*.py")
        for rel_path in [str(path.relative_to(tests_root).with_suffix(""))]
        if rel_path not in ignored_set and path.stem not in ignored_set
    }

    model_compat_files = {
        rel_path
        for rel_path, path in all_tests.items()
        if (not requested_tests or rel_path in requested_tests)
        and "mlx" not in rel_path
        and "ipex" not in rel_path
        and "xpu" not in rel_path
        and re.match(test_regex, rel_path)
        and _is_model_compat_test(rel_path, path)
    }

    torch_files = {
        rel_path
        for rel_path in all_tests
        if (not requested_tests or rel_path in requested_tests)
        and rel_path not in model_compat_files
        and "mlx" not in rel_path
        and "ipex" not in rel_path
        and "xpu" not in rel_path
        and re.match(test_regex, rel_path)
    }

    m4_files = {
        rel_path
        for rel_path in all_tests
        if ("mlx" in rel_path or "apple" in rel_path)
        and (rel_path in requested_tests if requested_tests else True)
        and re.match(test_regex, rel_path)
    }

    return (
        sorted(torch_files, key=_sort_key),
        sorted(model_compat_files, key=_sort_key),
        sorted(m4_files, key=_sort_key),
    )


def build_shared_env_matrix(
    *,
    group: str,
    tests: list[str],
    deps: dict[str, Any],
    config_file: str | Path,
) -> list[dict[str, str]]:
    envs: dict[str, dict[str, str]] = {}
    for test_name in tests:
        if has_specific_dependencies(test_name, deps):
            continue

        env_name = build_env_name(test_name, deps)
        if env_name not in SHARED_COMMON_ENVS:
            continue

        config = parse_test_config(config_file, group, test_name)
        python_version = str(config["py"])

        existing = envs.get(env_name)
        if existing is None:
            envs[env_name] = {
                "env_name": env_name,
                "python_version": python_version,
            }
            continue

        if existing["python_version"] != python_version:
            raise ValueError(
                f"conflicting python_version for env {env_name}: "
                f"{existing['python_version']} vs {python_version}"
            )

    return sorted(envs.values(), key=lambda item: item["env_name"])


def collect_prepare_env_rows(
    torch_envs: list[dict[str, str]],
    model_envs: list[dict[str, str]],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen_env_names: set[str] = set()

    for item in [*torch_envs, *model_envs]:
        env_name = item["env_name"]
        python_version = item["python_version"]
        if env_name in seen_env_names:
            continue
        seen_env_names.add(env_name)
        rows.append((env_name, python_version))

    return rows


def resolve_install_packages(test_name: str, deps_file: str | Path = DEFAULT_DEPS_FILE) -> tuple[list[str], list[str]]:
    deps = load_yaml(deps_file)
    test_path = resolve_test_path(test_name)
    specific_packages, common_packages = collect_packages(test_path, deps)
    specific_keys = {package_install_key(str(package)) for package in specific_packages}
    filtered_common = {
        package
        for package in common_packages
        if package_install_key(str(package)) not in specific_keys
    }
    return (
        [normalize_package_spec(str(package)) for package in specific_packages],
        [normalize_package_spec(str(package)) for package in filtered_common],
    )


def resolve_uninstall_packages(
    test_name: str,
    deps_file: str | Path,
) -> tuple[list[str], list[str]]:
    deps = load_yaml(deps_file)
    test_path = resolve_test_path(test_name)
    specific_packages, common_packages = collect_packages(test_path, deps)
    return (
        sorted(normalize_package_spec(str(package)) for package in specific_packages),
        sorted(normalize_package_spec(str(package)) for package in common_packages),
    )
