import hashlib
import re
from pathlib import Path
from typing import Any


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


def collect_pkgs(test_path: Path, deps: dict[str, Any]) -> tuple[set[Any], set[Any]]:
    specific_pkgs: set[Any] = set()
    common_pkgs = set(deps.get("common") or [])

    specific_pkgs.update(lookup_test_entry(deps.get("tests"), test_path))

    test_path_str = test_path.as_posix()
    for key, value in deps.items():
        if not (isinstance(key, str) and key.startswith("tests/")):
            continue
        if not test_path_str.startswith(key + "/"):
            continue

        if isinstance(value, list):
            specific_pkgs.update(value)
        elif isinstance(value, dict):
            specific_pkgs.update(lookup_test_entry(value, test_path))

    return specific_pkgs, common_pkgs


def has_specific_deps(test_name: str, deps: dict[str, Any]) -> bool:
    test_path = resolve_test_path(test_name)
    specific_pkgs, _ = collect_pkgs(test_path, deps)
    return bool(specific_pkgs)


def flatten_test_name(test_name: str) -> str:
    return test_name.removeprefix("tests/").removesuffix(".py").replace("/", "_")


def is_model_test(test_name: str) -> bool:
    normalized = test_name.removeprefix("tests/").removesuffix(".py")
    return normalized.startswith("models/")


_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")


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


def pkg_env_key(spec: str) -> str:
    spec = normalize_pkg_spec(spec)
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


def sanitize_env_token(value: str) -> str:
    token = value.lower().replace("_", "_").replace(".", "_")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "deps"


def build_specific_deps_env_suffix(pkgs: set[Any]) -> str:
    tokens = sorted({pkg_env_key(str(pkg)) for pkg in pkgs if str(pkg).strip()})
    if not tokens:
        return "deps"
    if len(tokens) == 1:
        return tokens[0]

    joined = "_".join(tokens)
    if len(joined) <= 48:
        return joined

    digest = hashlib.sha1("|".join(tokens).encode("utf-8")).hexdigest()[:10]
    return f"deps_{digest}"
