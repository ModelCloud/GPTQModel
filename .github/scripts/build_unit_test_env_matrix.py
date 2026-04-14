import argparse
import json
from pathlib import Path

import yaml

from deps_utils import has_specific_deps
from get_unit_test_env_name import build_env_name
from parse_test_config import parse_test_config

SHARED_COMMON_ENVS = {
    "gptqmodel_unit_tests_common",
    "gptqmodel_unit_tests_models_common",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--tests-json", required=True)
    parser.add_argument(
        "--deps-file",
        default=Path(__file__).with_name("deps.yaml"),
    )
    parser.add_argument(
        "--config-file",
        default=Path(__file__).with_name("test.yaml"),
    )
    args = parser.parse_args()

    tests = json.loads(args.tests_json or "[]")
    if not tests:
        print("[]")
        return

    with Path(args.deps_file).open("r", encoding="utf-8") as f:
        deps = yaml.safe_load(f) or {}

    envs: dict[str, dict[str, str]] = {}
    for test_name in tests:
        if has_specific_deps(test_name, deps):
            continue

        env_name = build_env_name(test_name, deps)
        if env_name not in SHARED_COMMON_ENVS:
            continue

        config = parse_test_config(args.config_file, args.group, test_name)
        py = str(config["py"])

        entry = envs.get(env_name)
        if entry is None:
            envs[env_name] = {
                "env_name": env_name,
                "python_version": py,
            }
            continue

        if entry["python_version"] != py:
            raise ValueError(
                f"conflicting python_version for env {env_name}: "
                f"{entry['python_version']} vs {py}"
            )

    print(json.dumps(sorted(envs.values(), key=lambda item: item["env_name"])))


if __name__ == "__main__":
    main()
