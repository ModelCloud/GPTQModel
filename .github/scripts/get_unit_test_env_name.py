import argparse
from pathlib import Path

import yaml
from deps_utils import build_specific_deps_env_suffix, collect_pkgs, flatten_test_name, is_model_test, resolve_test_path


def build_env_name(test_name: str, deps: dict) -> str:
    test_path = resolve_test_path(test_name)
    specific_pkgs, _ = collect_pkgs(test_path, deps)
    prefix = "gptqmodel_unit_tests_models" if is_model_test(test_name) else "gptqmodel_unit_tests"

    if specific_pkgs:
        return f"{prefix}_{build_specific_deps_env_suffix(specific_pkgs)}_common"

    if is_model_test(test_name):
        return f"gptqmodel_unit_tests_{flatten_test_name(test_name)}_common"

    return "gptqmodel_unit_tests_common"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-name", required=True)
    parser.add_argument(
        "--deps-file",
        default=Path(__file__).with_name("deps.yaml"),
    )
    args = parser.parse_args()

    with Path(args.deps_file).open("r", encoding="utf-8") as f:
        deps = yaml.safe_load(f) or {}

    print(
        build_env_name(
            test_name=args.test_name,
            deps=deps,
        )
    )


if __name__ == "__main__":
    main()
