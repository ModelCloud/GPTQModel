import json
from pathlib import Path
from typing import Any

import yaml


def parse_test_config(
        yaml_file: str | Path,
        group: str,
        test_name: str | None = None,
) -> dict[str, Any]:
    yaml_path = Path(yaml_file)
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

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

    # Group-level shared config overrides common defaults.
    for key, value in group_data.items():
        if not isinstance(value, dict):
            result[key] = value

    # Per-test config overrides group/common defaults. Missing per-test entries
    # fall back to the merged defaults.
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml-file",
        default=Path(__file__).with_name("test.yaml"),
    )
    parser.add_argument("--group", required=True)
    parser.add_argument("--test-name")
    parser.add_argument(
        "--flags-only",
        action="store_true",
        help="Return only config keys with value 1, for example: {'py': 1, 'gpu': 1}",
    )
    args = parser.parse_args()

    if args.flags_only:
        result = parse_test_config_flags(args.yaml_file, args.group, args.test_name)
    else:
        result = parse_test_config(args.yaml_file, args.group, args.test_name)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
