import argparse
from pathlib import Path

import yaml
from deps_utils import has_specific_deps


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

    print("1" if has_specific_deps(args.test_name, deps) else "0")


if __name__ == "__main__":
    main()
