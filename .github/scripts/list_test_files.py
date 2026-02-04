# file: .github/scripts/list_test_files.py
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional


def _split_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _strip_py_suffix(name: str) -> str:
    return name.removesuffix(".py")


def getFiles(
        ignored_test_files: Union[str, List[str]],
        test_names: str = "",
        test_regex: str = ".*",
        tests_root: Union[str, Path] = "tests",
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      (torch_test_files, m4_test_files)
      - torch_test_files: tests/**/test_*.py excluding mlx / ipex / xpu
      - m4_test_files: tests/**/test_*.py that contains mlx or apple
    """
    tests_root = Path(tests_root)

    input_test_files_list = [_strip_py_suffix(f) for f in _split_csv(test_names)]

    ignored_list = (
        ignored_test_files if isinstance(ignored_test_files, list) else _split_csv(ignored_test_files)
    )
    ignored_set = set(_strip_py_suffix(x) for x in ignored_list)

    # all tests under tests/**/test_*.py (includes tests/models/**)
    all_tests = {
        str(p.relative_to(tests_root).with_suffix(""))
        for p in tests_root.rglob("test_*.py")
        if p.stem not in ignored_set
    }

    # torch tests
    torch_test_files = {
        f
        for f in all_tests
        if (not input_test_files_list or f in input_test_files_list)
           and "mlx" not in f
           and "ipex" not in f
           and "xpu" not in f
           and re.match(test_regex, f)
    }

    # m4 tests
    m4_test_files = {
        f
        for f in all_tests
        if ("mlx" in f or "apple" in f)
           and ((f in input_test_files_list) if input_test_files_list else True)
           and re.match(test_regex, f)
    }

    return sorted(torch_test_files), sorted(m4_test_files)


def main() -> None:
    """
    Usage:
      test_files=$(python3 .github/scripts/list_test_files.py \
        --ignored-test-files "$IGNORED_TEST_FILES" \
        --test-names "${{ github.event.inputs.test_names }}" \
        --test-regex "${{ github.event.inputs.test_regex }}")

    Output:
      <torch_json>|<m4_json>
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ignored-test-files", default=os.getenv("IGNORED_TEST_FILES", ""))
    parser.add_argument("--test-names", default=os.getenv("TEST_NAMES", ""))
    parser.add_argument("--test-regex", default=os.getenv("TEST_REGEX", ".*"))
    parser.add_argument("--tests-root", default="tests")
    args = parser.parse_args()

    torch_files, m4_files = getFiles(
        ignored_test_files=args.ignored_test_files,
        test_names=args.test_names,
        test_regex=args.test_regex,
        tests_root=args.tests_root,
    )

    print(f"{json.dumps(torch_files)}|{json.dumps(m4_files)}")


if __name__ == "__main__":
    main()
