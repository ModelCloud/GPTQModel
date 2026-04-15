import argparse
import json

import requests
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def get_versions(package: str, version_spec: str) -> list[str]:
    specifier = SpecifierSet(version_spec)

    url = f"https://pypi.org/pypi/{package}/json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    all_versions = data["releases"].keys()

    matched = sorted(
        (Version(v) for v in all_versions if Version(v) in specifier),
        reverse=True,
    )
    return [str(v) for v in matched]


def main():
    parser = argparse.ArgumentParser(description="List matching PyPI versions as JSON")
    parser.add_argument("package", help="package name, e.g. setuptools")
    parser.add_argument("version", help='version spec, e.g. ">=77.0.1,<83"')
    args = parser.parse_args()

    versions = get_versions(args.package, args.version)
    print(json.dumps(versions))


if __name__ == "__main__":
    main()
