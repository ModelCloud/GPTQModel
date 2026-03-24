import argparse
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

TORCH_SOURCE_URL = "https://download.pytorch.org/whl/torch/"

MIN_TORCH_VERSION = "2.9.0"
MAX_TORCH_VERSION = "2.11.0"
MIN_CUDA_VERSION = 128
MAX_CUDA_VERSION = 130
MIN_PYTHON_VERSION = 310
MAX_PYTHON_VERSION = 314

TORCH_WHEEL_RE = re.compile(
    r"^torch-(?P<torch_version>\d+\.\d+\.\d+(?:\.post\d+)?)"
    r"(?:\+(?P<build_tag>[^-]+))?"
    r"-cp(?P<python_tag>\d+)-cp(?P<abi_tag>\d+t?)-(?P<platform>[^\"/]+)\.whl$"
)
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']')
VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:\.post(?P<post>\d+))?$"
)


@dataclass(frozen=True)
class ReleaseBuild:
    torch: str
    cuda: int
    python: str

    def as_matrix_item(self) -> dict[str, int | str]:
        return {
            "cuda": self.cuda,
            "torch": self.torch,
            "python": self.python,
        }


def version_key(version: str) -> tuple[int, int, int, int]:
    match = VERSION_RE.fullmatch(version)
    if not match:
        raise ValueError(f"unsupported torch version format: {version}")
    return tuple(int(match.group(name) or 0) for name in ("major", "minor", "patch", "post"))


def major_minor_key(version: str) -> tuple[int, int]:
    major, minor, _, _ = version_key(version)
    return major, minor


def python_key(python_version: str) -> tuple[int, int]:
    free_threaded = python_version.endswith("t")
    base = int(python_version[:-1] if free_threaded else python_version)
    return base, 0 if free_threaded else 1


def fetch_torch_index(source_url: str) -> str:
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "GPTQModel release matrix generator"},
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def extract_wheel_names(index_html: str) -> list[str]:
    wheel_names = []
    for link in HREF_RE.findall(index_html):
        parsed = urllib.parse.urlsplit(link)
        wheel_name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
        if wheel_name.endswith(".whl"):
            wheel_names.append(wheel_name)
    return wheel_names


def is_linux_x86_64(platform: str) -> bool:
    return "x86_64" in platform and ("linux" in platform or "manylinux" in platform)


def collect_version_list(wheel_names: list[str]) -> list[str]:
    versions = set()
    for wheel_name in wheel_names:
        match = TORCH_WHEEL_RE.match(wheel_name)
        if match:
            versions.add(match.group("torch_version"))
    return sorted(versions, key=version_key, reverse=True)


def collapse_versions_by_major_minor(versions: list[str]) -> list[str]:
    selected: dict[tuple[int, int], str] = {}
    for version in versions:
        key = major_minor_key(version)
        current = selected.get(key)
        if current is None or version_key(version) < version_key(current):
            selected[key] = version
    return sorted(selected.values(), key=version_key, reverse=True)


def parse_release_builds(wheel_names: list[str]) -> list[ReleaseBuild]:
    builds = set()
    for wheel_name in wheel_names:
        match = TORCH_WHEEL_RE.match(wheel_name)
        if not match:
            continue

        build_tag = match.group("build_tag") or ""
        platform = match.group("platform")
        if not build_tag.startswith("cu") or not is_linux_x86_64(platform):
            continue

        builds.add(
            ReleaseBuild(
                torch=match.group("torch_version"),
                cuda=int(build_tag.removeprefix("cu")),
                python=match.group("abi_tag"),
            )
        )

    return list(builds)


def filter_release_builds(
        builds: list[ReleaseBuild],
        min_torch_version: str,
        max_torch_version: str,
        min_cuda_version: int,
        max_cuda_version: int,
        min_python_version: int,
        max_python_version: int,
) -> list[ReleaseBuild]:
    min_torch_key = version_key(min_torch_version)
    max_torch_key = version_key(max_torch_version)

    filtered = []
    for build in builds:
        current_torch_key = version_key(build.torch)
        current_python = int(build.python.removesuffix("t"))
        if current_torch_key < min_torch_key or current_torch_key > max_torch_key:
            continue
        if build.cuda < min_cuda_version or build.cuda > max_cuda_version:
            continue
        if current_python < min_python_version or current_python > max_python_version:
            continue
        filtered.append(build)

    allowed_versions = set(
        collapse_versions_by_major_minor(sorted({build.torch for build in filtered}, key=version_key))
    )
    filtered = [build for build in filtered if build.torch in allowed_versions]

    return sorted(
        filtered,
        key=lambda build: (version_key(build.torch), build.cuda, python_key(build.python)),
        reverse=True,
    )


def build_matrix(builds: list[ReleaseBuild]) -> dict[str, list[dict[str, int | str]]]:
    return {"include": [build.as_matrix_item() for build in builds]}


def count_builds_by_torch(builds: list[ReleaseBuild]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for build in builds:
        counts[build.torch] = counts.get(build.torch, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: version_key(item[0]), reverse=True))


def build_debug_report(
        source_url: str,
        wheel_names: list[str],
        parsed_builds: list[ReleaseBuild],
        filtered_builds: list[ReleaseBuild],
) -> dict[str, object]:
    return {
        "source_url": source_url,
        "wheel_name_count": len(wheel_names),
        "version_list": collect_version_list(wheel_names),
        "collapsed_version_list": collapse_versions_by_major_minor(collect_version_list(wheel_names)),
        "parsed_release_build_count": len(parsed_builds),
        "parsed_release_build_count_by_torch": count_builds_by_torch(parsed_builds),
        "filtered_release_build_count": len(filtered_builds),
        "filtered_release_build_count_by_torch": count_builds_by_torch(filtered_builds),
        "filtered_release_builds": [build.as_matrix_item() for build in filtered_builds],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=TORCH_SOURCE_URL)
    parser.add_argument(
        "--mode",
        choices=("matrix", "versions", "debug"),
        default="matrix",
        help="matrix: output GitHub Actions matrix JSON; versions: output parsed torch version list; debug: output diagnostic JSON",
    )
    parser.add_argument("--min-torch-version", default=MIN_TORCH_VERSION)
    parser.add_argument("--max-torch-version", default=MAX_TORCH_VERSION)
    parser.add_argument("--min-cuda-version", type=int, default=MIN_CUDA_VERSION)
    parser.add_argument("--max-cuda-version", type=int, default=MAX_CUDA_VERSION)
    parser.add_argument("--min-python-version", type=int, default=MIN_PYTHON_VERSION)
    parser.add_argument("--max-python-version", type=int, default=MAX_PYTHON_VERSION)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wheel_names = extract_wheel_names(fetch_torch_index(args.source_url))
    parsed_builds = parse_release_builds(wheel_names)
    filtered_builds = filter_release_builds(
        builds=parsed_builds,
        min_torch_version=args.min_torch_version,
        max_torch_version=args.max_torch_version,
        min_cuda_version=args.min_cuda_version,
        max_cuda_version=args.max_cuda_version,
        min_python_version=args.min_python_version,
        max_python_version=args.max_python_version,
    )

    if args.mode == "versions":
        output = collect_version_list(wheel_names)
    elif args.mode == "debug":
        output = build_debug_report(
            source_url=args.source_url,
            wheel_names=wheel_names,
            parsed_builds=parsed_builds,
            filtered_builds=filtered_builds,
        )
    else:
        if not filtered_builds:
            raise RuntimeError(
                "no matching torch builds found; check network access, source URL, "
                "wheel link encoding, or the configured torch/cuda/python version ranges"
            )
        output = build_matrix(filtered_builds)

    print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
