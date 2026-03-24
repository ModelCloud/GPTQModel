import argparse
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

TORCH_SOURCE_URL = "https://download.pytorch.org/whl/torch/"

MIN_TORCH_VERSION = "2.8.0"
MAX_TORCH_VERSION = "2.11.0"
MIN_CUDA_VERSION = 126
MAX_CUDA_VERSION = 130
MIN_PYTHON_VERSION = 310
MAX_PYTHON_VERSION = 314

TORCH_WHEEL_RE = re.compile(
    r"^torch-(?P<torch_version>\d+\.\d+\.\d+(?:\.post\d+)?)"
    r"(?:\+(?P<build_tag>[^-]+))?"
    r"-cp(?P<python_tag>\d+)-cp(?P<abi_tag>\d+t?)-(?P<platform>[^\"/]+)\.whl$"
)
WHEEL_LINK_RE = re.compile(r'href=["\']([^"\']+\.whl)["\']')
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
    for link in WHEEL_LINK_RE.findall(index_html):
        parsed = urllib.parse.urlsplit(link)
        wheel_names.append(urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]))
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

    return sorted(
        filtered,
        key=lambda build: (version_key(build.torch), build.cuda, python_key(build.python)),
        reverse=True,
    )


def build_matrix(builds: list[ReleaseBuild]) -> dict[str, list[dict[str, int | str]]]:
    return {"include": [build.as_matrix_item() for build in builds]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=TORCH_SOURCE_URL)
    parser.add_argument(
        "--mode",
        choices=("matrix", "versions"),
        default="matrix",
        help="matrix: output GitHub Actions matrix JSON; versions: output parsed torch version list",
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

    if args.mode == "versions":
        output = collect_version_list(wheel_names)
    else:
        builds = filter_release_builds(
            builds=parse_release_builds(wheel_names),
            min_torch_version=args.min_torch_version,
            max_torch_version=args.max_torch_version,
            min_cuda_version=args.min_cuda_version,
            max_cuda_version=args.max_cuda_version,
            min_python_version=args.min_python_version,
            max_python_version=args.max_python_version,
        )
        if not builds:
            raise RuntimeError(
                "no matching torch builds found; check network access, source URL, "
                "wheel link encoding, or the configured torch/cuda/python version ranges"
            )
        output = build_matrix(builds)

    print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
