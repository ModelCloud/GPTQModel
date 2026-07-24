# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import functools
import glob
import json
import os
import re
import subprocess
import sys


def _parse_major_minor(s):
    m = re.search(r"(\d+)\.(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1)), 0
    return None, None


def _read_cuda_system_version(cuda_home):
    version_json = os.path.join(cuda_home, "version.json")
    if os.path.isfile(version_json):
        try:
            with open(version_json) as f:
                data = json.load(f)
            v = data.get("cuda", {}).get("version")
            if v:
                return _parse_major_minor(v)
        except (OSError, ValueError):
            pass
    version_txt = os.path.join(cuda_home, "version.txt")
    if os.path.isfile(version_txt):
        try:
            with open(version_txt) as f:
                return _parse_major_minor(f.read())
        except OSError:
            pass

    nvcc_path = os.path.join(cuda_home, "bin/nvcc")
    if os.path.exists(nvcc_path):
        try:
            output = subprocess.check_output(["nvcc", "-V"]).decode()
            m = re.search(r"release (\d+)\.(\d+),", output)
            if m:
                return int(m.group(1)), int(m.group(2))
        except BaseException:
            pass

    base = os.path.basename(os.path.realpath(cuda_home))
    m = re.match(r"cuda-(\d+(?:\.\d+)?)$", base)
    if m:
        return _parse_major_minor(m.group(1))
    return None, None


def _add_include_path(paths, include_dir):
    if not os.path.isdir(include_dir):
        return
    paths.append(include_dir)
    cccl = os.path.join(include_dir, "cccl")
    if os.path.isdir(cccl):
        paths.append(cccl)


def _collect_include_paths(root, recurse_components=False):
    paths = []
    _add_include_path(paths, os.path.join(root, "include"))
    if recurse_components:
        for name in sorted(os.listdir(root)):
            if name.startswith("cu") and name[2:].isdigit():
                continue
            _add_include_path(paths, os.path.join(root, name, "include"))
    return paths


_COMMON_BINARIES = ("nvcc", "ptxas", "cuobjdump", "nvdisasm", "compute-sanitizer")


def _find_binary(root, name):
    candidates = [
        os.path.join(root, "bin", name),
        os.path.join(root, "cuda_nvcc", "bin", name),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _collect_binaries(root):
    result = {}
    for name in _COMMON_BINARIES:
        path = _find_binary(root, name)
        if path is not None:
            result[name] = path
    return result


def _find_nvidia_pypi_cuda_paths():
    results = []
    seen = set()
    for entry in sys.path:
        nvidia_root = os.path.join(entry, "nvidia")
        if not os.path.isdir(nvidia_root) or nvidia_root in seen:
            continue
        seen.add(nvidia_root)
        if any(os.path.isdir(os.path.join(nvidia_root, sub)) for sub in ("cuda_runtime", "cuda_nvcc")):
            results.append(
                {
                    "source": "pypi",
                    "path": nvidia_root,
                    "major": 12,
                    "minor": None,
                    "binaries": _collect_binaries(nvidia_root),
                    "include_paths": _collect_include_paths(nvidia_root, recurse_components=True),
                }
            )
        cu13_root = os.path.join(nvidia_root, "cu13")
        if os.path.isdir(cu13_root) and os.path.isdir(os.path.join(cu13_root, "include")):
            results.append(
                {
                    "source": "pypi",
                    "path": cu13_root,
                    "major": 13,
                    "minor": None,
                    "binaries": _collect_binaries(cu13_root),
                    "include_paths": _collect_include_paths(cu13_root),
                }
            )
    return results


def _parse_target_version(v):
    if isinstance(v, int):
        return v, None
    if isinstance(v, (tuple, list)):
        major = int(v[0])
        minor = int(v[1]) if len(v) > 1 and v[1] is not None else None
        return major, minor
    return _parse_major_minor(str(v))


def filter_cuda_paths(
    target_version: int | str | tuple | list | None = None,
    required_headers: list | None = None,
    required_binaries: list | None = None,
    source: str | None = None,
):
    if target_version is None:
        import torch

        target_major, _ = _parse_major_minor(torch.version.cuda)
        target_minor = None
    else:
        target_major, target_minor = _parse_target_version(target_version)
    headers = required_headers or []
    binaries = required_binaries or []
    results = []
    matched_without_binaries = []

    def has_header(env, header):
        return any(os.path.exists(os.path.join(p, header)) for p in env["include_paths"])

    for env in find_all_cuda_paths():
        if env["major"] != target_major:
            continue
        if target_minor is not None and env["minor"] is not None:
            if env["minor"] != target_minor:
                continue
        if source is not None and env["source"] != source:
            continue
        if not all(has_header(env, h) for h in headers):
            continue
        matched_without_binaries.append(env)
        if not all(b in env["binaries"] for b in binaries):
            continue
        results.append(env)
    if target_minor is None:
        results.sort(key=lambda e: (e["minor"] is None, -(e["minor"] or 0)))
    if results:
        return results[0]

    is_cu12_nvcc_missing = (
        target_major == 12
        and "nvcc" in binaries
        and len(matched_without_binaries) > 0
        and all("nvcc" not in e["binaries"] for e in matched_without_binaries)
    )
    if is_cu12_nvcc_missing:
        raise RuntimeError(
            "No CUDA 12 nvcc found. The nvidia-cuda-nvcc-cu12 PyPI package does not "
            "ship the nvcc driver; please install the CUDA Toolkit "
            "(e.g. `apt install cuda-toolkit-12-x` or "
            "https://developer.nvidia.com/cuda-downloads)."
        )
    if target_major in (12, 13):
        raise RuntimeError(
            f"No suitable CUDA {target_major} environment found. "
            f"Try: pip install humming-kernels[cu{target_major}]"
        )
    raise RuntimeError(f"No suitable CUDA {target_major} environment found.")


@functools.lru_cache(maxsize=1)
def find_all_cuda_paths():
    results = []
    seen_real = set()
    candidates = ["/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"))
    if "CUDA_HOME" in os.environ:
        candidates.append(os.environ["CUDA_HOME"])

    for path in candidates:
        if not os.path.isdir(path):
            continue
        real = os.path.realpath(path)
        if real in seen_real:
            continue
        seen_real.add(real)
        major, minor = _read_cuda_system_version(path)
        results.append(
            {
                "source": "system",
                "path": path,
                "real_path": real,
                "major": major,
                "minor": minor,
                "binaries": _collect_binaries(path),
                "include_paths": _collect_include_paths(path),
            }
        )

    results.extend(_find_nvidia_pypi_cuda_paths())
    return results
