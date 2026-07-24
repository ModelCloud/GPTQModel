# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import functools
import glob
import hashlib
import importlib
import os
import platform
import re
import subprocess
import sys
import threading
from pathlib import Path

from filelock import FileLock

from . import jit as jit_utils


def get_native_arch() -> str | None:
    return {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(platform.machine().lower())


def get_precompiled_artifact_path(source_path, artifact_name) -> Path | None:
    arch = get_native_arch()
    if arch is None:
        return None

    source_path = Path(source_path)
    artifact_path = Path(__file__).parents[1] / "_native" / arch / artifact_name
    if not artifact_path.is_file():
        return None

    if source_path.is_dir():
        source_mtime = max(path.stat().st_mtime_ns for path in source_path.rglob("*") if path.is_file())
    else:
        source_mtime = source_path.stat().st_mtime_ns
    return artifact_path if source_mtime < artifact_path.stat().st_mtime_ns else None


def popen_and_reap(cmd, **kwargs):
    proc = subprocess.Popen(cmd, **kwargs)
    threading.Thread(target=proc.wait, name="humming-bg-build-reaper", daemon=True).start()
    return proc


def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()[0:16]


@functools.lru_cache(maxsize=1)
def get_cuda_include_path():
    cuda_include_path = os.getenv("CUDA_INCLUDE_PATH")
    if cuda_include_path is not None:
        return cuda_include_path
    cuda_home_path = os.getenv("CUDA_HOME")
    if cuda_home_path is not None:
        return os.path.join(cuda_home_path, "include")
    return "/usr/local/cuda/include/"


@functools.lru_cache(maxsize=8)
def get_cuda_command_path(name):
    if "/" in name:
        return name
    cuda_command_path = os.getenv(f"CUDA_{name.upper()}_PATH")
    if cuda_command_path is not None:
        return cuda_command_path
    cuda_home_path = os.getenv("CUDA_HOME")
    if cuda_home_path is not None:
        return os.path.join(cuda_home_path, "bin/" + name)

    cuda_command_path = f"/usr/local/cuda/bin/{name}"
    if os.path.exists(cuda_command_path):
        return cuda_command_path

    return name


@functools.lru_cache(maxsize=1)
def get_cuda_nvcc_version(nvcc_path):
    result = subprocess.run([nvcc_path, "--version"], stdout=subprocess.PIPE, text=True).stdout
    re_result = re.findall("release (\\d+\\.\\d+)", result)
    if not re_result:
        raise RuntimeError(f"Invalid NVCC: {nvcc_path}")
    return re_result[0]


@functools.lru_cache(maxsize=1)
def get_humming_tmp_dir() -> str:
    tmp_dir = os.getenv("HUMMING_TMP_DIR")
    if tmp_dir is not None:
        return tmp_dir
    dirname = os.path.join(os.path.expanduser("~"), ".humming/tmp/")
    Path(dirname).mkdir(exist_ok=True, parents=True)
    return dirname


@functools.lru_cache(maxsize=1)
def get_humming_cache_dir() -> str:
    cache_dir = os.getenv("HUMMING_CACHE_DIR")
    if cache_dir is not None:
        return cache_dir
    return os.path.join(os.path.expanduser("~"), ".humming/cache/")


@functools.lru_cache(maxsize=1)
def get_humming_module_dir() -> str:
    tmp_dirname = get_humming_tmp_dir()
    dirname = os.path.join(tmp_dirname, "module/")
    Path(dirname).mkdir(exist_ok=True, parents=True)
    return dirname


@functools.lru_cache(maxsize=1)
def get_humming_lock_dir() -> str:
    tmp_dirname = get_humming_tmp_dir()
    dirname = os.path.join(tmp_dirname, "lock/")
    Path(dirname).mkdir(exist_ok=True, parents=True)
    return dirname


def hash_path_content(path: str, releative: bool = False, text_only: bool = True) -> str:
    data = {}

    assert os.path.exists(path)
    if os.path.isfile(path):
        filename = path.split("/")[-1] if releative else path
        with open(path, "rb") as f:
            data[filename] = str(f.read())
    else:
        pattern = os.path.join(path, "**/*")
        for fullname in sorted(glob.glob(pattern, recursive=True)):
            filename = os.path.relpath(fullname, path) if releative else fullname
            if not os.path.isfile(fullname):
                continue
            try:
                with open(fullname, "r") as f:
                    data[filename] = f.read()
            except UnicodeDecodeError:
                if text_only:
                    continue
                with open(fullname, "rb") as f:
                    data[filename] = str(f.read())

    return hash_to_hex(str(data))


@functools.lru_cache(maxsize=1)
def get_humming_lock_filename(name: str) -> str:
    if not name.endswith(".lock"):
        name = name + ".lock"
    lock_dirname = get_humming_lock_dir()
    return os.path.join(lock_dirname, name)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def make_humming_module(func_name, result):
    dirname = get_humming_module_dir()
    if dirname not in sys.path:
        sys.path.append(dirname)

    content = f"def {func_name}():\n    return {result}\n"
    hash_hex = hash_to_hex(content)
    module_name = "humming_module_" + hash_hex

    lock_filename = jit_utils.get_humming_lock_filename(hash_hex)
    with FileLock(lock_filename):
        filename = module_name + ".py"
        if not (Path(dirname) / filename).exists():
            with open(Path(dirname) / filename, "w") as f:
                f.write(content)
                f.flush()

        importlib.invalidate_caches()
        return importlib.import_module(module_name)
