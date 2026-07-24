# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from filelock import FileLock

from . import jit as jit_utils
from .cuda import filter_cuda_paths


def _select_nvrtc_lib(lib_dir):
    unversioned = os.path.join(lib_dir, "libnvrtc.so")
    if os.path.exists(unversioned):
        return unversioned
    versioned = sorted(glob.glob(os.path.join(lib_dir, "libnvrtc.so.*")))
    if versioned:
        return versioned[-1]
    return None


def _find_nvrtc_lib_dir():
    env = filter_cuda_paths(required_headers=["nvrtc.h"])
    root = env["path"]
    candidates = [
        os.path.join(root, "lib64"),
        os.path.join(root, "lib"),
        *sorted(glob.glob(os.path.join(root, "*", "lib64"))),
        *sorted(glob.glob(os.path.join(root, "*", "lib"))),
    ]
    for d in candidates:
        lib_path = _select_nvrtc_lib(d)
        if lib_path is not None:
            return d, lib_path, env
    return None, None, env


_cached_binary_path = None


def get_nvrtc_library_path():
    _, lib_path, _ = _find_nvrtc_lib_dir()
    if lib_path is None:
        raise RuntimeError("Could not locate libnvrtc.so in CUDA path")
    return lib_path


def build_nvrtc_compile_binary(output_path, compiler="g++"):
    _, _, cuda_env = _find_nvrtc_lib_dir()
    include_paths = list(cuda_env["include_paths"])
    src_path = Path(__file__).parents[1] / "csrc" / "nvrtc_compile.cpp"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    cmd = [
        compiler,
        "-O2",
        "-std=c++17",
        str(src_path),
        *[f"-I{path}" for path in include_paths],
        "-ldl",
        "-o",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build nvrtc_compile:\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    os.replace(tmp_path, output_path)
    return output_path.as_posix()


def may_build_nvrtc_compile_binary():
    global _cached_binary_path
    if _cached_binary_path is not None:
        return _cached_binary_path

    src_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "nvrtc_compile.cpp")
    src_path = os.path.abspath(src_path)
    lib_dir, lib_path, cuda_env = _find_nvrtc_lib_dir()
    if lib_dir is None:
        raise RuntimeError("Could not locate libnvrtc.so in CUDA path")
    native_path = jit_utils.get_precompiled_artifact_path(src_path, "nvrtc_compile")
    if native_path is not None:
        _cached_binary_path = native_path.as_posix()
        return _cached_binary_path

    src_hash = jit_utils.hash_path_content(src_path, releative=True)
    include_paths = list(cuda_env["include_paths"])
    env_signature = json.dumps(
        {
            "lib_dir": lib_dir,
            "lib_path": lib_path,
            "include_paths": include_paths,
            "path": cuda_env["path"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    full_hash = jit_utils.hash_to_hex(src_hash + "$$" + env_signature)

    build_dir = Path(jit_utils.get_humming_cache_dir()) / "nvrtc_compile" / full_hash
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / "nvrtc_compile"

    if binary_path.exists():
        _cached_binary_path = binary_path.as_posix()
        return _cached_binary_path

    lock_filename = jit_utils.get_humming_lock_filename("nvrtc_compile_" + full_hash)
    with FileLock(lock_filename):
        if binary_path.exists():
            _cached_binary_path = binary_path.as_posix()
            return _cached_binary_path

        _cached_binary_path = build_nvrtc_compile_binary(binary_path)
        return _cached_binary_path


def build_nvrtc_compile_binary_in_bg():
    if os.getenv("HUMMING_DISABLE_PARALLEL_BUILD", "0") == "1":
        return None
    repo_root = Path(__file__).resolve().parents[3]
    cmd = "import gptqmodel.humming.utils.nvrtc; gptqmodel.humming.utils.nvrtc.may_build_nvrtc_compile_binary()"
    env = os.environ.copy()
    env["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    jit_utils.popen_and_reap(
        [sys.executable, "-c", cmd],
        env=env,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


build_nvrtc_compile_binary_in_bg()
