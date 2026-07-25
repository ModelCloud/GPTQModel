# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import ctypes
import os
import subprocess
from pathlib import Path

from filelock import FileLock

from . import jit as jit_utils

MODES = (
    "mma_e3m4_a",
    "mma_e3m4_b",
    "mma_e3m4_ab",
    "mma_e0m3_a",
    "mma_e0m3_b",
    "mma_e0m3_ab",
    "cvt_e3m4",
    "cvt_e0m3",
)

_RC_MESSAGE = {
    2: "unknown mode or unreadable file",
    3: "not an sm_120a cubin",
    4: "I/O error while writing",
    5: "unsupported instruction config (e.g. E0M3 requires scale_vec::4X)",
}

_cached_lib = None


def _load_lib(path):
    lib = ctypes.CDLL(path)
    # int cubin_patch(const char* path, const char* mode, int dry, int backup);
    lib.cubin_patch.restype = ctypes.c_int
    lib.cubin_patch.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    # int cubin_patch_buffer(uint8_t* data, size_t n, const char* mode, int dry);
    lib.cubin_patch_buffer.restype = ctypes.c_int
    lib.cubin_patch_buffer.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    return lib


def build_cubin_patcher(output_path, compiler="g++"):
    src_path = Path(__file__).parents[1] / "csrc" / "patch_cubin.cpp"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    cmd = [
        compiler,
        "-O2",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(src_path),
        "-o",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build libcubinpatch.so:\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    os.replace(tmp_path, output_path)
    return output_path.as_posix()


def may_build_cubin_patcher():
    global _cached_lib
    if _cached_lib is not None:
        return _cached_lib

    src_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "patch_cubin.cpp")
    src_path = os.path.abspath(src_path)
    native_path = jit_utils.get_precompiled_artifact_path(src_path, "libcubinpatch.so")
    if native_path is not None:
        _cached_lib = _load_lib(native_path.as_posix())
        return _cached_lib

    full_hash = jit_utils.hash_path_content(src_path, releative=True)

    build_dir = Path(jit_utils.get_humming_cache_dir()) / "cubin_patcher" / full_hash
    build_dir.mkdir(parents=True, exist_ok=True)
    lib_path = build_dir / "libcubinpatch.so"

    if lib_path.exists():
        _cached_lib = _load_lib(lib_path.as_posix())
        return _cached_lib

    lock_filename = jit_utils.get_humming_lock_filename("cubin_patcher_" + full_hash)
    with FileLock(lock_filename):
        if not lib_path.exists():
            build_cubin_patcher(lib_path)

    _cached_lib = _load_lib(lib_path.as_posix())
    return _cached_lib


def patch_cubin(cubin_path, mode, dry_run=False, backup=False):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    if not os.path.exists(cubin_path):
        raise FileNotFoundError(cubin_path)

    lib = may_build_cubin_patcher()
    n = lib.cubin_patch(
        str(cubin_path).encode(),
        mode.encode(),
        1 if dry_run else 0,
        1 if backup else 0,
    )
    if n < 0:
        rc = -n
        raise RuntimeError(
            f"patch_cubin failed on {cubin_path!r} (mode={mode}): {_RC_MESSAGE.get(rc, 'error')} (rc={rc})"
        )
    return n


def patch_cubin_bytes(data, mode, dry_run=False):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")

    lib = may_build_cubin_patcher()
    data = bytes(data)
    buf = ctypes.create_string_buffer(data, len(data))
    n = lib.cubin_patch_buffer(buf, len(data), mode.encode(), 1 if dry_run else 0)
    if n < 0:
        rc = -n
        raise RuntimeError(
            f"patch_cubin_bytes failed (mode={mode}): {_RC_MESSAGE.get(rc, 'error')} (rc={rc})"
        )
    return buf.raw, n


def patch_triton_compiled_kernel(compiled_kernel, mode):
    ck = compiled_kernel
    if getattr(ck, "module", None) is not None:
        return 0

    patched, n = patch_cubin_bytes(ck.asm["cubin"], mode)
    ck.asm["cubin"] = patched
    ck.kernel = patched
    return n


def triton_warmup_and_patch(kernel, *args, mode, grid, **kwargs):
    ck = kernel.warmup(*args, grid=grid, **kwargs)
    patch_triton_compiled_kernel(ck, mode)
    return ck
