# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.utils.cpp_extension
from filelock import FileLock

from ..utils import jit as jit_utils
from ..utils.cuda import filter_cuda_paths

_libs = {}
_launcher_inited = False


def register_op(
    name: str,
    impl_func: Callable,
    fake_impl_func: Callable | None = None,
    mutates_args: list[str] | None = None,
):
    mutates_args = [] if mutates_args is None else mutates_args
    schema_str = torch.library.infer_schema(impl_func, mutates_args=mutates_args)
    lib_name, op_name = name.split("::")

    if lib_name not in _libs:
        _lib = torch.library.Library(lib_name, "FRAGMENT")
        _libs[lib_name] = _lib

    _lib = _libs[lib_name]
    _lib.define(op_name + schema_str)
    _lib.impl(op_name, impl_func, dispatch_key="CUDA")
    if fake_impl_func is not None:
        with _shield_lazy_modules():
            _lib._register_fake(op_name, fake_impl_func)


@contextlib.contextmanager
def _shield_lazy_modules():
    saved = {}
    for name, mod in list(sys.modules.items()):
        if mod is not None and type(mod).__name__ == "_LazyModule":
            saved[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        sys.modules.update(saved)


def get_humming_launcher_build_dir(use_torch_stable_api: bool):
    import gptqmodel.humming as humming

    dirname = os.path.dirname(humming.__file__)
    launcher_code_hash = jit_utils.hash_path_content(
        path=os.path.join(dirname, "csrc/launcher/"),
        releative=True,
    )

    cache_dir = jit_utils.get_humming_cache_dir()
    torch_major, torch_minor = torch.__version__.split(".")[:2]
    version = "torch211_stable" if use_torch_stable_api else f"torch{torch_major}{torch_minor}_nostable"

    launcher_build_dir = os.path.join(cache_dir, f"launcher/{version}/{launcher_code_hash}")
    Path(launcher_build_dir).mkdir(exist_ok=True, parents=True)
    return launcher_build_dir


def _resolve_use_torch_stable_api() -> bool:
    from packaging.version import Version

    override = os.environ.get("HUMMING_USE_TORCH_STABLE_API")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    return Version(torch.__version__) >= Version("2.11")


def _get_precompiled_launcher_path() -> Path | None:
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.11"):
        return None

    humming_dir = Path(__file__).parents[1]
    csrc_dir = humming_dir / "csrc" / "launcher"
    return jit_utils.get_precompiled_artifact_path(csrc_dir, "libhumming_launcher.so")


def init_humming_launcher():
    global _launcher_inited
    if _launcher_inited:
        return

    USE_TORCH_STABLE_API = _resolve_use_torch_stable_api()
    lock_filename = jit_utils.get_humming_lock_filename("launcher")
    with FileLock(lock_filename):
        precompiled_path = _get_precompiled_launcher_path() if USE_TORCH_STABLE_API else None
        if precompiled_path is not None:
            torch.ops.load_library(str(precompiled_path))
            _launcher_inited = True
            return

        import gptqmodel.humming as humming

        build_dir = get_humming_launcher_build_dir(USE_TORCH_STABLE_API)
        torch_lock_file = os.path.join(build_dir, "lock")
        if os.path.exists(torch_lock_file):
            os.unlink(torch_lock_file)

        dirname = os.path.dirname(humming.__file__)
        filename = os.path.join(dirname, "csrc/launcher/launcher.cpp")

        cuda_env = filter_cuda_paths(
            required_headers=["cuda.h", "crt/host_defines.h", "cuda/std/cstdint"],
        )

        extra_cflags = ["-O3", f"-DUSE_TORCH_STABLE_API={int(USE_TORCH_STABLE_API)}"]
        if USE_TORCH_STABLE_API:
            extra_cflags.append("-DTORCH_TARGET_VERSION=0x020B000000000000")

        torch.utils.cpp_extension.load(
            name="humming_launcher",
            sources=[filename],
            extra_include_paths=list(cuda_env["include_paths"]),
            extra_ldflags=["-lcuda", "-lc10_cuda", "-ltorch_cuda"],
            extra_cflags=extra_cflags,
            build_directory=build_dir,
            is_python_module=False,
        )

        _launcher_inited = True


def build_humming_launcher_in_bg():
    if os.getenv("HUMMING_DISABLE_PARALLEL_BUILD", "0") == "1":
        return None
    repo_root = Path(__file__).resolve().parents[3]
    cmd = "import gptqmodel.humming.ops.utils; gptqmodel.humming.ops.utils.init_humming_launcher()"
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


build_humming_launcher_in_bg()
