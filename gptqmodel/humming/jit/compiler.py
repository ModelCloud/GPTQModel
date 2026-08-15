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
from pathlib import Path
from typing import Callable

import torch
from cuda.bindings import nvrtc
from filelock import FileLock

from ...utils.cpp import cuda_split_compile_flags
from ..utils import jit as jit_utils
from ..utils.cuda import filter_cuda_paths
from ..utils.nvrtc import get_nvrtc_library_path, may_build_nvrtc_compile_binary


class Compiler:
    @staticmethod
    def debug_kernel_flags():
        enabled = os.environ.get("HUMMING_DEBUG_KERNEL", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        flags = [f"-DHUMMING_DEBUG_KERNEL={int(enabled)}"]
        if enabled:
            clock_rate_khz = torch.cuda.get_device_properties().clock_rate
            timeout_clocks = int(clock_rate_khz) * 1000 * 10
            flags.append(f"-DHUMMING_DEBUG_KERNEL_TIMEOUT_CLOCKS={timeout_clocks}")
            flags.append("-lineinfo")
        return flags

    @classmethod
    def signature(self):
        raise NotImplementedError

    @staticmethod
    def humming_include_dir():
        dirname = os.path.dirname(__file__)
        dirname = os.path.abspath(dirname + "/../include/")
        return dirname

    @staticmethod
    def include_dirs():
        return [Compiler.humming_include_dir()]

    @staticmethod
    def cuh_last_update_time():
        dirname = Compiler.humming_include_dir()
        data = {}
        for filename in sorted(glob.glob(f"{dirname}/**/*.cuh", recursive=True)):
            data[filename] = os.stat(filename).st_mtime
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def compile(
        cls,
        code,
        sm_version,
        kernel_expr,
        disable_fast_math=False,
        postprocess_cubin: Callable | None = None,
    ):
        flags = cls.get_flags(sm_version, disable_fast_math)
        signature = f"{cls.__name__}$${cls.signature()}$${flags}$${kernel_expr}$${code}"
        signature += "$$" + Compiler.cuh_last_update_time()
        hash_hex = jit_utils.hash_to_hex(signature)

        cache_dirname = Path(os.path.join(jit_utils.get_humming_cache_dir(), hash_hex))
        cache_filename = cache_dirname / "kernel.cubin"
        cache_dirname.mkdir(exist_ok=True, parents=True)

        lock_filename = jit_utils.get_humming_lock_filename(hash_hex)
        with FileLock(lock_filename):
            if cache_filename.exists():
                return cache_filename.as_posix()

            cache_dirname.mkdir(exist_ok=True, parents=True)
            source_path = os.path.join(cache_dirname, "kernel.cu")
            with open(cache_dirname / "kernel.cu", "w") as f:
                f.write(code)
            with open(cache_dirname / "signature.txt", "w") as f:
                f.write(signature)

            compile_res = cls._compile(source_path, cache_dirname, sm_version, kernel_expr, flags)
            returncode, stdout, stderr = compile_res

            if returncode == 0 and postprocess_cubin is not None:
                filename = cache_dirname / "kernel_tmp.cubin"
                postprocess_cubin(filename.as_posix())

            with open(cache_dirname / "stdout.log", "w") as f:
                f.write(stdout)
            with open(cache_dirname / "stderr.log", "w") as f:
                f.write(stderr)

        if returncode != 0:
            print(stderr, flush=True)
            (cache_dirname / "kernel_tmp.cubin").unlink(missing_ok=True)
            raise RuntimeError(f"{cls} run failed")

        os.replace(cache_dirname / "kernel_tmp.cubin", cache_dirname / "kernel.cubin")
        return cache_filename.as_posix()

    @classmethod
    def get_flags(cls, sm_version, disable_fast_math=False):
        raise NotImplementedError

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr):
        raise NotImplementedError


class NVRTCCompiler(Compiler):
    _STD_HEADER_SHIMS: dict[str, str] = {
        "climits": "#include <cuda/std/climits>",
        "cfloat": "#include <cuda/std/cfloat>",
        "cstddef": """
            #include <cuda/std/cstddef>
            using namespace cuda::std;
        """,
        "cstdint": """
            #include <cuda/std/cstdint>
            #include <cuda/std/type_traits>
            using namespace cuda::std;
            namespace std {
            using cuda::std::is_same;
            using cuda::std::conditional_t;
            using cuda::std::conditional;
            using cuda::std::enable_if;
            using cuda::std::enable_if_t;
            }
        """,
        "type_traits": """
            #include <cuda/std/type_traits>
            namespace std {
            using cuda::std::is_same;
            using cuda::std::conditional_t;
            using cuda::std::conditional;
            using cuda::std::enable_if;
            using cuda::std::enable_if_t;
            }
        """,
        "cuda.h": """
            #pragma once
            #include <cuda/std/cstdint>
            using namespace cuda::std;
            typedef uint64_t cuuint64_t;
            #define CU_TENSOR_MAP_NUM_QWORDS 16
            typedef struct CUtensorMap_st {
                alignas(64) cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
            } CUtensorMap;
        """,
    }

    @classmethod
    def _get_std_header_shims(cls):
        names = []
        sources = []

        for header, content in cls._STD_HEADER_SHIMS.items():
            names.append(header.encode())
            sources.append(content.encode())
        return names, sources

    @classmethod
    def signature(cls):
        _, major, minor = nvrtc.nvrtcVersion()
        return f"nvrtc+{major}.{minor}"

    @classmethod
    def get_flags(cls, sm_version, disable_fast_math=False):
        flags = [
            f"--gpu-architecture=sm_{sm_version}",
            "-std=c++17",
            *cls.debug_kernel_flags(),
            "--use_fast_math",
            "--dopt=on",
            "-extra-device-vectorization",
            "--ptxas-options=-O3",
            "--ptxas-options=--register-usage-level=10",
            "--diag-suppress=39,161,174,177,940,1444",
            "-default-device",
        ]
        for d in cls._get_include_dirs():
            flags.append(f"-I{d}")
        if disable_fast_math:
            flags.remove("--use_fast_math")
        return flags

    @classmethod
    def _get_include_dirs(cls):
        env = filter_cuda_paths(required_headers=["cuda_runtime.h"])
        return list(cls.include_dirs()) + list(env["include_paths"])

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr, flags):
        binary_path = may_build_nvrtc_compile_binary()

        shims_dir = Path(cache_dirname) / "shims"
        shims_dir.mkdir(exist_ok=True)
        header_args = []
        for header, content in cls._STD_HEADER_SHIMS.items():
            shim_file = shims_dir / header
            with open(shim_file, "w") as f:
                f.write(content)
            header_args += ["--header", f"{header}={shim_file.as_posix()}"]

        target_path = (Path(cache_dirname) / "kernel_tmp.cubin").as_posix()
        cmd = [
            binary_path,
            "--nvrtc-path",
            get_nvrtc_library_path(),
            "--input",
            source_path,
            "--output",
            target_path,
            *header_args,
        ]
        if kernel_expr:
            name_expr = " ".join(kernel_expr.split())
            cmd += ["--name-expression", name_expr]
        cmd += ["--", *flags]

        with open(Path(cache_dirname) / "cmdline.json", "w") as f:
            json.dump(cmd, f, ensure_ascii=False)

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode, result.stdout, result.stderr


class NVCCCompiler(Compiler):
    @classmethod
    def _get_env(cls):
        return filter_cuda_paths(required_binaries=["nvcc"])

    @classmethod
    def signature(cls):
        nvcc_path = cls._get_env()["binaries"]["nvcc"]
        nvcc_version = jit_utils.get_cuda_nvcc_version(nvcc_path)
        return "nvcc+" + nvcc_version

    @classmethod
    def get_flags(cls, sm_version, disable_fast_math=False):
        nvcc_path = cls._get_env()["binaries"]["nvcc"]
        nvcc_version = jit_utils.get_cuda_nvcc_version(nvcc_path)
        cxx_flags = [
            "-fPIC",
            "-O3",
            "-Wno-deprecated-declarations",
            "-Wno-abi",
            "-fopenmp",
            "-lgomp",
        ]

        flags = [
            "-std=c++17",
            *cls.debug_kernel_flags(),
            "--ptxas-options=--register-usage-level=10",
            "--use_fast_math",
            "--diag-suppress=39,161,174,177,940,1444",
            *[f"-I{d}" for d in cls.include_dirs()],
            f"-gencode=arch=compute_{sm_version},code=sm_{sm_version}",
            "-cubin",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            f"--compiler-options={','.join(cxx_flags)}",
            *cuda_split_compile_flags(nvcc_version=nvcc_version),
        ]
        if disable_fast_math:
            flags.remove("--use_fast_math")
        return flags

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr, flags):
        if kernel_expr:
            with open(source_path, "a") as f:
                f.write(f"\nauto ptr = reinterpret_cast<void*>(&{kernel_expr});\n")

        nvcc_path = cls._get_env()["binaries"]["nvcc"]
        target_path = (cache_dirname / "kernel_tmp.cubin").as_posix()

        cmd = [nvcc_path, source_path, "-o", target_path] + flags
        with open(cache_dirname / "cmdline.json", "w") as f:
            json.dump(cmd, f, ensure_ascii=False)

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode, result.stdout, result.stderr
