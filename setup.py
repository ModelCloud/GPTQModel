# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except BaseException:
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except BaseException:
        sys.exit(
            "Both latest setuptools and wheel package are not found. "
            "Please upgrade to latest setuptools: `pip install -U setuptools`"
        )

# ---------------------------
# Helpers (no torch required)
# ---------------------------

def _detect_cuda_arch_list():
    # Step 1: check env override
    env_arch = _read_env("CUDA_ARCH_LIST")
    if env_arch:
        return env_arch

    # Step 2: try nvcc
    nvcc_out = _probe_cmd(["nvcc", "--list-gpu-arch"])
    if nvcc_out:
        # output lines like: "    sm_35" / "    sm_80"
        archs = []
        for line in nvcc_out.splitlines():
            line = line.strip()
            if line.startswith("sm_") or line.startswith("compute_"):
                try:
                    major = int(line.split("_")[1][0])
                    if major >= 6:  # only keep >= 6.0
                        archs.append(line.replace("compute_", "").replace("sm_", ""))
                except Exception:
                    continue
        if archs:
            return " ".join(sorted(set(a.replace("sm_", "") for a in archs)))

    # Step 3: try nvidia-smi (compute capability query)
    smi_out = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi_out:
        caps = []
        for line in smi_out.splitlines():
            line = line.strip()
            if line:
                caps.append(line)
        if caps:
            return " ".join(caps)

    # Step 4: fallback
    print("⚠️  Could not auto-detect CUDA arch list. Defaulting to 6.0+PTX")
    return "6.0+PTX"

def _read_env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default

def _probe_cmd(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None

def _probe_cuda_version():
    # Prefer env override
    v = _read_env("CUDA_VERSION")
    if v:
        return v
    # Try nvcc
    nvcc = _probe_cmd(["nvcc", "--version"])
    if nvcc:
        # Extract like "release 12.1"
        for tok in nvcc.replace(",", " ").split():
            if tok.count(".") == 1 and tok.replace(".", "").isdigit():
                return tok
    # Try nvidia-smi
    smi = _probe_cmd(["nvidia-smi"])
    if smi and "CUDA Version" in smi:
        # e.g. "CUDA Version: 12.2"
        import re
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", smi)
        if m:
            return m.group(1)
    return None

def _probe_rocm_version():
    v = _read_env("ROCM_VERSION")
    if v:
        return v
    # Try hipcc --version
    hip = _probe_cmd(["hipcc", "--version"])
    if hip:
        # Grab first x.y looking token
        import re
        m = re.search(r"\b([0-9]+\.[0-9]+)\b", hip)
        if m:
            return m.group(1)
    # Try /opt/rocm/.info/version
    try:
        p = Path("/opt/rocm/.info/version")
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None

def _bool_env(name, default=False):
    v = _read_env(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

def _parse_arch_list(s):
    # "8.0 8.6+PTX 9.0" -> ["8.0", "8.6+PTX", "9.0"]
    return [tok for tok in s.split() if tok.strip()]

def _has_cuda_v8_from_arch_list(arch_list):
    try:
        return any(float(a.split("+")[0]) >= 8.0 for a in arch_list)
    except Exception:
        return False

def _detect_cxx11_abi():
    # Prefer env override to avoid torch dependency
    v = _read_env("CXX11_ABI")
    if v in ("0", "1"):
        return int(v)
    # Fallback default (modern distros): 1
    return 1

def _torch_version_for_tag(torch_mod):
    # Return "major.minor" string for tag
    if torch_mod is None:
        v = _read_env("TORCH_VERSION")
        if v:
            return ".".join(v.split(".")[:2])
        return None
    try:
        return ".".join(torch_mod.__version__.split(".")[:2])
    except Exception:
        return None

def _cuda_version_for_tag(torch_mod):
    if torch_mod is None:
        v = _probe_cuda_version()
        return v
    try:
        return torch_mod.version.cuda  # e.g. "12.1"
    except Exception:
        return None

def _is_rocm(torch_mod):
    if torch_mod is None:
        return _probe_rocm_version() is not None
    try:
        return bool(torch_mod.version.hip)
    except Exception:
        return False

# ---------------------------
# Env and versioning
# ---------------------------

RELEASE_MODE = _read_env("RELEASE_MODE")
TORCH_CUDA_ARCH_LIST = _read_env("TORCH_CUDA_ARCH_LIST")
ROCM_VERSION = _probe_rocm_version()
SKIP_ROCM_VERSION_CHECK = _read_env("SKIP_ROCM_VERSION_CHECK")
BUILD_CUDA_EXT = _read_env("BUILD_CUDA_EXT", "1" if sys.platform != "darwin" else "0")
FORCE_BUILD = _bool_env("GPTQMODEL_FORCE_BUILD", False)

if ROCM_VERSION and not SKIP_ROCM_VERSION_CHECK:
    try:
        if float(ROCM_VERSION) < 6.2:
            sys.exit(
                "GPTQModel's compatibility with ROCm < 6.2 has not been verified. "
                "Set SKIP_ROCM_VERSION_CHECK=1 to proceed."
            )
    except Exception:
        pass

if TORCH_CUDA_ARCH_LIST:
    # sanitize list to >= sm_60
    archs = _parse_arch_list(TORCH_CUDA_ARCH_LIST)
    kept = []
    for arch in archs:
        try:
            if float(arch.split("+")[0]) >= 6.0:
                kept.append(arch)
            else:
                print(f"we do not support this compute arch: {arch}, skipped.")
        except Exception:
            kept.append(arch)
    new_list = " ".join(kept)
    if new_list != TORCH_CUDA_ARCH_LIST:
        os.environ["TORCH_CUDA_ARCH_LIST"] = new_list
        print(f"TORCH_CUDA_ARCH_LIST has been updated to '{new_list}'")

version_vars = {}
exec("exec(open('gptqmodel/version.py').read()); version=__version__", {}, version_vars)
gptqmodel_version = version_vars["version"]

BASE_WHEEL_URL = (
    "https://github.com/ModelCloud/GPTQModel/releases/download/{tag_name}/{wheel_name}"
)

def get_version_tag() -> str:
    if BUILD_CUDA_EXT != "1":
        return "cpu"

    if ROCM_VERSION:
        return f"rocm{'.'.join(str(ROCM_VERSION).split('.')[:2])}"

    cuda_version = _cuda_version_for_tag(torch_mod) or _probe_cuda_version()
    if not cuda_version:
        print(
            "Trying to compile GPTQModel for CUDA, but no CUDA version was detected. "
            "Set CUDA_VERSION env (e.g. 12.1)."
        )
        sys.exit(1)

    CUDA_VERSION = "".join(cuda_version.split("."))  # e.g. 12.1 -> "121"
    tv = _torch_version_for_tag(torch_mod)
    torch_tag = f"torch{tv}" if tv else "torchNA"

    return f"cu{CUDA_VERSION[:3]}{torch_tag}"

requirements = []
if not os.getenv("CI"):
    with open("requirements.txt", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip()]

# Decide HAS_CUDA_V8 without torch if possible
HAS_CUDA_V8 = False
if TORCH_CUDA_ARCH_LIST:
    HAS_CUDA_V8 = not ROCM_VERSION and _has_cuda_v8_from_arch_list(_parse_arch_list(TORCH_CUDA_ARCH_LIST))
else:
    # Soft probe via nvidia-smi
    smi = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi:
        try:
            caps = [float(x.strip()) for x in smi.splitlines() if x.strip()]
            HAS_CUDA_V8 = any(cap >= 8.0 for cap in caps)
        except Exception:
            HAS_CUDA_V8 = False

if RELEASE_MODE == "1":
    gptqmodel_version = f"{gptqmodel_version}+{get_version_tag()}"

include_dirs = ["gptqmodel_cuda"]
extensions = []
additional_setup_kwargs = {}

# ---------------------------
# Build CUDA extensions (optional, torch only if present)
# ---------------------------
if BUILD_CUDA_EXT == "1":
    # We need torch headers for PyTorch extensions. Only proceed if torch import works.
    if torch_mod is None:
        if FORCE_BUILD:
            sys.exit(
                "FORCE_BUILD is set but PyTorch is not importable. "
                "Install torch build deps first (see https://pytorch.org/) "
                "or unset GPTQMODEL_FORCE_BUILD to use prebuilt wheels."
            )
        # No torch -> we will still register bdist_wheel to attempt prebuilt fetch
    else:
        from distutils.sysconfig import get_python_lib

        from torch.utils import cpp_extension as cpp_ext  # noqa: F401

        conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
        if os.path.isdir(conda_cuda_include_dir):
            include_dirs.append(conda_cuda_include_dir)
            print(f"appending conda cuda include dir {conda_cuda_include_dir}")

        extra_link_args = []
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17", "-fopenmp", "-lgomp", "-DENABLE_BF16"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ],
        }

        CXX11_ABI = _detect_cxx11_abi()
        extra_compile_args["cxx"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]
        extra_compile_args["nvcc"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]

        if not ROCM_VERSION:
            extra_compile_args["nvcc"] += [
                "--threads", "8",
                "--optimize=3",
                "-lineinfo",
                "--resource-usage",
                "-Xfatbin", "-compress-all",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-diag-suppress=179,39",
            ]
        else:
            # hipify CUDA-like flags
            def _hipify_compile_flags(flags):
                modified_flags = []
                for flag in flags:
                    if flag.startswith("-") and "CUDA" in flag and not flag.startswith("-I"):
                        parts = flag.split("=", 1)
                        if len(parts) == 2:
                            flag_part, value_part = parts
                            modified_flag_part = flag_part.replace("CUDA", "HIP", 1)
                            modified_flags.append(f"{modified_flag_part}={value_part}")
                        else:
                            modified_flags.append(flag.replace("CUDA", "HIP", 1))
                    else:
                        modified_flags.append(flag)
                return modified_flags
            extra_compile_args["nvcc"] = _hipify_compile_flags(extra_compile_args["nvcc"])

        # Extensions (same as before, gated by ROCm / sm_80 availability)
        if sys.platform != "win32":
            if not ROCM_VERSION and HAS_CUDA_V8:
                extensions += [
                    cpp_ext.CUDAExtension(
                        "gptqmodel_marlin_kernels",
                        [
                            "gptqmodel_ext/marlin/marlin_cuda.cpp",
                            "gptqmodel_ext/marlin/marlin_cuda_kernel.cu",
                            "gptqmodel_ext/marlin/marlin_repack.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    ),
                    cpp_ext.CUDAExtension(
                        "gptqmodel_qqq_kernels",
                        [
                            "gptqmodel_ext/qqq/qqq.cpp",
                            "gptqmodel_ext/qqq/qqq_gemm.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    ),
                    cpp_ext.CUDAExtension(
                        "gptqmodel_exllama_eora",
                        [
                            "gptqmodel_ext/exllama_eora/eora/q_gemm.cu",
                            "gptqmodel_ext/exllama_eora/eora/pybind.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    ),
                    cpp_ext.CUDAExtension(
                        "gptqmodel_exllamav2_kernels",
                        [
                            "gptqmodel_ext/exllamav2/ext.cpp",
                            "gptqmodel_ext/exllamav2/cuda/q_matrix.cu",
                            "gptqmodel_ext/exllamav2/cuda/q_gemm.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    ),
                ]

            # both CUDA and ROCm compatible
            extensions += [
                cpp_ext.CUDAExtension(
                    "gptqmodel_exllama_kernels",
                    [
                        "gptqmodel_ext/exllama/exllama_ext.cpp",
                        "gptqmodel_ext/exllama/cuda_buffers.cu",
                        "gptqmodel_ext/exllama/cuda_func/column_remap.cu",
                        "gptqmodel_ext/exllama/cuda_func/q4_matmul.cu",
                        "gptqmodel_ext/exllama/cuda_func/q4_matrix.cu",
                    ],
                    extra_link_args=extra_link_args,
                    extra_compile_args=extra_compile_args,
                ),
            ]

        additional_setup_kwargs = {
            "ext_modules": extensions,
            "cmdclass": {"build_ext": cpp_ext.BuildExtension},
        }

# ---------------------------
# Cached wheel fetcher
# ---------------------------

class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        # Do not import torch here; allow explicit override via env
        xpu_avail = _bool_env("XPU_AVAILABLE", False)
        if FORCE_BUILD or xpu_avail:
            return super().run()

        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        wheel_filename = (
            f"{common_setup_kwargs['name']}-{gptqmodel_version}-"
            f"{python_version}-{python_version}-linux_x86_64.whl"
        )

        wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{version_vars['version']}", wheel_name=wheel_filename)
        print(f"Guessing wheel URL: {wheel_url}\nwheel name={wheel_filename}")

        try:
            import urllib.error  # noqa: F401
            import urllib.request
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = (
                f"{common_setup_kwargs['name']}-{gptqmodel_version}-{impl_tag}-{abi_tag}-{plat_tag}"
            )
            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except BaseException:
            print(f"Precompiled wheel not found at: {wheel_url}. Building from source...")
            super().run()

# ---------------------------
# Core metadata
# ---------------------------

common_setup_kwargs = {
    "version": gptqmodel_version,
    "name": "gptqmodel",
    "author": "ModelCloud",
    "author_email": "qubitium@modelcloud.ai",
    "description": "Production ready LLM model compression/quantization toolkit with hw accelerated inference support for both cpu/gpu via HF, vLLM, and SGLang.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/ModelCloud/GPTQModel",
    "project_urls": {"Homepage": "https://github.com/ModelCloud/GPTQModel"},
    "keywords": ["gptq", "quantization", "large-language-models", "transformers", "4bit", "llm"],
    "platforms": ["linux", "windows", "darwin"],
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
}

# ---------------------------
# setup()
# ---------------------------

setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "test": ["pytest>=8.2.2", "parameterized"],
        "quality": ["ruff==0.9.6", "isort==6.0.0"],
        "vllm": ["vllm>=0.8.5", "flashinfer-python>=0.2.1"],
        "sglang": ["sglang[srt]>=0.4.6", "flashinfer-python>=0.2.1"],
        "bitblas": ["bitblas==0.0.1-dev13"],
        "hf": ["optimum>=1.21.2"],
        "ipex": ["intel_extension_for_pytorch>=2.7.0"],
        "auto_round": ["auto_round>=0.3"],
        "logger": ["clearml", "random_word", "plotly"],
        "eval": ["lm_eval>=0.4.7", "evalplus>=0.3.1"],
        "triton": ["triton>=3.0.0"],
        "openai": ["uvicorn", "fastapi", "pydantic"],
        "mlx": ["mlx_lm>=0.24.0"],
    },
    include_dirs=include_dirs,
    python_requires=">=3.9.0",
    cmdclass=(
        {"bdist_wheel": CachedWheelsCommand, "build_ext": additional_setup_kwargs.get("cmdclass", {}).get("build_ext")}
        if BUILD_CUDA_EXT == "1" and additional_setup_kwargs
        else {"bdist_wheel": CachedWheelsCommand}
    ),
    ext_modules=additional_setup_kwargs.get("ext_modules", []),
    license="Apache-2.0",
    **common_setup_kwargs,
)
