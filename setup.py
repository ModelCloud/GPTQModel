# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

# ---------------------------
# Helpers (no torch required)
# ---------------------------

def _read_env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default

def _probe_cmd(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None

def _bool_env(name, default=False):
    v = _read_env(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

def _detect_rocm_version():
    v = _read_env("ROCM_VERSION")
    if v:
        return v
    hip = _probe_cmd(["hipcc", "--version"])
    if hip:
        import re
        m = re.search(r"\b([0-9]+\.[0-9]+)\b", hip)
        if m:
            return m.group(1)
    try:
        p = Path("/opt/rocm/.info/version")
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None

def _detect_cuda_arch_list():
    """Return TORCH_CUDA_ARCH_LIST style string for the *installed* GPUs only.
    Priority:
      1) CUDA_ARCH_LIST env override (verbatim)
      2) nvidia-smi compute_cap (actual devices)
    """
    # 1) explicit override
    env_arch = _read_env("CUDA_ARCH_LIST")
    if env_arch:
        return env_arch

    # 2) actual devices present
    smi_out = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi_out:
        caps = []
        for line in smi_out.splitlines():
            cap = line.strip()
            if not cap:
                continue
            # normalize like '8.0'
            try:
                major, minor = cap.split(".", 1)
                caps.append(f"{int(major)}.{int(minor)}")
            except Exception:
                # some drivers return just '8' -> treat as '8.0'
                if cap.isdigit():
                    caps.append(f"{cap}.0")
        caps = sorted(set(caps), key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])))
        if caps:
            # PyTorch prefers ';' separators
            return ";".join(caps)

    # 3) conservative default for modern datacenter GPUs (A100 et al.)
    raise Exception("Could not get compute capability from nvidia-smi. Please check nvidia-utils package is installed.")

def _parse_arch_list(s: str):
    # Accept semicolons, commas, and any whitespace as separators.
    # Keep tokens like "8.0", "8.0+PTX" intact (we’ll strip suffixes later).
    return [tok for tok in re.split(r"[;\s,]+", s) if tok.strip()]

def _has_cuda_v8_from_arch_list(arch_list):
    try:
        vals = []
        for a in arch_list:
            # Handle things like "8.0+PTX"
            base = a.split("+", 1)[0]
            vals.append(float(base))
        return any(v >= 8.0 for v in vals)
    except Exception:
        return False

def _detect_cxx11_abi():
    v = _read_env("CXX11_ABI")
    if v in ("0", "1"):
        return int(v)
    return 1

def _torch_version_for_release():
    # No torch import; allow env override
    v = _read_env("TORCH_VERSION")
    if v:
        parts = v.split(".")
        return ".".join(parts[:2])
    else:
        raise Exception("TORCH_VERSION not passed for wheel generation.")
    return None

def _is_rocm_available():
    return _detect_rocm_version() is not None

# If you already have _probe_cmd elsewhere, you can delete this copy.
def _probe_cmd(args, timeout=6):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception:
        return ""

def _first_token_line(s: str) -> str | None:
    for line in (s or "").splitlines():
        t = line.strip()
        if t:
            return t
    return None

def _detect_torch_version() -> str | None:
    # 1) uv pip show torch
    out = _probe_cmd(["uv", "pip", "show", "torch"])
    if out:
        m = re.search(r"^Version:\s*([^\s]+)\s*$", out, flags=re.MULTILINE)
        if m:
            return m.group(1)

    # 2) pip show torch (both 'pip' and 'python -m pip')
    for cmd in (["pip", "show", "torch"], [sys.executable, "-m", "pip", "show", "torch"]):
        out = _probe_cmd(cmd)
        if out:
            m = re.search(r"^Version:\s*([^\s]+)\s*$", out, flags=re.MULTILINE)
            if m:
                return m.group(1)

    # 3) conda list torch
    out = _probe_cmd(["conda", "list", "torch"])
    if out:
        # Typical line starts with: torch  2.4.1  ...
        for line in out.splitlines():
            if line.strip().startswith("torch"):
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 2 and re.match(r"^\d+\.\d+(\.\d+)?", parts[1]):
                    return parts[1]

    # 4) Fallback: importlib.metadata (does not import torch package module)
    try:
        import importlib.metadata as im  # py3.8+
        version = im.version("torch")
        if not version:
            raise Exception("torch not found")
    except Exception:
        raise Exception("Unable to detect torch version via uv/pip/conda/importlib. Please install torch >= 2.7.1")

def _major_minor(v: str) -> str:
    parts = v.split(".")
    return ".".join(parts[:2]) if parts else v

def _detect_cuda_version() -> str | None:
    # Priority: env → nvidia-smi → nvcc
    v = os.environ.get("CUDA_VERSION")
    if v and v.strip():
        return v.strip()

    # nvidia-smi (modern drivers expose cuda_version)
    out = _probe_cmd(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])
    if out:
        line = _first_token_line(out)
        if line and re.match(r"^\d+\.\d+(\.\d+)?$", line):
            return line

    # nvcc --version (parse 'release X.Y')
    out = _probe_cmd(["nvcc", "--version"])
    if out:
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return f"{m.group(1)}.{m.group(2)}"

    return None

def get_version_tag() -> str:
    # TODO FIX ME: cpu wheels don't have torch version tags?
    if BUILD_CUDA_EXT != "1":
        return "cpu"

    # TODO FIX ME: rocm wheels don't have torch version tags?
    if ROCM_VERSION:
        return f"rocm{ROCM_VERSION}"

    if not CUDA_VERSION:
        raise Exception("Trying to compile GPTQModel for CUDA/ROCm, but no cuda or rocm version was detected.")

    torch_suffix = f"torch{_major_minor(TORCH_VERSION)}"

    CUDA_VERSION_COMPACT = "".join(CUDA_VERSION.split("."))
    base = f"cu{CUDA_VERSION_COMPACT[:3]}"
    return f"{base}{torch_suffix}"


# ---------------------------
# Env and versioning
# ---------------------------

TORCH_VERSION = _read_env("TORCH_VERSION")
RELEASE_MODE = _read_env("RELEASE_MODE")
CUDA_VERSION = _read_env("CUDA_VERSION")
ROCM_VERSION = _read_env("ROCM_VERSION")
TORCH_CUDA_ARCH_LIST = _read_env("TORCH_CUDA_ARCH_LIST")


# respect user env then detect
if not TORCH_VERSION:
    TORCH_VERSION = _detect_torch_version()
if not CUDA_VERSION:
    CUDA_VERSION = _detect_cuda_version()
if not ROCM_VERSION:
    ROCM_VERSION = _detect_rocm_version()

SKIP_ROCM_VERSION_CHECK = _read_env("SKIP_ROCM_VERSION_CHECK")
FORCE_BUILD = _bool_env("GPTQMODEL_FORCE_BUILD", False)

# BUILD_CUDA_EXT:
# - If user sets explicitly, respect it.
# - Otherwise auto: enable only if CUDA or ROCm detected.
BUILD_CUDA_EXT = _read_env("BUILD_CUDA_EXT")
if BUILD_CUDA_EXT is None:
    BUILD_CUDA_EXT = "1" if (CUDA_VERSION or ROCM_VERSION) else "0"

if ROCM_VERSION and not SKIP_ROCM_VERSION_CHECK:
    try:
        if float(ROCM_VERSION) < 6.2:
            sys.exit(
                "GPTQModel's compatibility with ROCm < 6.2 has not been verified. "
                "Set SKIP_ROCM_VERSION_CHECK=1 to proceed."
            )
    except Exception:
        pass

# Handle CUDA_ARCH_LIST (public) and set TORCH_CUDA_ARCH_LIST for build toolchains
CUDA_ARCH_LIST = _detect_cuda_arch_list() if (BUILD_CUDA_EXT == "1" and not ROCM_VERSION) else None

if not TORCH_CUDA_ARCH_LIST and CUDA_ARCH_LIST:
    archs = _parse_arch_list(CUDA_ARCH_LIST)
    kept = []
    for arch in archs:
        try:
            base = arch.split("+", 1)[0]
            if float(base) >= 6.0:
                kept.append(arch)
            else:
                print(f"we do not support this compute arch: {arch}, skipped.")
        except Exception:
            kept.append(arch)

    # Use semicolons for TORCH_CUDA_ARCH_LIST (PyTorch likes this),
    TORCH_CUDA_ARCH_LIST = ";".join(kept)
    os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST

    print(f"CUDA_ARCH_LIST: {CUDA_ARCH_LIST}")
    print(f"TORCH_CUDA_ARCH_LIST: {TORCH_CUDA_ARCH_LIST}")

version_vars = {}
exec("exec(open('gptqmodel/version.py').read()); version=__version__", {}, version_vars)
gptqmodel_version = version_vars["version"]

# -----------------------------
# Prebuilt wheel download config
# -----------------------------
# Default template (GitHub Releases), can be overridden via env.
DEFAULT_WHEEL_URL_TEMPLATE = "https://github.com/ModelCloud/GPTQModel/releases/download/{tag_name}/{wheel_name}"
WHEEL_URL_TEMPLATE = os.environ.get("GPTQMODEL_WHEEL_URL_TEMPLATE")
WHEEL_BASE_URL = os.environ.get("GPTQMODEL_WHEEL_BASE_URL")
WHEEL_TAG = os.environ.get("GPTQMODEL_WHEEL_TAG")  # Optional override of release tag

def _resolve_wheel_url(tag_name: str, wheel_name: str) -> str:
    """
    Build the final wheel URL based on:
      1) GPTQMODEL_WHEEL_URL_TEMPLATE (highest priority)
      2) GPTQMODEL_WHEEL_BASE_URL (append /{wheel_name})
      3) DEFAULT_WHEEL_URL_TEMPLATE (GitHub Releases)
    """
    # Highest priority: explicit template
    if WHEEL_URL_TEMPLATE:
        tmpl = WHEEL_URL_TEMPLATE
        # If {wheel_name} or {tag_name} not present, treat as base and append name.
        if ("{wheel_name}" in tmpl) or ("{tag_name}" in tmpl):
            return tmpl.format(tag_name=tag_name, wheel_name=wheel_name)
        # Otherwise, join as base
        if tmpl.endswith("/"):
            return tmpl + wheel_name
        return tmpl + "/" + wheel_name

    # Next priority: base URL
    if WHEEL_BASE_URL:
        base = WHEEL_BASE_URL
        if base.endswith("/"):
            return base + wheel_name
        return base + "/" + wheel_name

    # Fallback: default GitHub template
    return DEFAULT_WHEEL_URL_TEMPLATE.format(tag_name=tag_name, wheel_name=wheel_name)

def get_version_for_release() -> str:
    # TODO FIX ME: cpu wheels don't have torch version tags?
    if BUILD_CUDA_EXT != "1":
        return "cpu"

    # TODO FIX ME: rocm wheels don't have torch version tags?
    if ROCM_VERSION:
        return f"rocm{'.'.join(str(ROCM_VERSION).split('.')[:2])}"

    if not CUDA_VERSION:
        print(
            "Trying to compile GPTQModel for CUDA, but no CUDA version was detected. "
            "Set CUDA_VERSION env (e.g. 12.1)."
        )
        sys.exit(1)

    # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
    return f"cu{CUDA_VERSION[:3]}torch{_major_minor(TORCH_VERSION)}"

requirements = []
if not os.getenv("CI"):
    with open("requirements.txt", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip()]

# Decide HAS_CUDA_V8 without torch
HAS_CUDA_V8 = False
if CUDA_ARCH_LIST:
    HAS_CUDA_V8 = not ROCM_VERSION and _has_cuda_v8_from_arch_list(_parse_arch_list(CUDA_ARCH_LIST))
else:
    smi = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi:
        try:
            caps = [float(x.strip()) for x in smi.splitlines() if x.strip()]
            HAS_CUDA_V8 = any(cap >= 8.0 for cap in caps)
        except Exception:
            HAS_CUDA_V8 = False

if RELEASE_MODE == "1":
    gptqmodel_version = f"{gptqmodel_version}+{get_version_for_release()}"

include_dirs = ["gptqmodel_cuda"]

extensions = []
additional_setup_kwargs = {}

# ---------------------------
# Build CUDA/ROCm extensions (only when enabled)
# ---------------------------
# -----------------------------
# Per-extension build toggles
# -----------------------------
def _env_enabled(val: str) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() not in ("0", "false", "off", "no")

def _env_enabled_any(names, default="1") -> bool:
    for n in names:
        if n in os.environ:
            return _env_enabled(os.environ.get(n))
    return _env_enabled(default)

BUILD_MARLIN      = _env_enabled_any(os.environ.get("GPTQMODEL_BUILD_MARLIN", "1"))
BUILD_EXLLAMA_V2  = _env_enabled(os.environ.get("GPTQMODEL_BUILD_EXLLAMA_V2", "1"))

# Optional kernels and not build by default. Enable compile with env flags
BUILD_QQQ         = _env_enabled(os.environ.get("GPTQMODEL_BUILD_QQQ", "0"))
BUILD_EORA        = _env_enabled(os.environ.get("GPTQMODEL_BUILD_EORA", "0"))
BUILD_EXLLAMA_V1  = _env_enabled(os.environ.get("GPTQMODEL_BUILD_EXLLAMA_V1", "0"))

if BUILD_CUDA_EXT == "1":
    # Import torch's cpp_extension only if we're truly building GPU extensions
    try:
        from distutils.sysconfig import get_python_lib

        from torch.utils import cpp_extension as cpp_ext  # type: ignore
    except Exception:
        if FORCE_BUILD:
            sys.exit(
                "FORCE_BUILD is set but PyTorch C++ extension headers are unavailable. "
                "Install torch build deps first (see https://pytorch.org/) or unset GPTQMODEL_FORCE_BUILD."
            )
        # If we can't import cpp_extension, fall back to prebuilt wheel path
        cpp_ext = None

    if cpp_ext is not None:
        # Optional conda CUDA runtime headers
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

        # Windows/OpenMP note: adjust flags as needed for MSVC if you add native Windows wheels
        if sys.platform == "win32":
            extra_compile_args["cxx"] = ["/O2", "/std:c++17", "/openmp", "/DNDEBUG", "/DENABLE_BF16"]

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
                "-diag-suppress=179,39,177",
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

        # Extensions (gate marlin/qqq/eora/exllamav2 on CUDA sm_80+ and non-ROCm)
        if sys.platform != "win32":
            if not ROCM_VERSION and HAS_CUDA_V8:
                if BUILD_MARLIN:
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
                        )
                    ]

                if BUILD_QQQ:
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_qqq_kernels",
                            [
                                "gptqmodel_ext/qqq/qqq.cpp",
                                "gptqmodel_ext/qqq/qqq_gemm.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]

                if BUILD_EORA:
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_exllama_eora",
                            [
                                "gptqmodel_ext/exllama_eora/eora/q_gemm.cu",
                                "gptqmodel_ext/exllama_eora/eora/pybind.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]
                if BUILD_EXLLAMA_V2:
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_exllamav2_kernels",
                            [
                                "gptqmodel_ext/exllamav2/ext.cpp",
                                "gptqmodel_ext/exllamav2/cuda/q_matrix.cu",
                                "gptqmodel_ext/exllamav2/cuda/q_gemm.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]

            # both CUDA and ROCm compatible
            if BUILD_EXLLAMA_V1:
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
                    )
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
        # No implicit torch checks; allow explicit override via env
        xpu_avail = _bool_env("XPU_AVAILABLE", False)
        if FORCE_BUILD or xpu_avail:
            return super().run()

        system_name = platform.system().lower()

        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        wheel_filename = (
            f"{common_setup_kwargs['name']}-{gptqmodel_version}-"
            f"{python_version}-{python_version}-{system_name}_x86_64.whl"
        )

        wheel_filename = f"{common_setup_kwargs['name']}-{gptqmodel_version}+{get_version_tag()}-{python_version}-{python_version}-linux_x86_64.whl"

        # Allow tag override via env; default to "v{gptqmodel_version}"
        tag_name = WHEEL_TAG if WHEEL_TAG else f"v{gptqmodel_version}"
        wheel_url = _resolve_wheel_url(tag_name=tag_name, wheel_name=wheel_filename)

        print(f"Resolved wheel URL: {wheel_url}\nwheel name={wheel_filename}")

        try:
            import urllib.request as req
            req.urlretrieve(wheel_url, wheel_filename)

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
print(f"CUDA {CUDA_ARCH_LIST}")
print(f"HAS_CUDA_V8 {HAS_CUDA_V8}")
print(f"SETUP_KWARGS {additional_setup_kwargs}")

setup(
    packages=find_packages(),
    # setup_requires=["setuptools>=80.9.0", "torch>=2.7.1"],
    install_requires=requirements,
    extras_require={
        "test": ["pytest>=8.2.2", "parameterized"],
        "quality": ["ruff==0.13.0", "isort==6.0.1"],
        "vllm": ["vllm>=0.8.5", "flashinfer-python>=0.2.1"],
        "sglang": ["sglang[srt]>=0.4.6", "flashinfer-python>=0.2.1"],
        "bitblas": ["bitblas==0.0.1-dev13"],
        "hf": ["optimum>=1.21.2"],
        # @deprecation after torch 2.9 is released
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
        if (BUILD_CUDA_EXT == "1" and additional_setup_kwargs)
        else {"bdist_wheel": CachedWheelsCommand}
    ),
    ext_modules=additional_setup_kwargs.get("ext_modules", []),
    license="Apache-2.0",
    **common_setup_kwargs,
)
