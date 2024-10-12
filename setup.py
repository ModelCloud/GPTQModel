import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

os.environ["BUILD_CUDA_EXT"] = "1"

TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST")

version_vars = {}
exec("exec(open('gptqmodel/version.py').read()); version=__version__", {}, version_vars)
gptqmodel_version = version_vars['version']

common_setup_kwargs = {
    "version": gptqmodel_version,
    "name": "gptqmodel",
    "author": "ModelCloud",
    "author_email":"qubitium@modelcloud.ai",
    "description": "A LLM quantization package with user-friendly apis. Based on GPTQ algorithm.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/ModelCloud/GPTQModel",
    "keywords": ["gptq", "quantization", "large-language-models", "transformers", "4bit", "llm"],
    "platforms": ["linux"],
    "classifiers": [
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
}

CUDA_RELEASE = os.environ.get("CUDA_RELEASE", None)
BUILD_CUDA_EXT = True
COMPILE_MARLIN = True

if BUILD_CUDA_EXT:
    import torch

    default_cuda_version = torch.version.cuda
    CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", default_cuda_version).split("."))

    if not CUDA_VERSION:
        print(
            f"Trying to compile GPTQModel for CUDA, but Pytorch {torch.__version__} "
            "is installed without CUDA support."
        )
        sys.exit(1)

    # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
    if CUDA_RELEASE == "1":
        common_setup_kwargs["version"] += f"+cu{CUDA_VERSION[:3]}torch{'.'.join(torch.version.__version__.split('.')[:2])}"

with open('requirements.txt') as f:
    requirement_list = f.read().splitlines()
    if os.getenv("CI"):
        requirements = []
    else:
        requirements = requirement_list
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import torch  # noqa: E402

if TORCH_CUDA_ARCH_LIST is None:
    at_least_one_cuda_v6 = any(torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))
else:
    at_least_one_cuda_v6 = True

if not at_least_one_cuda_v6:
    raise EnvironmentError(
        "GPTQModel requires at least one GPU device with CUDA compute capability >= `6.0`."
    )

extras_require = {
    "test": ["pytest>=8.2.2", "parameterized"],
    "quality": ["ruff==0.4.9", "isort==5.13.2"],
    'vllm': ["vllm>=0.6.2", "flashinfer==0.1.6"],
    'sglang': ["sglang>=0.3.2", "flashinfer==0.1.6"],
    'bitblas': ["bitblas>=0.0.1.dev13"],
    'hf': ["optimum>=1.21.2"],
    'qbits': ["intel_extension_for_transformers>=1.4.2"],
}

include_dirs = ["gptqmodel_cuda"]

additional_setup_kwargs = {}
if BUILD_CUDA_EXT:
    from distutils.sysconfig import get_python_lib

    from torch.utils import cpp_extension as cpp_ext

    conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")

    print("conda_cuda_include_dir", conda_cuda_include_dir)
    if os.path.isdir(conda_cuda_include_dir):
        include_dirs.append(conda_cuda_include_dir)
        print(f"appending conda cuda include dir {conda_cuda_include_dir}")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17", 
            "-fopenmp", 
            "-lgomp", 
            "-DENABLE_BF16"
            "-Wno-switch-bool",
        ],
        "cxx": ["-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-DENABLE_BF16",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--threads",
            "4",
            "-Xfatbin",
            "-compress-all",
            "-diag-suppress=179,39,186",
            "--use_fast_math",
        ],
    }

    extensions = []

    # Marlin is not ROCm-compatible, CUDA only
    if COMPILE_MARLIN:
        extensions.append(
            cpp_ext.CUDAExtension(
                "gptqmodel_marlin_cuda",
                [
                    "gptqmodel_ext/marlin/marlin_cuda.cpp",
                    "gptqmodel_ext/marlin/marlin_cuda_kernel.cu",
                    "gptqmodel_ext/marlin/marlin_repack.cu",
                ],
                extra_compile_args=extra_compile_args,
            )
        )

        extensions.append(
            cpp_ext.CUDAExtension(
                "gptqmodel_marlin_cuda_inference",
                [
                    "gptqmodel_ext/marlin_inference/marlin_cuda.cpp",
                    "gptqmodel_ext/marlin_inference/marlin_cuda_kernel.cu",
                    "gptqmodel_ext/marlin_inference/marlin_repack.cu",
                ],
                extra_compile_args=extra_compile_args,
            )
        )

    extensions.append(
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
    )
    extensions.append(
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
    )

    additional_setup_kwargs = {"ext_modules": extensions, "cmdclass": {"build_ext": cpp_ext.BuildExtension}}

common_setup_kwargs.update(additional_setup_kwargs)
setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    include_dirs=include_dirs,
    python_requires=">=3.9.0",
    **common_setup_kwargs,
)
