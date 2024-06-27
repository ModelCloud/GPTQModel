import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

version_vars = {}
exec("exec(open('gptqmodel/version.py').read()); version=__version__", {}, version_vars)
gptqmodel_version = version_vars['version']


common_setup_kwargs = {
    "version": gptqmodel_version,
    "name": "gptqmodel",
    "author": "ModelCloud",
    "description": "A LLM quantization package with user-friendly apis. Based on GPTQ algorithm.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/ModelCloud/GPTQModel",
    "keywords": ["gptq", "quantization", "large-language-models", "transformers", "4bit", "llm"],
    "platforms": ["linux"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "Environment :: GPU :: NVIDIA CUDA :: 12.1",
        "Environment :: GPU :: NVIDIA CUDA :: 12.2",
        "Environment :: GPU :: NVIDIA CUDA :: 12.3",
        "Environment :: GPU :: NVIDIA CUDA :: 12.4",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
}


PYPI_RELEASE = os.environ.get("PYPI_RELEASE", None)
BUILD_CUDA_EXT = True
COMPILE_MARLIN = True
UNSUPPORTED_COMPUTE_CAPABILITIES = ["3.5", "3.7", "5.0", "5.2", "5.3"]

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
    if not PYPI_RELEASE:
        common_setup_kwargs["version"] += f"+cu{CUDA_VERSION}"


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extras_require = {
    "test": ["pytest>=8.2.2", "parameterized"],
    "quality": ["ruff==0.4.9", "isort==5.13.2"],
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
            "-Wno-switch-bool",
        ],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--threads",
            "2",
            "-Xfatbin",
            "-compress-all",
        ],
    }

    extensions = [
        cpp_ext.CUDAExtension(
            "gptqmodel_cuda_64",
            [
                "gptqmodel_ext/cuda_64/gptqmodel_cuda_64.cpp",
                "gptqmodel_ext/cuda_64/gptqmodel_cuda_kernel_64.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
        cpp_ext.CUDAExtension(
            "gptqmodel_cuda_256",
            [
                "gptqmodel_ext/cuda_256/gptqmodel_cuda_256.cpp",
                "gptqmodel_ext/cuda_256/gptqmodel_cuda_kernel_256.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ]

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
    python_requires=">=3.8.0",
    **common_setup_kwargs,
)
