import os
import subprocess
import sys
import urllib
import urllib.error
import urllib.request
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

CUDA_RELEASE = os.environ.get("CUDA_RELEASE", None)

TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST")

version_vars = {}
exec("exec(open('gptqmodel/version.py').read()); version=__version__", {}, version_vars)
gptqmodel_version = version_vars['version']

BASE_WHEEL_URL = (
    "https://github.com/ModelCloud/GPTQModel/releases/download/{tag_name}/{wheel_name}"
)

BUILD_CUDA_EXT = True
COMPILE_MARLIN = True

if os.environ.get("GPTQMODEL_FORCE_BUILD", None):
    FORCE_BUILD = True
else:
    FORCE_BUILD = False

common_setup_kwargs = {
    "version": gptqmodel_version,
    "name": "gptqmodel",
    "author": "ModelCloud",
    "author_email": "qubitium@modelcloud.ai",
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


def get_version_tag(is_cuda_release: bool = True) -> str:
    import torch

    if not BUILD_CUDA_EXT:
        return common_setup_kwargs["version"]

    default_cuda_version = torch.version.cuda
    CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", default_cuda_version).split("."))

    if not CUDA_VERSION:
        print(
            f"Trying to compile GPTQModel for CUDA, but Pytorch {torch.__version__} "
            "is installed without CUDA support."
        )
        sys.exit(1)

    # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
    if is_cuda_release:
        return f"cu{CUDA_VERSION[:3]}torch{'.'.join(torch.version.__version__.split('.')[:2])}"

    return common_setup_kwargs["version"]


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
    if not at_least_one_cuda_v6:
        BUILD_CUDA_EXT = False

if BUILD_CUDA_EXT:
    if CUDA_RELEASE == "1":
        common_setup_kwargs["version"] += f"+{get_version_tag(True)}"
else:
    common_setup_kwargs["version"] += "+cpu"

additional_setup_kwargs = {}

include_dirs = ["gptqmodel_cuda"]

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

    extensions = [
        cpp_ext.CUDAExtension(
            "gptqmodel_cuda_64",
            [
                "gptqmodel_ext/cuda_64/gptqmodel_cuda_64.cpp",
                "gptqmodel_ext/cuda_64/gptqmodel_cuda_kernel_64.cu"
            ]
        ),
        cpp_ext.CUDAExtension(
            "gptqmodel_cuda_256",
            [
                "gptqmodel_ext/cuda_256/gptqmodel_cuda_256.cpp",
                "gptqmodel_ext/cuda_256/gptqmodel_cuda_kernel_256.cu"
            ]
        )
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


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        if FORCE_BUILD:
            return super().run()

        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

        wheel_filename = f"{common_setup_kwargs['name']}-{gptqmodel_version}+{get_version_tag()}-{python_version}-{python_version}-linux_x86_64.whl"

        wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{gptqmodel_version}", wheel_name=wheel_filename)
        print(f"Guessing wheel URL: {wheel_url}\nwheel name={wheel_filename}")

        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{common_setup_kwargs['name']}-{gptqmodel_version}+{get_version_tag()}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)

            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "test": ["pytest>=8.2.2", "parameterized"],
        "quality": ["ruff==0.4.9", "isort==5.13.2"],
        'vllm': ["vllm>=0.6.2", "flashinfer==0.1.6"],
        'sglang': ["sglang>=0.3.2", "flashinfer==0.1.6"],
        'bitblas': ["bitblas==0.0.1.dev13"],
        'hf': ["optimum>=1.21.2"],
        'ipex': ["intel_extension_for_pytorch>=2.5.0"],
        'auto_round': ["auto_round>=0.3"],
        'logger': ["clearml", "random_word", "device-smi", "plotly"],
        'eval': ["lm_eval>=0.4.4"],
    },
    include_dirs=include_dirs,
    python_requires=">=3.9.0",
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": cpp_ext.BuildExtension}
    if BUILD_CUDA_EXT
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    ext_modules=extensions,
    **common_setup_kwargs
)
