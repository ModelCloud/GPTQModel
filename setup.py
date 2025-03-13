# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import urllib
import urllib.error
import urllib.request
from pathlib import Path

import torch
from setuptools import find_packages, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except BaseException:
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except BaseException:
        sys.exit("Both latest setuptools and wheel package are not found.  Please upgrade to latest setuptools: `pip install -U setuptools`")

RELEASE_MODE = os.environ.get("RELEASE_MODE", None)

TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST")

ROCM_VERSION = os.environ.get('ROCM_VERSION', None)
SKIP_ROCM_VERSION_CHECK = os.environ.get('SKIP_ROCM_VERSION_CHECK', None)

if ROCM_VERSION is None and torch.version.hip:
    ROCM_VERSION = ".".join(torch.version.hip.split(".")[:2]) # print(torch.version.hip) -> 6.3.42131-fa1d09cbd
    os.environ["ROCM_VERSION"] = ROCM_VERSION

if ROCM_VERSION is not None and float(ROCM_VERSION) < 6.2 and not SKIP_ROCM_VERSION_CHECK:
    sys.exit(
        "GPTQModel's compatibility with ROCM versions below 6.2 has not been verified. If you wish to proceed, please set the SKIP_ROCM_VERSION_CHECK environment."
    )

if TORCH_CUDA_ARCH_LIST:
    arch_list = " ".join([arch for arch in TORCH_CUDA_ARCH_LIST.split() if float(arch.split('+')[0]) >= 6.0 or print(f"we do not support this compute arch: {arch}, skipped.") is not None])
    if arch_list != TORCH_CUDA_ARCH_LIST:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
        print(f"TORCH_CUDA_ARCH_LIST has been updated to '{arch_list}'")

version_vars = {}
exec("exec(open('gptqmodel/version.py').read()); version=__version__", {}, version_vars)
gptqmodel_version = version_vars['version']

BASE_WHEEL_URL = (
    "https://github.com/ModelCloud/GPTQModel/releases/download/{tag_name}/{wheel_name}"
)

BUILD_CUDA_EXT = sys.platform != "darwin"

if os.environ.get("GPTQMODEL_FORCE_BUILD", None):
    FORCE_BUILD = True
else:
    FORCE_BUILD = False

extensions = []
common_setup_kwargs = {
    "version": gptqmodel_version,
    "name": "gptqmodel",
    "author": "ModelCloud",
    "author_email": "qubitium@modelcloud.ai",
    "description": "Production ready LLM model compression/quantization toolkit with hw accelerated inference support for both cpu/gpu via HF, vLLM, and SGLang.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/ModelCloud/GPTQModel",
    "project_urls": {
        "Homepage": "https://github.com/ModelCloud/GPTQModel",
    },
    "keywords": ["gptq", "quantization", "large-language-models", "transformers", "4bit", "llm"],
    "platforms": ["linux", "windows", "darwin"],
    "classifiers": [
        "License :: OSI Approved :: Apache Software License",
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

def get_version_tag() -> str:
    if not BUILD_CUDA_EXT:
        return "cpu"

    if ROCM_VERSION:
        return f"rocm{ROCM_VERSION}"

    cuda_version = os.environ.get("CUDA_VERSION", torch.version.cuda)
    if not cuda_version or not cuda_version.split("."):
        print(
            f"Trying to compile GPTQModel for CUDA, but Pytorch {torch.__version__} "
            "is installed without CUDA support."
        )
        sys.exit(1)


    CUDA_VERSION = "".join(cuda_version.split("."))

    # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
    return f"cu{CUDA_VERSION[:3]}torch{'.'.join(torch.version.__version__.split('.')[:2])}"

requirements = []
if not os.getenv("CI"):
    with open('requirements.txt') as f:
        requirements = [line.strip() for line in f if line.strip()]


if TORCH_CUDA_ARCH_LIST is None:
    HAS_CUDA_V8 = any(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count()))

    got_cuda_v6 = any(torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))
    got_cuda_between_v6_and_v8 = any(6 <= torch.cuda.get_device_capability(i)[0] < 8 for i in range(torch.cuda.device_count()))

    # not validated for compute < 6
    if not got_cuda_v6 and not torch.version.hip:
        BUILD_CUDA_EXT = False

        if sys.platform == "win32" and 'cu+' not in torch.__version__:
            print("No CUDA device detected: avoid installing torch from PyPi which may not have bundle CUDA support for Windows.\nInstall via PyTorch: `https://pytorch.org/get-started/locally/`")

    # if cuda compute is < 8.0, always force build since we only compile cached wheels for >= 8.0
    if BUILD_CUDA_EXT and not FORCE_BUILD:
        if got_cuda_between_v6_and_v8:
            FORCE_BUILD = True
else:
    HAS_CUDA_V8 = not ROCM_VERSION and len([arch for arch in TORCH_CUDA_ARCH_LIST.split() if float(arch.split('+')[0]) >= 8]) > 0

if RELEASE_MODE == "1":
    common_setup_kwargs["version"] += f"+{get_version_tag()}"

additional_setup_kwargs = {}

include_dirs = ["gptqmodel_cuda"]

extensions = []

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
            "-DENABLE_BF16",
        ],
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

    # torch >= 2.6.0 may require extensions to be build with CX11_ABI=1
    CXX11_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

    extra_compile_args["cxx"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]
    extra_compile_args["nvcc"] += [ f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}" ]

    # nvidia (nvcc) only compile flags that rocm doesn't support
    if not ROCM_VERSION:
        extra_compile_args["nvcc"] += [
            "--threads", "8",
            "--optimize=3",
            "-lineinfo",
            "--resource-usage",
            "-Xfatbin",
            "-compress-all",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-diag-suppress=179,39",  # 186
        ]

    extensions = []

    # TODO: VC++: error lnk2001 unresolved external symbol cublasHgemm
    if sys.platform != "win32":# TODO: VC++: fatal error C1061: compiler limit : blocks nested too deeply
        # https://rocm.docs.amd.com/projects/HIPIFY/en/docs-6.1.0/tables/CUDA_Device_API_supported_by_HIP.html
        # nv_bfloat16 and nv_bfloat162 (2x bf16) missing replacement in ROCm
        if not ROCM_VERSION:
            if HAS_CUDA_V8:
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
                            "gptqmodel_ext/qqq/qqq_gemm.cu"
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    ),
                    cpp_ext.CUDAExtension(
                        'gptqmodel_exllama_eora',
                        [
                            "gptqmodel_ext/exllama_eora/eora/q_gemm.cu",
                            "gptqmodel_ext/exllama_eora/eora/pybind.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    ),
                    # TODO: VC++: error lnk2001 unresolved external symbol cublasHgemm
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

        # both cuda and rocm compatible
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

    additional_setup_kwargs = {"ext_modules": extensions, "cmdclass": {"build_ext": cpp_ext.BuildExtension}}

class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        if FORCE_BUILD or torch.xpu.is_available():
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
        except BaseException:
            print(f"Precompiled wheel not found in url: {wheel_url}. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "test": ["pytest>=8.2.2", "parameterized"],
        "quality": ["ruff==0.9.6", "isort==6.0.0"],
        'vllm': ["vllm>=0.7.3",  "flashinfer-python>=0.2.1"],
        'sglang': ["sglang[srt]>=0.3.2",  "flashinfer-python>=0.2.1"],
        'bitblas': ["bitblas==0.0.1-dev13"],
        'hf': ["optimum>=1.21.2"],
        'ipex': ["intel_extension_for_pytorch>=2.6.0"],
        'auto_round': ["auto_round>=0.3"],
        'logger': ["clearml", "random_word", "plotly"],
        'eval': ["lm_eval>=0.4.7", "evalplus>=0.3.1"],
        'triton': ["triton>=2.0.0"],
        'openai': ["uvicorn", "fastapi", "pydantic"],
        'mlx': ["mlx_lm>=0.20.6"]
    },
    include_dirs=include_dirs,
    python_requires=">=3.9.0",
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": cpp_ext.BuildExtension}
    if BUILD_CUDA_EXT
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    ext_modules=extensions,
    license="Apache 2.0",
    **common_setup_kwargs
)
