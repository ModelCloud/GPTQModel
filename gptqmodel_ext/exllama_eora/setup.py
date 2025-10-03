# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='eora',
    version='0.1.0',
    author='Maksim Khadkevich',
    author_email='mkhadkevich@nvidia.com',
    description='Highly optimized EORA CUDA matmul kernel for 4 bit GPTQ inference.',
    install_requires=['torch'],
    packages=['eora'],
    ext_modules=[
        cpp_extension.CUDAExtension(
            'gptqmodel_exllama_eora',
            [
                "eora/q_gemm.cu",
                "eora/pybind.cu",
            ],
            include_dirs=[os.path.abspath("."), os.path.abspath("eora")],
            extra_compile_args={
                'cxx': ['-std=c++20'], 
                'nvcc': ['-std=c++20'],
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
