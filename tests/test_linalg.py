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

# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import unittest  # noqa: E402
import torch # noqa: E402
from logbar import LogBar # noqa: E402
from parameterized import parameterized  # noqa: E402

log = LogBar.shared()

ROCM = torch.device("cuda:0") # fake cuda

# REQUIREMENT NOTES:
# `pip install logbar parameterized`
class Test(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float32, 2048),
            (torch.float64, 2048),
            (torch.float32, 8192),
            (torch.float64, 8192),
        ]
    )
    def test_linalg_eigh(self, dtype: torch.dtype, size: int):
        matrix = torch.randn([size, size], device=ROCM, dtype=dtype)
        torch.linalg.eigh(matrix)

    @parameterized.expand(
        [
            (torch.float32, 2048),
            (torch.float64, 2048),
            (torch.float32, 8192),
            (torch.float64, 8192),
        ]
    )
    def test_linalg_eigh_magma(self, dtype: torch.dtype, size: int):
        # force `magma` backend for linalg
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend="magma")

        matrix = torch.randn([size, size], device=ROCM, dtype=dtype)
        torch.linalg.eigh(matrix)

        torch.backends.cuda.preferred_linalg_library(backend=original_backend)