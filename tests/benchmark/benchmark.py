# Copyright 2025 ModelCloud
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

from benchmark_test import BenchmarkTest
from parameterized import parameterized  # noqa: E402

from gptqmodel import BACKEND


class TestInference(BenchmarkTest):
    @parameterized.expand(
        [
            (BACKEND.TORCH, 'cuda', 292.50),
            (BACKEND.TORCH, 'cpu', 5.50),
            (BACKEND.TORCH, 'xpu', 58.20),
            (BACKEND.TORCH, 'mps', 3.40),
        ]
    )
    def test_inference(self, backend, device, tokens_per_second):
        if device == 'mps':
            self.skipTest("MacOS env skip")
        self.benchmark(backend=backend, device=device, tokens_per_second=tokens_per_second)
