# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from benchmark_test import BenchmarkTest
from parameterized import parameterized  # noqa: E402

from gptqmodel import BACKEND


class TestInference(BenchmarkTest):
    @parameterized.expand(
        [
            (BACKEND.TORCH, 'cuda', 210),
            # (BACKEND.TORCH, 'cpu', 5.50),
            # (BACKEND.TORCH, 'xpu', 58.20),
            # (BACKEND.TORCH, 'mps', 3.40),
        ]
    )
    def test_inference(self, backend, device, tokens_per_second):
        if device == 'mps':
            self.skipTest("MacOS env skip")
        self.benchmark(backend=backend, device=device, tokens_per_second=tokens_per_second)
