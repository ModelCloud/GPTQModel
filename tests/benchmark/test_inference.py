from benchmark_test import BenchmarkTest
from gptqmodel import BACKEND
from parameterized import parameterized  # noqa: E402


class TestInference(BenchmarkTest):
    @parameterized.expand(
        [
            (BACKEND.TORCH, 'cuda', 49.20),
            # (BACKEND.TORCH, 'cpu', 49.20),
            # (BACKEND.TORCH, 'xpu', 49.20),
            # (BACKEND.TORCH, 'mps', 49.20),
        ]
    )
    def test_inference(self, backend, device, tokens_per_second):
        self.benchmark(backend=backend, device=device, tokens_per_second=tokens_per_second)