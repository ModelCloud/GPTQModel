from benchmark_test import BenchmarkTest
from gptqmodel import BACKEND
from parameterized import parameterized  # noqa: E402


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
            self.skipTest(f"MacOS env skip")
        self.benchmark(backend=backend, device=device, tokens_per_second=tokens_per_second)