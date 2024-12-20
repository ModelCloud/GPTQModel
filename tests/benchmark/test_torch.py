from benchmark_test import BenchmarkTest
from gptqmodel import BACKEND
from parameterized import parameterized  # noqa: E402


class TestTorch(BenchmarkTest):
    TOKENS_PER_SECOND = 49.20
    INFERENCE_BACKEND = BACKEND.TORCH

    @parameterized.expand(
        [
            'cuda'
        ]
    )
    def test_torch(self, device):
        self.DEVICE = device
        self.benchmark()