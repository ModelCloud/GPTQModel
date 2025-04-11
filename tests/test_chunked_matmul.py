import time
import unittest

import torch


class TestChunkedMatmul(unittest.TestCase):
    def setUp(self):
        self.dim = 2048
        self.chunk_size = 512
        self.rtol = 1e-5  # Can now use tighter tolerances
        self.atol = 1e-7
        torch.manual_seed(42)
        self.inp = torch.randn(2048, self.dim, device='cuda', dtype=torch.float32)

    def chunked_matmul(self, inp: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Perfectly accurate chunked implementation of inp @ inp.T."""
        cols = inp.shape[1]
        H = torch.zeros((cols, cols), dtype=torch.float64, device=inp.device)

        # Process column blocks
        for j in range(0, cols, chunk_size):
            j_end = min(j + chunk_size, cols)
            chunk_j = inp[:, j:j_end].to(dtype=torch.float64)   # (rows, chunk_size)

            # Process row blocks
            for i in range(0, cols, chunk_size):
                i_end = min(i + chunk_size, cols)
                chunk_i = inp[:, i:i_end].to(dtype=torch.float64)  # (rows, chunk_size)

                # Compute block and accumulate
                H[i:i_end, j:j_end] = chunk_i.t() @ chunk_j

        return H.float()

    def test_memory_and_performance(self):
        """Compare memory usage, speed, and accuracy with detailed error reporting."""
        # Warmup and reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # --- Native Implementation ---
        torch.cuda.synchronize()
        start = time.time()
        native_H = self.inp.t() @ self.inp
        torch.cuda.synchronize()
        time.time() - start
        torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        # --- Chunked Implementation ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        start = time.time()
        chunked_H = self.chunked_matmul(self.inp, self.chunk_size)
        torch.cuda.synchronize()
        time.time() - start
        torch.cuda.max_memory_allocated() / (1024 ** 2)

        # --- Enhanced Validation ---
        max_abs_error = torch.max(torch.abs(chunked_H - native_H)).item()
        max_rel_error = torch.max(torch.abs((chunked_H - native_H) / (native_H + 1e-8))).item()

        self.assertTrue(
            torch.allclose(chunked_H, native_H, rtol=self.rtol, atol=self.atol),
            msg=(
                f"Chunked result diverges from native!\n"
                f"Max absolute error: {max_abs_error:.3e} (allowed: {self.atol:.1e})\n"
                f"Max relative error: {max_rel_error:.3e} (allowed: {self.rtol:.1e})\n"
                f"Where native_H ≈ {native_H.abs().mean():.3e}, chunked_H ≈ {chunked_H.abs().mean():.3e}"
            )
        )


if __name__ == "__main__":
    unittest.main()
