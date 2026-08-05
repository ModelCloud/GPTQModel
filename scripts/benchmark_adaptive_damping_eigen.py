# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark adaptive-damping eigenvalue estimators for speed and accuracy.

This benchmarks the estimators used by adaptive Hessian damping under a variety
of realistic ill-conditioned SPD spectrum shapes:

    random          - eigenvalues spread uniformly in log-space
    middle_skewed   - eigenvalues concentrated in the middle of the spectrum
    left_skewed     - many small eigenvalues, few large
    right_skewed    - many large eigenvalues, few small
    semi_left       - most eigenvalues in the lower half
    semi_right      - most eigenvalues in the upper half
    left_and_right  - bimodal: clusters of small and large eigenvalues
    combo           - random mix of left-skewed and right-skewed samples

Run with the target GPU visible, e.g.:
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark_adaptive_damping_eigen.py
"""

import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch


def eigenvalues_for_distribution(
    d: int,
    cond: float,
    distribution: str,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Return a positive eigenvalue vector with the requested distribution.

    The returned vector is sorted ascending and scaled so the largest value is
    ``1.0`` and the smallest is ``1 / cond``.
    """

    rng = np.random.default_rng(seed)

    def beta(shape, alpha, beta):
        return torch.from_numpy(rng.beta(alpha, beta, size=shape).astype(np.float32)).to(device)

    # Draw normalized positions on [0, 1] for the log-spaced eigenvalues.
    if distribution == "random":
        alpha, beta_param = 1.0, 1.0
    elif distribution == "middle_skewed":
        alpha, beta_param = 2.0, 2.0
    elif distribution == "left_skewed":
        alpha, beta_param = 2.0, 8.0
    elif distribution == "right_skewed":
        alpha, beta_param = 8.0, 2.0
    elif distribution == "semi_left":
        alpha, beta_param = 2.0, 5.0
    elif distribution == "semi_right":
        alpha, beta_param = 5.0, 2.0
    elif distribution == "left_and_right":
        half = d // 2
        left = beta(half, 2.0, 8.0)
        right = beta(d - half, 8.0, 2.0)
        x = torch.cat([left, right])
        x = x.sort()[0]
        return _scale_eigenvalues(x, cond)
    elif distribution == "combo":
        mix = torch.from_numpy(rng.random(d).astype(np.float32)).to(device)
        left = beta(d, 2.0, 8.0)
        right = beta(d, 8.0, 2.0)
        x = mix * left + (1.0 - mix) * right
        x = x.sort()[0]
        return _scale_eigenvalues(x, cond)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    x = beta(d, alpha, beta_param)
    x = x.sort()[0]
    return _scale_eigenvalues(x, cond)


def _scale_eigenvalues(x: torch.Tensor, cond: float) -> torch.Tensor:
    """Map sorted normalized positions ``x`` in [0, 1] to log-spaced eigenvalues."""

    log_min = -math.log10(cond)
    return (10.0 ** (x * log_min)).clamp(1.0 / cond, 1.0)


def make_spd_matrix(
    d: int,
    device: torch.device,
    cond: float = 1e5,
    distribution: str = "random",
    seed: int = 42,
) -> torch.Tensor:
    """Return a symmetric positive-definite matrix with a controlled eigenvalue distribution."""

    eig = eigenvalues_for_distribution(d, cond, distribution, device, seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float32, device=device))
    return Q @ torch.diag(eig) @ Q.t()


def true_lmax(H: torch.Tensor) -> float:
    return torch.linalg.eigvalsh(H)[-1].item()


def lanczos_max(H: torch.Tensor, n_iter: int = 10) -> float:
    A = H.contiguous()
    largest_vals, _ = torch.lobpcg(A, k=1, largest=True, niter=n_iter)
    return largest_vals[0].item()


def power_iteration_max(H: torch.Tensor, n_iter: int = 10, seed: int = 0) -> float:
    gen = torch.Generator(device=H.device).manual_seed(seed)
    d = H.shape[0]
    v = torch.randn(d, generator=gen, dtype=H.dtype, device=H.device)
    v = v / torch.linalg.norm(v)
    for _ in range(n_iter):
        v = H @ v
        v = v / torch.linalg.norm(v)
    return (v @ (H @ v)).item()


def diagonal_proxy(H: torch.Tensor) -> float:
    return H.diagonal().max().item()


def timed(fn: Callable[[torch.Tensor], float], H: torch.Tensor, repeats: int = 5) -> Tuple[float, float]:
    """Return median wall-time (ms) and the last result of ``fn(H)``."""

    device = H.device
    for _ in range(2):
        fn(H)
    torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn(H)
        end_event.record()
        torch.cuda.synchronize(device)
        times.append(start_event.elapsed_time(end_event))

    return float(np.median(times)), result


def mean_diag(H: torch.Tensor) -> float:
    return H.diagonal().float().mean().item()


def v3_damp_percent(lambda_max: float, mean_diag_val: float, base: float = 0.05, alpha: float = 0.25,
                    module_factor: float = 1.0, clamp_min: float = 0.02, clamp_max: float = 0.08) -> float:
    spectral_ratio = lambda_max / max(mean_diag_val, 1e-18)
    damp = base * module_factor * (spectral_ratio ** alpha)
    return max(clamp_min, min(damp, clamp_max))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}")

    d = 4096
    cond = 1e5
    n_iter = 10
    base = 0.05
    alpha = 0.25
    clamp_min = 0.02
    clamp_max = 0.08
    distributions = [
        "random",
        "middle_skewed",
        "left_skewed",
        "right_skewed",
        "semi_left",
        "semi_right",
        "left_and_right",
        "combo",
    ]
    methods: List[Tuple[str, Callable[[torch.Tensor], float]]] = [
        ("diagonal", diagonal_proxy),
        ("power_iteration", lambda H: power_iteration_max(H, n_iter=n_iter)),
        ("lanczos", lambda H: lanczos_max(H, n_iter=n_iter)),
        ("eigh", true_lmax),
    ]

    header = (
        f"{'distribution':<18} | {'condition':>12} | {'method':<18} | "
        f"{'time_ms':>10} | {'lmax_err':>10} | {'damp_err':>10}"
    )
    print(header)
    print("-" * len(header))

    all_results: List[Dict[str, float]] = []
    for distribution in distributions:
        H = make_spd_matrix(d, device, cond=cond, distribution=distribution, seed=42)
        true_lm = true_lmax(H)
        true_min = torch.linalg.eigvalsh(H)[0].item()
        mean_d = mean_diag(H)
        true_damp = v3_damp_percent(true_lm, mean_d, base=base, alpha=alpha,
                                    clamp_min=clamp_min, clamp_max=clamp_max)
        matrix_condition = true_lm / max(abs(true_min), 1e-18)

        for name, fn in methods:
            ms, est_lmax = timed(fn, H)
            est_damp = v3_damp_percent(est_lmax, mean_d, base=base, alpha=alpha,
                                       clamp_min=clamp_min, clamp_max=clamp_max)
            lmax_err = abs(est_lmax - true_lm) / max(abs(true_lm), 1e-18)
            damp_err = abs(est_damp - true_damp) / max(abs(true_damp), 1e-18)
            print(
                f"{distribution:<18} | {matrix_condition:>12.2e} | {name:<18} | "
                f"{ms:>10.3f} | {lmax_err:>10.4f} | {damp_err:>10.4f}"
            )
            all_results.append(
                {
                    "distribution": distribution,
                    "condition": matrix_condition,
                    "method": name,
                    "time_ms": ms,
                    "lmax_err": lmax_err,
                    "damp_err": damp_err,
                }
            )

    print("\nSummary: median time (ms) per method")
    summary_header = f"{'method':<18} | {'median_ms':>10}"
    print(summary_header)
    print("-" * len(summary_header))
    for name, _ in methods:
        times = [r["time_ms"] for r in all_results if r["method"] == name]
        print(f"{name:<18} | {np.median(times):>10.3f}")


if __name__ == "__main__":
    main()
