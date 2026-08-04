# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""DeepSeek-V4-Flash-0731 GEMM speed sweep for Swordfish, Marlin, and Machete.

Runs a synthetic 4-bit GPTQ symmetric (g=128) weight for every Linear shape in
`deepseek-ai/DeepSeek-V4-Flash-0731` and reports throughput in TFLOPS. Each
backend is warmed up twice and then measured over 10 iterations; OOM and
unsupported-configuration entries are recorded as "OOM" or "N/A".
"""

import gc
import unittest
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.swordfish import SwordfishLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.logger import setup_logger

log = setup_logger()


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        return f"Swordfish requires Blackwell (sm100+); found sm{major}{minor}"
    return None


_SKIP_REASON = _skip_reason()

# Unique (in_features, out_features) Linear shapes extracted from
# `deepseek-ai/DeepSeek-V4-Flash-0731` with `init_empty_weights()`.
DEEPSEEK_V4_SHAPES: List[Tuple[int, int]] = [
    (1024, 32768),
    (1024, 8192),
    (2048, 4096),
    (4096, 1024),
    (4096, 129280),
    (4096, 2048),
    (4096, 256),
    (4096, 512),
    (4096, 64),
    (4096, 8192),
    (8192, 4096),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _quantize_sym(weight: torch.Tensor, bits: int, group_size: int):
    out_features, in_features = weight.shape
    half_range = 1 << (bits - 1)
    num_groups = in_features // group_size

    q = torch.empty_like(weight)
    scales = torch.zeros((out_features, num_groups), dtype=weight.dtype, device=weight.device)
    for g in range(num_groups):
        mask = slice(g * group_size, (g + 1) * group_size)
        block = weight[:, mask]
        max_abs = block.abs().max(dim=1, keepdim=True).values
        max_abs[max_abs == 0] = 1.0
        scale = max_abs / (half_range - 1)
        q_block = torch.round(block / scale).clamp(-(half_range), half_range - 1) + half_range
        q[:, mask] = q_block
        scales[:, g : g + 1] = scale

    zeros = torch.full((out_features, num_groups), half_range, dtype=weight.dtype, device=weight.device)
    return q, scales, zeros


def _pack_torch_reference(
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    device: torch.device,
):
    linear = nn.Linear(in_features, out_features, bias=False, dtype=weight.dtype, device="cpu")
    with torch.no_grad():
        linear.weight.copy_(weight)

    torch_linear = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        register_buffers=True,
    )
    g_idx = (torch.arange(in_features, dtype=torch.int32) // group_size)
    torch_linear.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    torch_linear = torch_linear.to(device=device)
    torch_linear.post_init()
    return torch_linear


def _measure(fn, x, warmup: int = 2, iters: int = 100) -> float:
    """Return average elapsed time in milliseconds."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestSwordfishSpeedDeepSeekV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from gptqmodel.utils.swordfish import prewarm_swordfish_extension

        prewarm_swordfish_extension()

    def _build_backend(self, torch_linear: TorchLinear, backend_class, dtype: torch.dtype, device: torch.device):
        """Instantiate a backend, copy the packed GPTQ buffers, and run post_init."""
        module = backend_class(
            bits=torch_linear.bits,
            group_size=torch_linear.requested_group_size,
            desc_act=torch_linear.desc_act,
            sym=torch_linear.sym,
            in_features=torch_linear.in_features,
            out_features=torch_linear.out_features,
            bias=False,
            dtype=dtype,
        )
        module.qweight = nn.Parameter(
            torch_linear.qweight.data.detach().clone().contiguous(), requires_grad=False
        )
        module.scales = nn.Parameter(
            torch_linear.scales.data.detach().clone().contiguous(), requires_grad=False
        )
        if torch_linear.g_idx is not None and torch_linear.g_idx.numel() > 0:
            module.g_idx = nn.Parameter(
                torch_linear.g_idx.data.detach().clone().contiguous(), requires_grad=False
            )
        if torch_linear.qzeros is not None and torch_linear.qzeros.numel() > 0:
            module.qzeros = nn.Parameter(
                torch_linear.qzeros.data.detach().clone().contiguous(), requires_grad=False
            )
        module = module.to(device=device)
        module.post_init()
        return module

    def _benchmark_backend_for_shape(
        self,
        backend_class,
        name: str,
        torch_linear: TorchLinear,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[Tuple[int, int, int], str]:
        """Return {(in, out, m): throughput string} for one backend."""
        try:
            module = self._build_backend(torch_linear, backend_class, dtype, device)
        except Exception as exc:
            log.warning("%s init failed for %dx%d: %s", name, torch_linear.in_features, torch_linear.out_features, exc)
            return {}

        results: Dict[Tuple[int, int, int], str] = {}
        for m in BATCH_SIZES:
            x = torch.randn((m, torch_linear.in_features), dtype=dtype, device=device) * 0.5
            try:
                ms = _measure(lambda x, mod=module: mod(x), x)
                flops = 2 * m * torch_linear.in_features * torch_linear.out_features
                tflops = flops / (ms * 1e-3) / 1e12
                results[(torch_linear.in_features, torch_linear.out_features, m)] = f"{tflops:6.2f}"
            except torch.cuda.OutOfMemoryError:
                results[(torch_linear.in_features, torch_linear.out_features, m)] = "   OOM"
            except Exception as exc:
                log.warning("%s m=%d failed for %dx%d: %s", name, m, torch_linear.in_features, torch_linear.out_features, exc)
                results[(torch_linear.in_features, torch_linear.out_features, m)] = "   N/A"
            finally:
                del x
        del module
        torch.cuda.empty_cache()
        gc.collect()
        return results

    def test_deepseek_v4_flash_0731_speed(self):
        bits = 4
        group_size = 128
        dtype = torch.bfloat16
        device = torch.device("cuda:0")

        backends: List[Tuple[str, type]] = [("Swordfish", SwordfishLinear)]
        unavailable: List[str] = []

        try:
            from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
            from gptqmodel.utils.marlin import marlin_runtime_available

            if marlin_runtime_available():
                backends.append(("Marlin", MarlinLinear))
            else:
                from gptqmodel.utils.marlin import marlin_runtime_error
                unavailable.append(f"Marlin: {marlin_runtime_error()}")
        except Exception as exc:
            unavailable.append(f"Marlin: {exc}")

        try:
            from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
            from gptqmodel.utils.machete import machete_runtime_available, machete_runtime_error

            if machete_runtime_available():
                backends.append(("Machete", MacheteLinear))
            else:
                unavailable.append(f"Machete: {machete_runtime_error()}")
        except Exception as exc:
            unavailable.append(f"Machete: {exc}")

        # Dense baseline for comparison.
        dense_results: Dict[Tuple[int, int, int], str] = {}

        all_results: Dict[str, Dict[Tuple[int, int, int], str]] = {}
        for name, _ in backends:
            all_results[name] = {}

        for in_features, out_features in DEEPSEEK_V4_SHAPES:
            torch.manual_seed(42)
            weight = torch.randn((out_features, in_features), dtype=dtype, device="cpu") * 0.5
            q, scales, zeros = _quantize_sym(weight, bits, group_size)
            torch_linear = _pack_torch_reference(
                bits, group_size, in_features, out_features, q, scales, zeros, device
            )

            # Dense FP16 baseline from the same quantized checkpoint.
            with torch.no_grad():
                dense_weight = torch_linear.dequantize_weight().to(device=device, dtype=dtype)

            def _dense_fn(x, weight=dense_weight):
                return torch.matmul(x, weight)

            for m in BATCH_SIZES:
                x = torch.randn((m, in_features), dtype=dtype, device=device) * 0.5
                try:
                    ms = _measure(_dense_fn, x)
                    flops = 2 * m * in_features * out_features
                    tflops = flops / (ms * 1e-3) / 1e12
                    dense_results[(in_features, out_features, m)] = f"{tflops:6.2f}"
                except torch.cuda.OutOfMemoryError:
                    dense_results[(in_features, out_features, m)] = "   OOM"
                except Exception:
                    dense_results[(in_features, out_features, m)] = "   N/A"
                finally:
                    del x

            for name, backend_class in backends:
                backend_results = self._benchmark_backend_for_shape(
                    backend_class, name, torch_linear, dtype, device
                )
                all_results[name].update(backend_results)

            del torch_linear, dense_weight
            torch.cuda.empty_cache()
            gc.collect()

        # Print one consolidated table.
        width = 90
        print("\nDeepSeek-V4-Flash-0731 4-bit GPTQ speed (TFLOPS) on B300")
        print("=" * width)
        if unavailable:
            print("Unavailable backends: " + "; ".join(unavailable))
            print("-" * width)
        header = f"{'K':>6} {'N':>7} {'M':>4} {'Dense':>8}"
        for name, _ in backends:
            header += f" {name:>10}"
        print(header)
        print("-" * width)
        for in_features, out_features in DEEPSEEK_V4_SHAPES:
            for m in BATCH_SIZES:
                row = f"{in_features:>6} {out_features:>7} {m:>4} {dense_results.get((in_features, out_features, m), '     -'):>8}"
                for name, _ in backends:
                    row += f" {all_results[name].get((in_features, out_features, m), '     -'):>10}"
                print(row)
        print("=" * width)
