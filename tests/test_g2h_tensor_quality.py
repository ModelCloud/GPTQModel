# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# test_g2h_tensor_quality.py
# Check whether major activation dtypes round-trip from GPU -> CPU -> GPU without
# accumulating numerical error. We use GPU 6 as requested, exercise both pinned
# and pageable host buffers, and report per-shape metrics so the test doubles as
# a quick experiment.

import pytest
import torch


def _roundtrip_metrics(src: torch.Tensor, device: torch.device, pin_memory: bool):
    if pin_memory:
        host = torch.empty_like(src, device="cpu", pin_memory=True)
        host.copy_(src, non_blocking=True)
        # Ensure the async copy completes before reusing the tensor.
        torch.cuda.synchronize(device)
        roundtrip = host.to(device, copy=True, non_blocking=True)
    else:
        host = src.to("cpu", copy=True)
        roundtrip = host.to(device, copy=True)

    diff = (src.float() - roundtrip.float()).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "nonzero_elements": int((diff != 0).sum().item()),
    }


@pytest.mark.cuda
@pytest.mark.parametrize("pin_memory", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_gpu_cpu_gpu_roundtrip_lossless(dtype, pin_memory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")

    gpu_index = 6
    if torch.cuda.device_count() <= gpu_index:
        pytest.skip(f"Need at least {gpu_index + 1} CUDA devices to exercise GPU {gpu_index}.")

    device = torch.device(f"cuda:{gpu_index}")

    # A mix of common activation shapes, including one large 2D tensor to stress memory copies.
    shapes = [
        (1, 12288),       # Transformer MLP activation
        (16, 1024, 64),   # Attention block with batch & heads
        (3, 224, 224),    # Vision-style activation
        (4096, 4096),     # Large square matrix to amplify any copy issues
    ]

    # Ensure deterministic data so that repeated runs are comparable.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    results = []

    with torch.cuda.device(device):
        for shape in shapes:
            src = torch.randn(shape, device=device, dtype=dtype)
            metrics = _roundtrip_metrics(src, device, pin_memory=pin_memory)
            results.append({"shape": shape, "dtype": dtype, "pin_memory": pin_memory, **metrics})

            # Expect no change; any non-zero indicates data corruption during the round-trip.
            assert metrics["max_abs_diff"] == 0.0, (
                f"Max diff {metrics['max_abs_diff']} detected for shape {shape} dtype {dtype} pin_memory={pin_memory}"
            )
            assert metrics["nonzero_elements"] == 0, (
                f"Found {metrics['nonzero_elements']} differing elements for shape {shape} dtype {dtype} pin_memory={pin_memory}"
            )

    for r in results:
        print(
            f"GPU6 round-trip dtype={r['dtype']} shape={r['shape']} pin_memory={r['pin_memory']}: "
            f"max_abs_diff={r['max_abs_diff']} mean_abs_diff={r['mean_abs_diff']} "
            f"nonzero_elements={r['nonzero_elements']}"
        )
