# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# test_d2h_concurrency.py
# pytest -q -s test_d2h_concurrency.py
import math
import time

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for this test"
)

def _mb(nbytes): return nbytes / (1024**2)

def _banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def test_three_d2h_transfers_concurrency_vs_serial():
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    _banner(
        f"GPU: {props.name} | asyncEngineCount={getattr(props, 'asyncEngineCount', 'n/a')} | "
        f"PCIe/Link: unknown (PyTorch doesn't expose)\n"
        "Expectation: multiple D2H on a single GPU serialize onto one D2H engine."
    )

    torch.cuda.set_device(dev)

    # Use a size large enough to dominate overhead but not stress CI.
    # ~256 MiB each => 3 * 256 MiB = 768 MiB total device RAM + pinned host buffers.
    elements = (256 * 1024 * 1024) // 2  # fp16 => 2 bytes/elt
    dtype = torch.float16

    # Device tensors
    d0 = torch.empty(elements, dtype=dtype, device=dev)
    d1 = torch.empty_like(d0)
    d2 = torch.empty_like(d0)

    # Pinned host buffers (required for async copies)
    h0 = torch.empty_like(d0, device="cpu", pin_memory=True)
    h1 = torch.empty_like(d1, device="cpu", pin_memory=True)
    h2 = torch.empty_like(d2, device="cpu", pin_memory=True)

    # Warmup: one D2H copy to touch paths
    h0.copy_(d0, non_blocking=True)
    torch.cuda.synchronize()

    # --- Serialized on a single stream ---
    s_serial = torch.cuda.Stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s_serial):
        h0.copy_(d0, non_blocking=True)
        h1.copy_(d1, non_blocking=True)
        h2.copy_(d2, non_blocking=True)
    torch.cuda.synchronize()
    serial_time = time.perf_counter() - t0

    # --- Launched concurrently on three streams ---
    s0, s1, s2 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.cuda.stream(s0):
        h0.copy_(d0, non_blocking=True)
    with torch.cuda.stream(s1):
        h1.copy_(d1, non_blocking=True)
    with torch.cuda.stream(s2):
        h2.copy_(d2, non_blocking=True)
    torch.cuda.synchronize()
    concurrent_time = time.perf_counter() - t1

    total_mb = 3 * _mb(d0.numel() * d0.element_size())
    print(f"\nTransferred total ~{total_mb:.1f} MiB (3 x ~{total_mb/3:.1f} MiB) D2H")
    print(f"[SERIAL]     {serial_time:.4f} s  | ~{total_mb/serial_time:.1f} MiB/s effective")
    print(f"[CONCURRENT] {concurrent_time:.4f} s  | ~{total_mb/concurrent_time:.1f} MiB/s effective")

    # We expect little to no speedup when "concurrent" (same-direction copies share the D2H engine).
    # Allow some tolerance either way depending on driver/runtime details.
    assert concurrent_time >= 0.8 * serial_time, (
        "Unexpected large speedup from concurrent D2H; "
        "this would contradict single-engine D2H behavior."
    )
    assert concurrent_time <= 1.3 * serial_time, (
        "Concurrent D2H took much longer than serialized; "
        "this suggests overheads far above expectation."
    )

def test_h2d_d2h_bidirectional_overlap_possible():
    """Optional: demonstrate one H2D can overlap one D2H if GPU has ≥2 copy engines."""
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    if getattr(props, "asyncEngineCount", 0) < 2:
        pytest.skip("GPU reports <2 copy engines; bidirectional overlap unlikely.")

    torch.cuda.set_device(dev)

    elements = (128 * 1024 * 1024) // 1  # 128 MiB in bytes (uint8)
    dtype = torch.uint8

    # Host buffers (pinned) and device tensors
    h_src = torch.empty(elements, dtype=dtype, device="cpu", pin_memory=True)
    h_dst = torch.empty(elements, dtype=dtype, device="cpu", pin_memory=True)
    d_buf = torch.empty(elements, dtype=dtype, device=dev)

    # Warmup
    d_buf.copy_(h_src, non_blocking=True)
    h_dst.copy_(d_buf, non_blocking=True)
    torch.cuda.synchronize()

    # Baseline: serialize H2D then D2H on one stream
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s):
        d_buf.copy_(h_src, non_blocking=True)  # H2D
        h_dst.copy_(d_buf, non_blocking=True)  # D2H
    torch.cuda.synchronize()
    serial = time.perf_counter() - t0

    # Overlap: H2D on one stream, D2H on another (should overlap on separate engines)
    sh2d, sd2h = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.cuda.stream(sh2d):
        d_buf.copy_(h_src, non_blocking=True)  # H2D
    with torch.cuda.stream(sd2h):
        h_dst.copy_(d_buf, non_blocking=True)  # D2H
    torch.cuda.synchronize()
    overlapped = time.perf_counter() - t1

    print(f"\n[H2D->D2H] SERIAL   {serial:.4f} s")
    print(f"[H2D||D2H] OVERLAP  {overlapped:.4f} s  (expect <= ~serial)")

    # Expect some overlap benefit (not necessarily 2x).
    assert overlapped <= 0.9 * serial or math.isclose(overlapped, serial, rel_tol=0.05)
