# SPDX-License-Identifier: Apache-2.0
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as implementation
from gptqmodel.utils.kernel import KernelConfig, configure_kernel, get_kernel_config


def test_workspace_is_private_per_stream(monkeypatch):
    """Each stream owns its scratch so concurrent launches cannot overwrite it."""
    calls = []
    allocated = []

    configure_kernel(KernelConfig())

    def allocate(device, K, batch):
        allocated.append((device, K, batch))
        return (f"costs-{len(allocated)}", f"edges-{len(allocated)}")

    monkeypatch.setattr(implementation, "_workspace_buffers", {})
    monkeypatch.setattr(implementation, "_allocate_temp_buffers", allocate)
    monkeypatch.setattr(
        implementation,
        "ext",
        SimpleNamespace(quantize_tiles=lambda *args: calls.append(args)),
    )
    tiles = SimpleNamespace(device="cuda:0", shape=(4, 256))
    stream_a = object()
    stream_b = object()
    current = {"stream": stream_a}
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current["stream"])

    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)
    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)
    current["stream"] = stream_b
    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)

    # One buffer per stream, reused by every launch on that stream.
    assert allocated == [(torch.device("cuda:0"), 4, 256)] * 2
    assert calls[0][3] is calls[1][3]
    assert calls[0][4] is calls[1][4]
    assert calls[2][3] is not calls[0][3]
    assert calls[2][4] is not calls[0][4]


def test_tile_batch_defaults_and_clamps():
    """The scratch slot count is opt-in: unchanged by default, bounded when set."""
    configure_kernel(KernelConfig())
    assert implementation._tile_batch_size() == 256

    for value, expected in ((8, 8), (1, 1), (4096, 256)):
        configure_kernel(KernelConfig(exl3_tile_batch=value))
        assert implementation._tile_batch_size() == expected, value

    with pytest.raises(ValueError):
        KernelConfig(exl3_tile_batch=0)


def test_oversized_tile_sets_are_split_into_launches(monkeypatch):
    """More tiles than scratch slots must split the launch, not overflow it."""
    calls = []

    monkeypatch.setattr(implementation, "_workspace_buffers", {})
    monkeypatch.setattr(
        implementation,
        "_allocate_temp_buffers",
        lambda device, K, batch: (object(), object()),
    )
    monkeypatch.setattr(
        implementation,
        "ext",
        SimpleNamespace(quantize_tiles=lambda tiles, out, idx, *rest: calls.append(tiles.shape[0])),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    configure_kernel(KernelConfig(exl3_tile_batch=2))

    tiles = torch.zeros((5, 256))
    implementation._quantize_tiles_with_workspace(
        tiles, torch.zeros_like(tiles), torch.zeros_like(tiles), 4, False, False
    )
    assert calls == [2, 2, 1]

    # At or below the slot count the launch is unchanged.
    calls.clear()
    small = torch.zeros((2, 256))
    implementation._quantize_tiles_with_workspace(
        small, torch.zeros_like(small), torch.zeros_like(small), 4, False, False
    )
    assert calls == [2]


@pytest.mark.parametrize(
    ("bits", "scale_search", "limit"),
    [(2.5, False, 512), (3.0, False, 512), (3.5, False, 1024),
     (2.5, True, 1024), (3.0, True, 1024), (3.5, True, 1024)],
)
def test_q4nx_launch_limits_follow_history_size_and_keep_explicit_caps(monkeypatch, bits, scale_search, limit):
    class CudaTiles:
        is_cuda = True

        def __init__(self, count):
            self.shape = (count, 256)

        def __getitem__(self, rows):
            start, stop, _ = rows.indices(self.shape[0])
            return CudaTiles(stop - start)

    calls = []

    def quantize(tiles, output, indices, kv, search):
        assert tiles.shape == output.shape == indices.shape
        calls.append((tiles.shape[0], kv, search))

    monkeypatch.setattr(implementation.torch.version, "hip", None)
    monkeypatch.setattr(implementation, "ext", SimpleNamespace(quantize_tiles_q4nx=quantize))
    previous = get_kernel_config()
    try:
        for policy, expected in [(KernelConfig(), [limit, 5]), (KernelConfig(exl3_tile_batch=256), [256, 5])]:
            configure_kernel(policy)
            tiles = CudaTiles(sum(expected))
            implementation._quantize_tiles_with_workspace(
                tiles, tiles, tiles, bits, False, False, q4nx=True, scale_search=scale_search,
            )
            assert calls == [(size, int(4 * bits), scale_search) for size in expected]
            calls.clear()
    finally:
        configure_kernel(previous)


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_concurrent_streams_do_not_overwrite_tile_workspace(device):
    if torch.cuda.device_count() <= torch.device(device).index:
        pytest.skip(f"requires {device}")
    generator = torch.Generator().manual_seed(152)
    tiles = [torch.randn((16, 256), generator=generator).to(device) for _ in range(4)]
    config = {"K": 4, "mcg": True}
    expected = [
        tuple(value.cpu() for value in implementation.quantize_tiles(tile, config))
        for tile in tiles
    ]
    barrier = threading.Barrier(4)

    def quantize(index):
        stream = torch.cuda.Stream(device=device)
        barrier.wait()
        with torch.cuda.stream(stream):
            for _ in range(3):
                actual = implementation.quantize_tiles(tiles[index], config)
                stream.synchronize()
                assert all(
                    torch.equal(value.cpu(), reference)
                    for value, reference in zip(actual, expected[index])
                )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(quantize, range(4)))


def test_quant_stream_is_shared_per_device_across_threads(monkeypatch):
    """Every worker thread must reuse one quantization stream per device.

    The caching allocator pools blocks per stream, so a stream per worker
    stranded one cache (and one 512 MiB tile scratch) per worker for work that a
    per-device lock serializes anyway (issue #4586).
    """
    created = []

    class FakeCuda:
        @staticmethod
        def Stream(device=None):
            created.append(device)
            return object()

    fake_torch = SimpleNamespace(device=torch.device, cuda=FakeCuda())
    monkeypatch.setattr(implementation, "torch", fake_torch)
    # Keep the synthetic stream objects out of subsequent native launches.
    monkeypatch.setattr(implementation, "_quant_streams", {})

    barrier = threading.Barrier(8)
    results = []
    results_lock = threading.Lock()

    def resolve():
        barrier.wait()
        resolved = (
            implementation.get_quant_stream("cuda:0"),
            implementation.get_quant_stream(torch.device("cuda:0")),
            implementation.get_quant_stream("cuda:1"),
        )
        with results_lock:
            results.append(resolved)

    workers = [threading.Thread(target=resolve) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
    assert not any(worker.is_alive() for worker in workers)

    # One creation per device, not one per caller thread.
    assert len(created) == 2, created
    assert results
    for resolved in results:
        assert resolved[0] is resolved[1]
        assert resolved[0] is not resolved[2]
    assert len({id(resolved[0]) for resolved in results}) == 1
    assert len({id(resolved[2]) for resolved in results}) == 1


_CONCURRENT_QUANTIZE_SCRIPT = """
import threading

import torch

from gptqmodel.exllamav3.ext import exllamav3_runtime_available
from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as implementation

if not torch.cuda.is_available():
    print("NO_CUDA")
    raise SystemExit(4)
if not exllamav3_runtime_available():
    print("EXTENSION_UNAVAILABLE")
    raise SystemExit(3)

device = torch.device("cuda:0")
streams_seen = set()
scratch_seen = set()
lock = threading.Lock()

_original_stream = implementation.get_quant_stream
_original_alloc = implementation._allocate_temp_buffers


def counted_stream(target):
    stream = _original_stream(target)
    with lock:
        streams_seen.add(id(stream))
    return stream


def counted_alloc(*args, **kwargs):
    buffers = _original_alloc(*args, **kwargs)
    with lock:
        scratch_seen.add(tuple(id(tensor) for tensor in buffers))
    return buffers


implementation.get_quant_stream = counted_stream
implementation._allocate_temp_buffers = counted_alloc

shapes = [(256, 128), (256, 256), (384, 128), (384, 256)]
fixtures = []
for seed, (k, n) in enumerate(shapes):
    weight_generator = torch.Generator().manual_seed(900 + seed)
    activation_generator = torch.Generator().manual_seed(1900 + seed)
    weight = torch.randn((k, n), generator=weight_generator) * 0.02
    activations = torch.randn((256, k), generator=activation_generator)
    fixtures.append((weight, (2.0 / 256) * activations.T @ activations))


def run(index):
    weight, hessian = fixtures[index]
    args = {
        "K": 4,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": 787,
        "mcg": True,
    }
    # quantize_exl3 regularizes the weight in place, so every call needs its own
    # copy of the fixture; sharing one would make the comparison meaningless.
    _, _, packed = implementation.quantize_exl3(
        weight.to(device).clone(),
        {"H": hessian.to(device).clone(), "count": 256, "finalized": False},
        args,
        return_weight_q=False,
    )
    torch.cuda.synchronize(device)
    metrics = args["error_metrics"]
    assert metrics["hessian_metric_status"] == "ok", metrics
    assert metrics["hessian_weighted_relative_error"] < 0.05, metrics
    return {key: value.detach().cpu() for key, value in packed.items()}


reference = [run(index) for index in range(len(shapes))]

results = [None] * len(shapes)
errors = []
barrier = threading.Barrier(len(shapes))


def worker(index):
    try:
        barrier.wait(120)
        results[index] = run(index)
    except BaseException as exc:  # noqa: BLE001 - report, do not mask
        errors.append(f"{type(exc).__name__}: {exc}")


threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(shapes))]
for item in threads:
    item.start()
for item in threads:
    item.join(900)

if errors:
    print("ERRORS:", errors)
    raise SystemExit(5)
if any(result is None for result in results):
    print("INCOMPLETE")
    raise SystemExit(6)
for index, result in enumerate(results):
    for key in reference[index]:
        if not torch.equal(result[key], reference[index][key]):
            print("MISMATCH", index, key)
            raise SystemExit(7)

print("STREAMS", len(streams_seen))
print("SCRATCH", len(scratch_seen))
print("OK")
raise SystemExit(0)
"""


def _free_threaded_build() -> bool:
    """True only for a free-threaded (3.13t/3.14t) interpreter.

    The child process sets ``PYTHON_GIL=0``; a regular build aborts with
    "Disabling the GIL is not supported by this build" before running the
    script, so the test can neither pass nor observe its sentinels (#4768).
    """

    return sysconfig.get_config_var("Py_GIL_DISABLED") == 1


@pytest.mark.skipif(
    not _free_threaded_build(),
    reason="the PYTHON_GIL=0 subprocess needs a free-threaded build (#4768)",
)
def test_concurrent_quantize_shares_one_stream_and_scratch_per_device():
    """Four concurrent projections must not each own a stream and a scratch set.

    This is the measured defect: one quantization stream per worker means one
    allocator pool per worker, so the tile scratch and the allocator's free
    blocks multiply by the worker count even though one device lock serializes
    the kernels. The subprocess also proves that sharing the stream did not
    alias the scratch: every concurrently produced trellis matches the packed
    result of an isolated call, and each call's Hessian-weighted error stays
    inside the normal quantization range.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    if not torch.cuda.is_available():
        pytest.skip("EXL3 quantization requires CUDA/HIP")

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHON_GIL"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_QUANTIZE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    stdout = completed.stdout
    if "EXTENSION_UNAVAILABLE" in stdout:
        pytest.skip("the EXL3 extension is unavailable on this host")
    assert completed.returncode == 0, stdout + completed.stderr
    assert "STREAMS 1" in stdout, stdout
    assert "SCRATCH 1" in stdout, stdout
    assert "OK" in stdout
