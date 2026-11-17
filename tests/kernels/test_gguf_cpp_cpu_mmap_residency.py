# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GGUF_CPP_CPU keeps one resident copy of an mmap-loaded packed weight.

``GGUFCppKernel.post_init`` copies an unaligned safetensors mmap view into a
32-byte aligned buffer so the forward can hand it to ggml without a copy
(#4395). The mmap pages it faulted in used to stay mapped next to that copy
for the life of the model (+522 MiB ``RssFile`` on Llama-3.2-1B q4_k_m, i.e.
the weight resident twice). They are now reclaimed once the copy exists.
Residency is read per page from ``/proc/self/pagemap``.
"""

from __future__ import annotations

import mmap
import struct
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import (
    _GGML_CPU_WEIGHT_ALIGNMENT,
    GGUFCppKernel,
    _reclaim_file_pages,
)


IN_FEATURES = 2048
OUT_FEATURES = 512
BITS = "q4_k_s"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc/self/pagemap"
)


def _require_bridge() -> None:
    GGUFCppKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCppKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python unavailable: {err}")


def _whole_pages(tensor: torch.Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    end = start + tensor.numel() * tensor.element_size()
    page = mmap.PAGESIZE
    return -(-start // page), end // page


def _resident_pages(tensor: torch.Tensor) -> int:
    """Pages of ``tensor`` mapped in this process (pagemap bit 63)."""

    first, last = _whole_pages(tensor)
    with open("/proc/self/pagemap", "rb") as pagemap:
        pagemap.seek(first * 8)
        entries = pagemap.read((last - first) * 8)
    return sum(
        (entry >> 63) & 1 for (entry,) in struct.iter_unpack("<Q", entries)
    )


def _kernel() -> GGUFCppKernel:
    return GGUFCppKernel(
        bits=BITS,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        exact_matmul=False,
    ).eval()


def _packed() -> torch.Tensor:
    torch.manual_seed(4395)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float32).eval()
    source = GGUFTorchLinear(
        bits=BITS,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    source.pack_original(linear, scales=None, zeros=None)
    return source.qweight.clone()


def test_post_init_reclaims_the_mmap_pages_it_copied(tmp_path):
    _require_bridge()
    packed = _packed()
    path = tmp_path / "model.safetensors"
    # Freshly written: its page-cache pages may still be dirty, as after a
    # quantize -> save -> load in one process.
    save_file({"qweight": packed}, str(path))

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        mapped = handle.get_tensor("qweight")
        assert mapped.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT != 0
        first, last = _whole_pages(mapped)
        pages = last - first
        assert pages > 16
        assert torch.equal(mapped, packed)  # faults every page in
        assert _resident_pages(mapped) == pages

        kernel = _kernel()
        kernel.qweight = mapped
        kernel.post_init()

        # One resident copy: the aligned buffer, not the buffer plus the mmap.
        assert kernel.qweight.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT == 0
        assert kernel.qweight.data_ptr() != mapped.data_ptr()
        assert _resident_pages(mapped) == 0
        assert torch.equal(kernel.qweight, packed)
        # Reclaiming never changes data: the mmap view re-faults from the file.
        assert torch.equal(mapped, packed)

        reference = _kernel()
        reference.load_state_dict({"qweight": packed}, strict=True)
        reference.post_init()
        x = torch.randn(3, IN_FEATURES, dtype=torch.float32)
        with torch.inference_mode():
            assert torch.equal(kernel(x), reference(x))
        assert torch.equal(kernel.state_dict()["qweight"], packed)


def test_copy_on_write_pages_are_kept(tmp_path):
    _require_bridge()
    packed = _packed()
    path = tmp_path / "model.safetensors"
    save_file({"qweight": packed}, str(path))

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        mapped = handle.get_tensor("qweight")
        assert mapped.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT != 0
        # A write into the private mapping makes a copy-on-write page that
        # exists nowhere else; unmapping it would revert the byte.
        first, _last = _whole_pages(mapped)
        offset = first * mmap.PAGESIZE - mapped.data_ptr()
        flat = mapped.view(-1)
        flat[offset] = flat[offset] ^ 0xFF
        written = mapped.clone()

        assert _reclaim_file_pages(mapped) == 0
        assert torch.equal(mapped, written)

        kernel = _kernel()
        kernel.qweight = mapped
        kernel.post_init()
        assert torch.equal(kernel.qweight, written)
        assert torch.equal(mapped, written)


def test_anonymous_memory_is_never_advised():
    # Heap memory would be swapped out, not dropped; only file pages qualify.
    storage = torch.empty(64 * mmap.PAGESIZE + 1, dtype=torch.uint8)
    storage.fill_(7)
    view = storage[1:]
    assert _reclaim_file_pages(view) == 0
    assert _resident_pages(view) == _whole_pages(view)[1] - _whole_pages(view)[0]
    assert bool((view == 7).all())
