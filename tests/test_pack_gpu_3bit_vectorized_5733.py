# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#5733: ``pack_gpu`` packs 3-bit/planar blocks with one vectorized call.

``pack_gpu`` itself needs CUDA, so these tests exercise the block helper it now
uses and compare it against the per-32-row-group packer it replaced. The
per-group oracle is the former ``_pack_gpu_rows_3_values`` body, copied
verbatim, and the two paths must stay bit-identical.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import (
    _pack_code_cols_3,
    _pack_gpu_block_3_or_planar,
    _pack_rows_3,
)
from gptqmodel.utils.planar_packing import planar_pack_rows


MASK32 = (1 << 32) - 1


def _per_group_pack_3(blk: torch.Tensor) -> torch.Tensor:
    """The int64 per-32-row-group packer that ``pack_gpu`` used before #5733."""

    device = blk.device
    mask = torch.tensor(MASK32, dtype=torch.int64, device=device)
    range10 = torch.arange(10, dtype=torch.int64, device=device).view(-1, 1)
    x = blk.to(torch.int64)
    A = ((x[0:10] << (range10 * 3)).sum(dim=0, dtype=torch.int64)) & mask
    A = (A | ((x[10] << 30) & mask)) & mask

    B = ((x[10] >> 2) & 1) & mask
    B = (B | ((x[11:21] << (range10 * 3 + 1)).sum(dim=0, dtype=torch.int64))) & mask
    B = (B | ((x[21] << 31) & mask)) & mask

    C = ((x[21] >> 1) & 0x3) & mask
    C = (C | ((x[22:32] << (range10 * 3 + 2)).sum(dim=0, dtype=torch.int64))) & mask

    return torch.stack([A, B, C], dim=0)


def _per_group_qweight_3(codes: torch.Tensor) -> torch.Tensor:
    groups = codes.shape[0] // 32
    return torch.cat(
        [
            _per_group_pack_3(codes[g * 32:(g + 1) * 32]).to(torch.int32)
            for g in range(groups)
        ],
        dim=0,
    )


def _per_group_qzeros_3(zeros: torch.Tensor) -> torch.Tensor:
    chunks = []
    for g in range(zeros.shape[1] // 32):
        block = zeros[:, g * 32:(g + 1) * 32]
        chunks.append(_per_group_pack_3(block.T).transpose(0, 1).to(torch.int32))
    return torch.cat(chunks, dim=1)


@pytest.mark.parametrize(
    "rows,cols",
    ((32, 32), (64, 96), (96, 64), (256, 160), (1024, 128)),
)
def test_pack_gpu_block_matches_per_group_3bit(rows, cols):
    torch.manual_seed(5733)
    codes = torch.randint(0, 8, (rows, cols), dtype=torch.int32)

    packed = _pack_gpu_block_3_or_planar(codes, bits=3, planar=False)

    assert torch.equal(packed, _per_group_qweight_3(codes))


@pytest.mark.parametrize("bits", (3, 5, 6, 7))
@pytest.mark.parametrize("rows,cols", ((32, 32), (64, 96), (128, 160)))
def test_pack_gpu_block_matches_per_group_planar(bits, rows, cols):
    torch.manual_seed(5733 + bits)
    codes = torch.randint(0, 1 << bits, (rows, cols), dtype=torch.int32)

    packed = _pack_gpu_block_3_or_planar(codes, bits=bits, planar=True)

    expected = torch.cat(
        [
            planar_pack_rows(codes[g * 32:(g + 1) * 32], bits)
            for g in range(rows // 32)
        ],
        dim=0,
    )
    assert torch.equal(packed, expected)


@pytest.mark.parametrize("groups,cols", ((1, 32), (5, 64), (32, 160)))
def test_pack_gpu_qzeros_matches_per_group_3bit(groups, cols):
    torch.manual_seed(5733)
    zeros = torch.randint(0, 8, (groups, cols), dtype=torch.int64)

    packed = _pack_code_cols_3(zeros)

    assert torch.equal(packed, _per_group_qzeros_3(zeros))


def test_pack_rows_3_builds_offsets_on_the_code_device():
    """The shift grid must follow the codes, or CUDA packing fails (#5733)."""

    codes = torch.zeros((32, 8), dtype=torch.int32, device="meta")

    packed = _pack_rows_3(codes)

    assert packed.device.type == "meta"
    assert packed.shape == (3, 8)


def test_pack_gpu_block_rejects_unsupported_layout():
    with pytest.raises(NotImplementedError, match="Unsupported bits=4"):
        _pack_gpu_block_3_or_planar(torch.zeros((32, 32), dtype=torch.int32), bits=4, planar=False)
