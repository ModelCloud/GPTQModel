# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.utils.nf12_codec import rowmajor_pack_cpu
from gptqmodel.utils.nf12_tenstorrent import prepare_nf12
from gptqmodel.utils.nf12_ttnn_rows import _transcode_rows


def test_row_storage_preserves_shared_bytes_through_noc_alignment():
    base = 57
    generator = torch.Generator().manual_seed(15)
    shape = (520, 129)
    raw = torch.randint(0, 4096, shape, dtype=torch.int16, generator=generator)
    raw = ((raw & 2047) | ((raw & 2048) << 4)) + (base << 8)
    # A valid code that would occupy the marker is escaped too.
    raw[0, :128] = base << 8
    raw[513, 121:129] = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32641, -63], dtype=torch.int16)
    descriptor, weight = prepare_nf12(raw.view(torch.bfloat16), "raw.rows", exp_base=base)
    packed, escapes, count = _transcode_rows(descriptor, weight)
    expected, expected_escapes = rowmajor_pack_cpu(weight.packed, weight.escapes, base)
    assert packed.shape == (520, 384)
    assert count == len(expected_escapes)
    torch.testing.assert_close(packed[:, :288], expected[:len(raw)], atol=0, rtol=0)
    torch.testing.assert_close(packed[:, 288:], torch.zeros((520, 96), dtype=torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(escapes.reshape(-1, 4)[:count], expected_escapes, atol=0, rtol=0)
