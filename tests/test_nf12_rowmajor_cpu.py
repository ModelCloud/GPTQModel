# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import struct

import torch

from gptqmodel.utils.nf12_codec import encode, rowmajor_pack_cpu


def _cuda_pack_order(packed, canonical_escapes, base):
    n, width = packed.shape
    k_blocks = width // 96
    source = packed.numpy().tobytes()
    output = bytearray(len(source))
    escapes = []
    for values in canonical_escapes.tolist():
        escapes.extend((values[0:2] + values[4:6], values[2:4] + values[6:8]))
    for tile in range(n // 16 * k_blocks):
        nb, kb = divmod(tile, k_blocks)
        for tid in range(128):
            lane, atom = divmod(tid, 4)
            source_offset = tile * 1536
            meta = struct.unpack_from("<4B", source, source_offset + tid * 4)
            low_offset = source_offset + 512 * (1 + atom // 2) + lane * 16 + (atom % 2) * 8
            original_escape = struct.unpack_from("<I", source, low_offset)[0]
            row_group, subset = divmod(lane, 4)
            group = atom * 4 + subset
            for upper in range(2):
                meta_low, meta_high = meta[upper], meta[upper + 2]
                lows = source[low_offset + 2 * upper : low_offset + 2 * upper + 2]
                start = low_offset + 4 + 2 * upper
                lows += source[start : start + 2]
                if not any(meta):
                    low_word = struct.pack("<I", original_escape * 2 + upper)
                elif not (meta_low or meta_high):
                    low_word = struct.pack("<I", len(escapes))
                    escapes.append([base * 256 + value for value in lows])
                else:
                    low_word = lows
                row = nb * 16 + row_group + upper * 8
                dest = (row * k_blocks + kb) * 96
                struct.pack_into("2B", output, dest + 2 * group, meta_low, meta_high)
                output[dest + 32 + 4 * group : dest + 36 + 4 * group] = low_word
    return torch.frombuffer(output, dtype=torch.uint8).reshape_as(packed), torch.tensor(escapes, dtype=torch.int16)


def test_cpu_rowmajor_pack_matches_shared_cuda_hip_bytes_across_slabs():
    bits = torch.full((16, 128), 0x3F00, dtype=torch.int16)
    bits[8:, :64] = -16640
    bits[:8, 64:] = -16640
    bits[0, 64] = 0x7FC0
    bits[8, 64] = -32768
    bits[1, 65] = 1
    bits[9, 65] = -1
    packed, escapes, base = encode(bits.repeat(33, 1).view(torch.bfloat16), exp_base=63)
    actual, actual_escapes = rowmajor_pack_cpu(packed, escapes, base)
    expected, expected_escapes = _cuda_pack_order(packed, escapes, base)
    assert len(escapes) == 2 * 33
    assert len(expected_escapes) == 2 * len(escapes) + 254 * 33
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_escapes, expected_escapes, atol=0, rtol=0)
