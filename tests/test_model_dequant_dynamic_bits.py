# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``convert_gptq_file`` must resolve per-module ``dynamic`` bit widths.

GPTQ checkpoints can mix widths, for example a 3-bit body with a 4-bit
``lm_head``. The decoder has to pick each module's width from ``dynamic``
instead of applying the checkpoint default everywhere.
"""

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.utils.model_dequant import convert_gptq_file, pack_cols


def _pack_rows(values, bits):
    values = values.to(torch.int32)
    rows, cols = values.shape
    if bits == 3:
        blocks = rows // 32
        values_i64 = (values.to(torch.int64) & 0x7).reshape(blocks, 32, cols)
        packed = torch.zeros((blocks * 3, cols), dtype=torch.int32)
        for block in range(blocks):
            word0 = torch.zeros((cols,), dtype=torch.int64)
            for idx in range(10):
                word0 |= values_i64[block, idx, :] << (3 * idx)
            word0 |= values_i64[block, 10, :] << 30
            word1 = (values_i64[block, 10, :] >> 2) & 0x1
            for idx in range(10):
                word1 |= values_i64[block, 11 + idx, :] << (1 + 3 * idx)
            word1 |= values_i64[block, 21, :] << 31
            word2 = (values_i64[block, 21, :] >> 1) & 0x3
            for idx in range(10):
                word2 |= values_i64[block, 22 + idx, :] << (2 + 3 * idx)
            packed[block * 3 + 0, :] = word0.to(torch.int32)
            packed[block * 3 + 1, :] = word1.to(torch.int32)
            packed[block * 3 + 2, :] = word2.to(torch.int32)
        return packed

    pack_factor = 32 // bits
    words = (values.to(torch.int64) & ((1 << bits) - 1)).reshape(rows // pack_factor, pack_factor, cols)
    shifts = (torch.arange(pack_factor, dtype=torch.int64) * bits).view(1, pack_factor, 1)
    return (words << shifts).sum(dim=1).to(torch.int32)


def _oracle_weight(codes, zeros, scales, group_size):
    in_features = codes.shape[0]
    group = in_features if group_size <= 0 else group_size
    g_idx = torch.arange(in_features, dtype=torch.long) // group
    return ((codes.to(torch.float64) - zeros.to(torch.float64)[g_idx]) * scales.to(torch.float64)[g_idx]).T


def _module_tensors(prefix, bits, in_features, out_features, group_size, seed):
    generator = torch.Generator().manual_seed(seed)
    high = (1 << bits) - 1
    groups = 1 if group_size <= 0 else in_features // group_size
    codes = torch.randint(0, high + 1, (in_features, out_features), generator=generator, dtype=torch.int32)
    zeros = torch.randint(0, high + 1, (groups, out_features), generator=generator, dtype=torch.int32)
    scales = torch.rand((groups, out_features), generator=generator, dtype=torch.float32) + 0.01
    g_idx = (torch.arange(in_features, dtype=torch.long) // (in_features if groups == 1 else group_size)).to(torch.int32)
    tensors = {
        f"{prefix}.qweight": _pack_rows(codes, bits),
        f"{prefix}.qzeros": pack_cols(zeros, bits, pack_dtype=torch.int32),
        f"{prefix}.scales": scales,
        f"{prefix}.g_idx": g_idx,
    }
    return tensors, _oracle_weight(codes, zeros, scales, group_size)


@pytest.mark.parametrize(
    "default_bits,override_bits",
    [(3, 4), (4, 8)],
    ids=["3bit-main-4bit-head", "4bit-main-8bit-head"],
)
def test_convert_gptq_file_resolves_dynamic_bits(tmp_path, default_bits, override_bits):
    body = "model.layers.0.self_attn.q_proj"
    head = "lm_head"
    body_tensors, body_expected = _module_tensors(body, default_bits, 32, 32, 32, 1)
    head_tensors, head_expected = _module_tensors(head, override_bits, 32, 64, 32, 2)
    path = tmp_path / "model.safetensors"
    save_file({**body_tensors, **head_tensors}, str(path))

    config = {
        "bits": default_bits,
        "group_size": 32,
        "checkpoint_format": "gptq_v2",
        "dynamic": {head: {"bits": override_bits}},
    }
    decoded = convert_gptq_file(path, torch.float32, config, "cpu")

    torch.testing.assert_close(decoded[f"{body}.weight"], body_expected.to(torch.float32), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(decoded[f"{head}.weight"], head_expected.to(torch.float32), rtol=1e-6, atol=1e-6)


def test_convert_gptq_file_accepts_string_bits(tmp_path):
    prefix = "model.layers.0.self_attn.q_proj"
    tensors, expected = _module_tensors(prefix, 4, 32, 32, 32, 3)
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    decoded = convert_gptq_file(
        path,
        torch.float32,
        {"bits": "4", "group_size": 32, "checkpoint_format": "gptq_v2"},
        "cpu",
    )

    torch.testing.assert_close(decoded[f"{prefix}.weight"], expected.to(torch.float32), rtol=1e-6, atol=1e-6)
