# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exactness and contract coverage for the standard-P32 MLX kernel."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    reconstruct_p32_window_inner_weight,
    repack_p32_planar_to_window,
)
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_p32_mlx import (
    qvq_mlx_p32_window_gemv,
    qvq_mlx_repack_p32_planar_to_window,
)

P32_RATES = (1, 1.5, 2, 2.5, 3, 3.5)


def _p32_case(bits: float, m: int, *, k: int = 32, n: int = 64):
    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(32000 + transition_bits * 100 + m)
    tile_count = (k // 16) * (n // 16)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (128, tile_count),
        generator=generator,
        dtype=torch.int32,
    )
    planar = planar_pack_rows(edges, transition_bits).T.contiguous()
    selectors = torch.randint(0, 2, (tile_count * 8,), generator=generator, dtype=torch.uint8)
    bank_ids = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([3], dtype=torch.uint8)
    x = torch.randn((m, k), generator=generator, dtype=torch.float32)
    return x, planar, bank_ids, bank_alt_id


@pytest.mark.parametrize("bits", P32_RATES)
def test_qvq_mlx_p32_repack_is_bit_exact_and_storage_neutral(bits):
    _, planar, _, _ = _p32_case(bits, 1)
    expected = repack_p32_planar_to_window(planar, bits=bits)
    actual = qvq_mlx_repack_p32_planar_to_window(mx.array(planar.numpy()), bits)
    mx.eval(actual)

    assert actual.shape == planar.shape
    assert actual.size == planar.numel()
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


@pytest.mark.parametrize("bits", P32_RATES)
@pytest.mark.parametrize("m", (1, 2, 4, 8, 17))
def test_qvq_mlx_p32_window_gemv_matches_dense_oracle(bits, m):
    x, planar, bank_ids, bank_alt_id = _p32_case(bits, m)
    window = repack_p32_planar_to_window(planar, bits=bits)
    inner = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=x.shape[1],
        out_features=64,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    expected = x @ inner
    actual = qvq_mlx_p32_window_gemv(
        mx.array(x.numpy()),
        mx.array(window.numpy()),
        bits,
        out_features=64,
        bank_ids=mx.array(bank_ids.numpy()),
        bank_alt_id=int(bank_alt_id.item()),
    )
    mx.eval(actual)

    assert actual.dtype == mx.float32
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(actual)),
        expected,
        rtol=2e-6,
        atol=1e-5,
    )


def test_qvq_mlx_p32_window_contract_rejects_lossy_dimensions_and_bad_split():
    x, planar, bank_ids, bank_alt_id = _p32_case(2, 1)
    window = repack_p32_planar_to_window(planar, bits=2)
    inputs = (
        mx.array(x.numpy()),
        mx.array(window.numpy()),
        mx.array(bank_ids.numpy()),
    )

    with pytest.raises(TypeError, match="out_features must be an integer"):
        qvq_mlx_p32_window_gemv(
            inputs[0],
            inputs[1],
            2,
            out_features=64.0,
            bank_ids=inputs[2],
            bank_alt_id=int(bank_alt_id.item()),
        )
    with pytest.raises(ValueError, match="split override"):
        qvq_mlx_p32_window_gemv(
            inputs[0],
            inputs[1],
            2,
            out_features=64,
            bank_ids=inputs[2],
            bank_alt_id=int(bank_alt_id.item()),
            _split_count_override=3,
        )
