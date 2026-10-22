# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

import gptqmodel.nn_modules.exllamav3_torch as exllamav3_torch


@pytest.fixture(autouse=True)
def _clear_cpu_bf16x3_cache():
    exllamav3_torch._cpu_bf16x3_support.cache_clear()
    yield
    exllamav3_torch._cpu_bf16x3_support.cache_clear()


def _make_module(
    size: int = 1024,
    bits: int = 4,
    seed: int = 731,
    *,
    in_features: int | None = None,
    out_features: int | None = None,
    trellis: torch.Tensor | None = None,
) -> exllamav3_torch.ExllamaV3TorchLinear:
    in_features = size if in_features is None else in_features
    out_features = size if out_features is None else out_features
    generator = torch.Generator().manual_seed(seed)
    if trellis is None:
        trellis = torch.randint(
            -32768,
            32767,
            (in_features // 16, out_features // 16, bits * 16),
            dtype=torch.int16,
            generator=generator,
        )
    tensors = {
        "trellis": trellis,
        "suh": (torch.rand(in_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "svh": (torch.rand(out_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "bias": torch.randn(out_features, generator=generator, dtype=torch.float16) * 0.01,
    }
    return exllamav3_torch.ExllamaV3TorchLinear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name="cpu_bf16x3",
        tensors=tensors,
    ).eval()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double()
    expected = expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


def _pack_symbols(symbols: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack low-bit EXL3 symbols into trellis words without using the runtime decoder."""

    words = torch.zeros(*symbols.shape[:-1], 16, bits, dtype=torch.int64)
    mask = (1 << bits) - 1
    for pos in range(16):
        value = symbols[..., pos::16] & mask
        bit_offset = pos * bits
        word_idx = bit_offset // 16
        bit_in_word = bit_offset % 16
        if bit_in_word + bits <= 16:
            words[..., word_idx] |= value << (16 - bit_in_word - bits)
        else:
            bits_first = 16 - bit_in_word
            bits_second = bits - bits_first
            words[..., word_idx] |= value >> bits_second
            words[..., word_idx + 1] |= (value & ((1 << bits_second) - 1)) << (16 - bits_second)

    packed = words.reshape(*symbols.shape[:-1], 16 * bits)
    packed = packed.view(*symbols.shape[:-1], -1, 2).flip(-1).reshape(*symbols.shape[:-1], 16 * bits)
    return packed.to(torch.int16)


def _encode_symbols(symbols: torch.Tensor, bits: int) -> torch.Tensor:
    """Reconstruct the tail-biting EXL3 state stream from the stored low-bit symbols."""

    mask = (1 << 16) - 1
    warmup = (16 + bits - 1) // bits - 1
    state = torch.zeros(symbols.shape[:-1], dtype=torch.int64)
    for index in range(256 - warmup, 256):
        state = ((state << bits) | symbols[..., index]) & mask

    encoded = torch.empty_like(symbols)
    for index in range(256):
        state = ((state << bits) | symbols[..., index]) & mask
        encoded[..., index] = state
    return encoded


def _codebook_3inst(encoded: torch.Tensor) -> torch.Tensor:
    raw = (encoded * 89226354 + 64248484) & 0xFFFFFFFF
    raw = 0x3B603B60 ^ (raw & 0x8FFF8FFF)
    halves = torch.stack((raw & 0xFFFF, (raw >> 16) & 0xFFFF), dim=-1)
    halves = halves.to(torch.uint16).contiguous().view(torch.float16).double()
    return halves[..., 0] + halves[..., 1]


def _tensor_core_perm_inverse() -> torch.Tensor:
    perm = [0] * 256
    for tile in range(32):
        row0 = (tile % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        col0 = tile // 4
        col1 = col0 + 8
        perm[tile * 8 + 0] = row0 * 16 + col0
        perm[tile * 8 + 1] = row1 * 16 + col0
        perm[tile * 8 + 2] = row2 * 16 + col0
        perm[tile * 8 + 3] = row3 * 16 + col0
        perm[tile * 8 + 4] = row0 * 16 + col1
        perm[tile * 8 + 5] = row1 * 16 + col1
        perm[tile * 8 + 6] = row2 * 16 + col1
        perm[tile * 8 + 7] = row3 * 16 + col1
    inverse = [0] * len(perm)
    for index, value in enumerate(perm):
        inverse[value] = index
    return torch.tensor(inverse, dtype=torch.long)


def _hadamard_128_reference() -> torch.Tensor:
    hadamard = torch.ones((1, 1), dtype=torch.float64)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), dim=1),
                torch.cat((hadamard, -hadamard), dim=1),
            ),
            dim=0,
        )
    return hadamard / math.sqrt(128.0)


def _independent_weight_oracle(
    symbols: torch.Tensor,
    bits: int,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    encoded = _encode_symbols(symbols, bits)
    decoded = _codebook_3inst(encoded)[..., _tensor_core_perm_inverse()]
    tiles_k, tiles_n = decoded.shape[:2]
    inner = decoded.view(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(
        tiles_k * 16,
        tiles_n * 16,
    )

    hadamard = _hadamard_128_reference()
    inner = (hadamard @ inner.view(-1, 128, inner.shape[1])).view_as(inner)
    inner *= suh.double().unsqueeze(1)
    inner = (inner.view(inner.shape[0], -1, 128) @ hadamard).view_as(inner)
    inner *= svh.double().unsqueeze(0)
    return inner


def _oracle(module: exllamav3_torch.ExllamaV3TorchLinear, x: torch.Tensor) -> torch.Tensor:
    weight = module.get_weight_tensor(torch.float32).double()
    bias = module.bias.double()
    return x.double() @ weight + bias


def _matmul_precision() -> str | None:
    matmul = getattr(torch.backends.mkldnn, "matmul", None)
    return None if matmul is None else matmul.fp32_precision


def _fp32_matmul(
    module: exllamav3_torch.ExllamaV3TorchLinear,
    x: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    previous_precision = _matmul_precision()
    torch.backends.mkldnn.matmul.fp32_precision = precision
    try:
        output = x.float() @ module.get_weight_tensor(torch.float32)
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = previous_precision
    return output + module.bias.float()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 7, 25, 64])
def test_exllamav3_torch_cpu_bf16x3_matches_fp64_oracle(dtype, rows):
    module = _make_module()
    generator = torch.Generator().manual_seed(rows + (1000 if dtype == torch.float16 else 0))
    x = torch.randn(rows, module.in_features, dtype=dtype, generator=generator)
    previous_precision = _matmul_precision()

    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype == dtype
    assert _matmul_precision() == previous_precision

    oracle = _oracle(module, x)
    raw_error = _relative_l2(actual, oracle)
    visible_error = _relative_l2(actual, oracle.to(dtype))

    assert raw_error <= 2e-3
    assert visible_error <= 2e-3


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_exllamav3_torch_chunked_forward_matches_independent_oracle(bits, dtype):
    in_features = 384
    out_features = 256
    generator = torch.Generator().manual_seed(bits + (2000 if dtype == torch.float16 else 0))
    symbols = torch.randint(
        0,
        1 << bits,
        (in_features // 16, out_features // 16, 256),
        dtype=torch.int64,
        generator=generator,
    )
    module = _make_module(
        bits=bits,
        seed=bits,
        in_features=in_features,
        out_features=out_features,
        trellis=_pack_symbols(symbols, bits),
    )
    x = torch.randn(7, in_features, dtype=dtype, generator=generator)

    with torch.inference_mode():
        actual = module(x)

    expected = x.double() @ _independent_weight_oracle(symbols, bits, module.suh, module.svh) + module.bias.double()
    raw_error = _relative_l2(actual, expected)
    visible_error = _relative_l2(actual, expected.to(dtype))

    assert actual.dtype == dtype
    assert raw_error <= 2e-3
    assert visible_error <= 2e-3


def test_exllamav3_torch_forward_decodes_per_call_without_dense_cache(monkeypatch):
    monkeypatch.setattr(exllamav3_torch, "_EXL3_OUTPUT_TILES_PER_CHUNK", 8)
    module = _make_module(size=512)
    x = torch.randn(7, module.in_features, dtype=torch.bfloat16)
    dense_elements = module.in_features * module.out_features
    cache_attributes = ("_cache_signature", "_inner_weight_fp32", "_weight_fp32")

    calls = []
    original_decode = module._decode_inner_weight

    def counted_decode(tile_n_start=0, tile_n_end=None):
        calls.append((tile_n_start, tile_n_end))
        return original_decode(tile_n_start, tile_n_end)

    monkeypatch.setattr(module, "_decode_inner_weight", counted_decode)

    with torch.inference_mode():
        module(x)
    first_chunks = [
        (start, min(start + 8, module.out_features // 16))
        for start in range(0, module.out_features // 16, 8)
    ]
    assert calls == first_chunks
    assert all(not hasattr(module, attribute) for attribute in cache_attributes)
    assert all(
        not isinstance(value, torch.Tensor) or value.numel() != dense_elements
        for value in module.__dict__.values()
    )

    with torch.inference_mode():
        module(x)
    assert calls == first_chunks * 2

    with torch.inference_mode():
        module.get_weight_tensor(torch.float32)
    assert calls[-1] == (0, None)
    assert all(not hasattr(module, attribute) for attribute in cache_attributes)
    assert all(
        not isinstance(value, torch.Tensor) or value.numel() != dense_elements
        for value in module.__dict__.values()
    )


def test_exllamav3_torch_unpack_indices_slices_output_tiles():
    module = _make_module(size=512)

    full = module._unpack_indices()
    for start, end in ((0, 8), (8, 16), (3, 7)):
        assert torch.equal(module._unpack_indices(start, end), full[:, start:end])


@pytest.mark.parametrize("bits", range(1, 17))
def test_exllamav3_torch_unpack_indices_matches_independent_oracle_for_all_bit_widths(bits):
    in_features = 384
    out_features = 256
    generator = torch.Generator().manual_seed(3000 + bits)
    symbols = torch.randint(
        0,
        1 << bits,
        (in_features // 16, out_features // 16, 256),
        dtype=torch.int64,
        generator=generator,
    )
    module = _make_module(
        bits=bits,
        trellis=_pack_symbols(symbols, bits),
        in_features=in_features,
        out_features=out_features,
    )

    assert torch.equal(module._unpack_indices(), _encode_symbols(symbols, bits))


def test_exllamav3_torch_zero_rows_preserves_shape():
    module = _make_module(size=256)
    x = torch.empty(0, module.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        output = module(x)

    assert output.shape == (0, module.out_features)
    assert output.dtype == x.dtype


def test_exllamav3_torch_cpu_bf16x3_scope_restores_precision():
    if not exllamav3_torch._cpu_bf16x3_support():
        pytest.skip("oneDNN BF16x3 is unavailable on this host")

    previous_precision = _matmul_precision()
    if previous_precision is None:
        pytest.skip("torch does not expose oneDNN matmul precision")

    with exllamav3_torch._cpu_bf16x3_matmul():
        assert _matmul_precision() == "bf16"
    assert _matmul_precision() == previous_precision


def test_exllamav3_torch_cpu_bf16x3_arithmetic_matches_ieee():
    if not exllamav3_torch._cpu_bf16x3_support():
        pytest.skip("oneDNN BF16x3 is unavailable on this host")

    module = _make_module()
    x = torch.randn(25, module.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        bf16x3 = _fp32_matmul(module, x, "bf16")
        ieee = _fp32_matmul(module, x, "ieee")

    assert _relative_l2(bf16x3, ieee) <= 1e-5


def test_exllamav3_torch_cpu_bf16x3_shape_gate():
    assert not exllamav3_torch._cpu_bf16x3_should_use(1, 512, 512)
    assert not exllamav3_torch._cpu_bf16x3_should_use(7, 512, 512)
    assert exllamav3_torch._cpu_bf16x3_should_use(32, 512, 512)
    assert exllamav3_torch._cpu_bf16x3_should_use(1, 1024, 1024)
    assert not exllamav3_torch._cpu_bf16x3_should_use(0, 1024, 1024)


def test_exllamav3_torch_cpu_bf16x3_falls_back_without_avx512_bf16(monkeypatch):
    monkeypatch.setattr(exllamav3_torch, "has_avx512_bf16", lambda: False)
    exllamav3_torch._cpu_bf16x3_support.cache_clear()

    assert not exllamav3_torch._cpu_bf16x3_support()

    module = _make_module()
    x = torch.randn(7, module.in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = module(x)

    oracle = _oracle(module, x)
    assert _relative_l2(actual, oracle.to(torch.bfloat16)) <= 2e-3


def test_exllamav3_torch_cpu_bf16x3_env_disable(monkeypatch):
    monkeypatch.setenv(exllamav3_torch._CPU_BF16X3_ENV, "0")
    exllamav3_torch._cpu_bf16x3_support.cache_clear()

    assert not exllamav3_torch._cpu_bf16x3_support()
