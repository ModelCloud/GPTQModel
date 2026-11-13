# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.model import find_modules, is_embeddings_module_quantized


def test_find_modules_includes_embeddings():
    model = nn.Module()
    model.embed_tokens = nn.Embedding(16, 8)
    model.proj = nn.Linear(8, 8)

    modules = find_modules(model)

    assert modules["embed_tokens"] is model.embed_tokens
    assert modules["proj"] is model.proj


def test_quantized_embedding_detection_reads_safetensors_keys(tmp_path):
    save_file(
        {
            "model.embed_tokens.qweight": torch.zeros((2, 2), dtype=torch.int32),
            "model.embed_tokens.scales": torch.ones((2, 2), dtype=torch.float16),
            "lm_head.weight": torch.zeros((2, 2), dtype=torch.float16),
        },
        tmp_path / "model.safetensors",
    )

    assert is_embeddings_module_quantized(
        model_dir=str(tmp_path),
        input_embed_name="model.embed_tokens",
        output_embed_name="lm_head",
    ) == (True, False)


def test_quantized_embedding_detection_reads_sharded_index(tmp_path):
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "lm_head.qweight": "model-00002-of-00002.safetensors",
            "lm_head.qzeros": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    assert is_embeddings_module_quantized(
        model_dir=str(tmp_path),
        input_embed_name="model.embed_tokens",
        output_embed_name="lm_head",
    ) == (False, True)


def test_embedding_kernel_is_role_only():
    assert TorchQuantEmbeddings.SUPPORTS_BACKEND_SELECTION is False


# The row-decode gate is table-size based; use a table above the measured
# crossover so the tests exercise the row path on both CPU and GPU.
ROW_DECODE_VOCAB = 65536
ROW_DECODE_DIM = 1024


def _pack_embedding(
    bits: int,
    fmt: FORMAT,
    vocab: int,
    dim: int,
    group_size: int = 32,
    pack_dtype: torch.dtype = torch.int32,
):
    torch.manual_seed(787)
    embedding = nn.Embedding(vocab, dim)
    groups = math.ceil(vocab / group_size)
    module = TorchQuantEmbeddings(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=vocab,
        out_features=dim,
        pack_dtype=pack_dtype,
        bias=False,
        format=fmt,
    )
    module.pack_original(
        embedding,
        scales=torch.rand(dim, groups) * 0.02,
        zeros=torch.randint(0, 1 << bits, (dim, groups)),
        g_idx=torch.arange(vocab, dtype=torch.int32) // group_size,
    )
    module.optimize = lambda *args, **kwargs: None
    module.post_init()
    return module.eval()


def _boundary_ids(vocab: int, pack_factor: int) -> torch.Tensor:
    """Ids around every storage-word boundary, including the 32-token one."""
    ids = {0, 1, vocab - 1, vocab - 2}
    for boundary in (pack_factor, 2 * pack_factor, 31, 32, 33, 63, 64, 65, 127, 128, 129):
        ids.update(
            token
            for token in (boundary - 1, boundary, boundary + 1)
            if 0 <= token < vocab
        )
    return torch.tensor(sorted(ids), dtype=torch.long).reshape(1, -1)


@pytest.mark.parametrize(
    "bits,fmt",
    [(bits, FORMAT.GPTQ_V2) for bits in (2, 3, 4, 8)]
    + [(bits, FORMAT.GPTQ_P) for bits in (2, 3, 4, 5, 6, 7, 8)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_row_decode_matches_full_table(bits, fmt, dtype):
    vocab = ROW_DECODE_VOCAB if fmt == FORMAT.GPTQ_V2 else ROW_DECODE_DIM
    module = _pack_embedding(bits, fmt, vocab=vocab, dim=ROW_DECODE_DIM).to(dtype)
    reference = module.dequantize_weight()
    limit = module.EMBEDDING_ROW_DECODE_LIMIT

    # Boundaries of every packing width, repeated ids, and a shuffled sample
    # that crosses scale groups.
    cases = [
        torch.tensor([[0, 1, 2, 31, 32, 33, 127, 128, vocab - 1, vocab - 2, 7, 7]]),
        torch.randperm(vocab)[:limit].reshape(2, -1),
        torch.randint(0, vocab, (4, 3)),
    ]
    for input_ids in cases:
        torch.testing.assert_close(module(input_ids), F.embedding(input_ids, reference), rtol=0, atol=0)
        assert module(input_ids).dtype == dtype

    over_limit = torch.randint(0, vocab, (1, limit + 5))
    torch.testing.assert_close(module(over_limit), F.embedding(over_limit, reference), rtol=0, atol=0)


@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_row_decode_uses_storage_word_width(bits, pack_dtype, dtype):
    module = _pack_embedding(
        bits,
        FORMAT.GPTQ_V2,
        vocab=ROW_DECODE_VOCAB,
        dim=ROW_DECODE_DIM,
        pack_dtype=pack_dtype,
    ).to(dtype)
    reference = module.dequantize_weight()
    pack_factor = module.pack_factor

    input_ids = _boundary_ids(ROW_DECODE_VOCAB, pack_factor)
    assert input_ids.numel() <= module.EMBEDDING_ROW_DECODE_LIMIT

    torch.testing.assert_close(
        module(input_ids), F.embedding(input_ids, reference), rtol=0, atol=0
    )
    assert module(input_ids).dtype == dtype


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.GPTQ_P])
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_row_decode_all_formats_and_storage_widths(
    bits, fmt, pack_dtype, dtype
):
    # A small table keeps the full format x bits x pack_dtype matrix cheap; the
    # real table-size gate is covered by test_embedding_row_decode_matches_full_table.
    module = _pack_embedding(
        bits, fmt, vocab=512, dim=128, pack_dtype=pack_dtype
    ).to(dtype)
    module.EMBEDDING_ROW_DECODE_MIN_TABLE_BYTES = 0
    reference = module.dequantize_weight()
    input_ids = _boundary_ids(512, module.pack_factor)

    dequant_calls = []
    original = module.dequantize_weight
    module.dequantize_weight = lambda *args, **kwargs: (
        dequant_calls.append(1),
        original(*args, **kwargs),
    )[1]
    result = module(input_ids)

    # The row path, not the whole-table fallback, produced this result.
    assert dequant_calls == []
    torch.testing.assert_close(result, F.embedding(input_ids, reference), rtol=0, atol=0)
    assert result.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="accelerator required")
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16]
)
def test_embedding_row_decode_matches_full_table_on_device(bits, pack_dtype, dtype):
    device = torch.device("cuda")
    module = _pack_embedding(
        bits, FORMAT.GPTQ_V2, vocab=512, dim=128, pack_dtype=pack_dtype
    ).to(dtype).to(device)
    module.EMBEDDING_ROW_DECODE_MIN_TABLE_BYTES = 0
    # The reference must be the portable Torch decode, not a backend-specific
    # Triton layout, so the test compares the same arithmetic on the device.
    module._triton_dequant_enabled = False
    reference = module.dequantize_weight()
    input_ids = _boundary_ids(512, module.pack_factor).to(device)

    dequant_calls = []
    original = module.dequantize_weight
    module.dequantize_weight = lambda *args, **kwargs: (
        dequant_calls.append(1),
        original(*args, **kwargs),
    )[1]
    result = module(input_ids)

    assert dequant_calls == []
    torch.testing.assert_close(result, F.embedding(input_ids, reference), rtol=0, atol=0)
    assert result.dtype == dtype


@pytest.mark.parametrize(
    "fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.GPTQ_P]
)
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_embedding_row_index_metadata_tracks_pack_width(bits, fmt, pack_dtype):
    vocab = 256
    module = _pack_embedding(
        bits, fmt, vocab=vocab, dim=64, pack_dtype=pack_dtype
    )
    state = module._embedding_row_decode_state()

    assert state is not None
    assert module.pack_dtype_bits == {
        torch.int8: 8,
        torch.int16: 16,
        torch.int32: 32,
    }[pack_dtype]
    tokens = torch.arange(vocab, dtype=torch.int32)
    expected_rows = torch.div(tokens, module.pack_factor, rounding_mode="floor").to(torch.int32)
    expected_shifts = (tokens.remainder(module.pack_factor) * bits).to(torch.int32)
    torch.testing.assert_close(state["word_rows"], expected_rows)
    torch.testing.assert_close(state["shifts"], expected_shifts)


@pytest.mark.parametrize(
    "bits,fmt,pack_dtype",
    [
        (3, FORMAT.GPTQ, torch.int8),
        (3, FORMAT.GPTQ_V2, torch.int16),
        (3, FORMAT.GPTQ_P, torch.int8),
        (5, FORMAT.GPTQ_P, torch.int8),
        (6, FORMAT.GPTQ_P, torch.int16),
        (7, FORMAT.GPTQ_P, torch.int8),
    ],
)
def test_embedding_rejects_storage_widths_outside_layout_contract(
    bits, fmt, pack_dtype
):
    if bits == 3:
        ok, error = TorchQuantEmbeddings.validate(
            bits=bits,
            group_size=32,
            sym=True,
            desc_act=False,
            in_features=256,
            out_features=64,
            pack_dtype=pack_dtype,
            dtype=torch.float16,
        )
        assert not ok
        assert isinstance(error, NotImplementedError)
        return

    with pytest.raises(NotImplementedError):
        _pack_embedding(bits, fmt, vocab=256, dim=64, pack_dtype=pack_dtype)


def test_embedding_row_decode_matrix_covers_declared_support():
    assert set(TorchQuantEmbeddings.SUPPORTS_BITS) == {2, 3, 4, 5, 6, 7, 8}
    assert set(TorchQuantEmbeddings.SUPPORTS_DTYPES) == {
        torch.float16,
        torch.bfloat16,
    }
    assert set(TorchQuantEmbeddings.SUPPORTS_PACK_DTYPES) == {
        torch.int8,
        torch.int16,
        torch.int32,
    }


def test_embedding_row_decode_keeps_packed_storage(monkeypatch):
    module = _pack_embedding(
        4, FORMAT.GPTQ_V2, vocab=ROW_DECODE_VOCAB, dim=ROW_DECODE_DIM
    ).to(torch.float16)
    qweight = module.qweight
    packed_bytes = qweight.untyped_storage().nbytes()

    dequant_calls = []
    original = module.dequantize_weight

    def tracked(*args, **kwargs):
        dequant_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "dequantize_weight", tracked)
    with torch.inference_mode():
        for _ in range(3):
            module(torch.tensor([[1, 2, 3]]))

    # Row decode must not unpack the whole table, and the packed weights stay
    # resident (the same 4-bit words) between forwards.
    assert dequant_calls == []
    assert module.qweight is qweight
    assert module.qweight.untyped_storage().nbytes() == packed_bytes
    assert module.EMBEDDING_ROW_DECODE_LIMIT >= 1


def test_embedding_row_decode_follows_weight_mutation():
    module = _pack_embedding(
        4, FORMAT.GPTQ_V2, vocab=ROW_DECODE_VOCAB, dim=ROW_DECODE_DIM
    ).to(torch.float16)
    input_ids = torch.tensor([[5, 6, 7]])
    first = module(input_ids)

    with torch.no_grad():
        module.qweight.zero_()
    second = module(input_ids)

    torch.testing.assert_close(second, F.embedding(input_ids, module.dequantize_weight()), rtol=0, atol=0)
    assert not torch.equal(first, second)
