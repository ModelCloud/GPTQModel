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


def _pack_embedding(bits: int, fmt: FORMAT, vocab: int, dim: int, group_size: int = 32):
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
        pack_dtype=torch.int32,
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
