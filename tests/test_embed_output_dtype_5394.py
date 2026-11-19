# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A packed embedding must follow a whole-model dtype cast (#5394).

``TorchQuantEmbeddings`` remembered the model's compute dtype in a plain
attribute. ``Module.to()``/``.float()``/``.half()`` only convert parameters and
buffers, so the embedding kept emitting the pre-cast dtype and a downstream
converted linear rejected it::

    RuntimeError: expected m1 and m2 to have the same dtype

The recorded dtype now lives in a zero-element non-persistent buffer, which the
normal module conversion walks.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.quantization.config import FORMAT


def _packed_embedding(in_features: int = 32, out_features: int = 16) -> TorchQuantEmbeddings:
    embedding = TorchQuantEmbeddings(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    with torch.no_grad():
        embedding.qweight.fill_(0x44444444)
        embedding.qzeros.fill_(0x22222222)
        embedding.scales.fill_(0.5)
    embedding.post_init()
    return embedding


def _model(dtype: torch.dtype):
    embedding = _packed_embedding()
    embedding.set_embedding_output_dtype(dtype)
    model = torch.nn.Sequential(
        embedding, torch.nn.Linear(16, 4, bias=False, dtype=dtype)
    )
    return embedding, model


@pytest.mark.parametrize("cast", ["float", "half"])
def test_whole_model_cast_keeps_embedding_in_step(cast):
    embedding, model = _model(torch.bfloat16)
    ids = torch.tensor([[0, 1, 31]])

    before = model(ids)
    before_embedding = embedding(ids)
    assert before.dtype == torch.bfloat16

    getattr(model, cast)()
    target = torch.float32 if cast == "float" else torch.float16

    # The recorded dtype follows the cast instead of staying at BF16.
    assert embedding.embedding_output_dtype() == target
    assert embedding.scales.dtype == target
    assert embedding(ids).dtype == target

    after = model(ids)
    assert after.dtype == target
    # The fixture's embedding values are exact, so the gathered rows must match
    # bit for bit; the projection only differs by its accumulation precision.
    assert torch.equal(embedding(ids), before_embedding.to(target))
    assert torch.allclose(after.float(), before.float(), atol=1e-2, rtol=1e-2)


def test_output_dtype_anchor_is_not_persisted():
    embedding, _ = _model(torch.bfloat16)
    assert not any(
        "_embedding_output_dtype_anchor" in key for key in embedding.state_dict()
    )


def test_clearing_the_recorded_dtype_falls_back_to_the_scale_dtype():
    embedding, model = _model(torch.bfloat16)
    embedding.set_embedding_output_dtype(None)
    assert embedding.embedding_output_dtype() == embedding.scales.dtype

    model.float()
    assert embedding.embedding_output_dtype() == torch.float32
    assert model(torch.tensor([[0, 1, 31]])).dtype == torch.float32
