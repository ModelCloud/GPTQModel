# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A packed-embedding checkpoint must load through ``subfolder=`` too (#5386).

The loader's packed-embedding probe used the repository root instead of the
resolved checkpoint directory, so ``GPTQModel.load("bundle", subfolder="endpoint")``
failed on a checkpoint whose embedding was replaced by packed tensors. The probe
now inspects the selected checkpoint directory (#5461); this pins the reported
packed-embedding scenario.
"""

import torch
from safetensors import safe_open
from test_requant_embed_endpoint_5116 import _quantized_checkpoint, _requantize_input

from gptqmodel import GPTQModel


def _embedding_keys(checkpoint_dir):
    with safe_open(str(checkpoint_dir / "model.safetensors"), framework="pt") as handle:
        return set(handle.keys())


def test_packed_embedding_loads_through_the_parent_directory_subfolder(tmp_path):
    checkpoint = _quantized_checkpoint()
    model = GPTQModel.load(checkpoint, device="cpu", dtype=torch.bfloat16)
    _requantize_input(model, device=torch.device("cpu"))

    bundle = tmp_path / "bundle"
    endpoint = bundle / "endpoint"
    model.save(str(endpoint))

    keys = _embedding_keys(endpoint)
    assert "model.embed_tokens.qweight" in keys
    assert "model.embed_tokens.weight" not in keys

    direct = GPTQModel.load(str(endpoint), device="cpu", dtype=torch.bfloat16)
    nested = GPTQModel.load(str(bundle), subfolder="endpoint", device="cpu", dtype=torch.bfloat16)

    assert type(nested.model.get_input_embeddings()) is type(direct.model.get_input_embeddings())

    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        direct_logits = direct.model(ids).logits
        nested_logits = nested.model(ids).logits

    torch.testing.assert_close(nested_logits, direct_logits, rtol=0, atol=0)
