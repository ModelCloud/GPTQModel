# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from gptqmodel.models.inkling2_loop.configuration import Inkling2LoopVisionConfig
from gptqmodel.models.inkling2_loop.vision import Inkling2LoopVisionModel


@torch.no_grad()
def test_packed_inkling2_vision_attention_preserves_token_and_head_axes():
    config = Inkling2LoopVisionConfig(
        hidden_size=16, num_attention_heads=2, num_hidden_layers=1, patch_size=2,
        posemb_grid_size=4, projection_hidden_dim=24, text_hidden_size=32,
    )
    tower = Inkling2LoopVisionModel(config).eval()
    generator = torch.Generator().manual_seed(17)
    pixels = torch.randn(3, 4, 4, 3, generator=generator)
    grids = torch.tensor([[1, 1, 2], [1, 1, 1]])

    def check_attention(attention, args, kwargs, result):
        hidden = args[0]
        boundaries = kwargs["cu_seqlens"].tolist()
        cos, sin = kwargs["position_embeddings"]
        phase = torch.complex(cos[:, None, ::2], sin[:, None, ::2])
        q, k, v = (
            projection(hidden).reshape(-1, config.num_attention_heads, config.head_dim)
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
        )

        def rotate(states):
            pairs = torch.view_as_complex(states.unflatten(-1, (-1, 2)))
            return torch.view_as_real(pairs * phase).flatten(-2)

        q, k = rotate(q), rotate(k)
        pieces = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            heads = (states[start:end].transpose(0, 1) for states in (q, k, v))
            pieces.append(F.scaled_dot_product_attention(*heads).transpose(0, 1))
        expected = attention.out_proj(torch.cat(pieces).flatten(1))
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    handle = tower.layers[0].self_attn.register_forward_hook(check_attention, with_kwargs=True)
    try:
        packed = tower(pixels, grids).pooler_output
    finally:
        handle.remove()

    individual = [tower(patch, grid[None]).pooler_output for patch, grid in zip(pixels.split((2, 1)), grids)]
    assert packed.shape == (3, config.text_hidden_size)
    torch.testing.assert_close(packed, torch.cat(individual), atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_inkling2_vision_position_embeddings_follow_patch_merge_order():
    config = Inkling2LoopVisionConfig(
        hidden_size=16, num_attention_heads=2, num_hidden_layers=1, patch_size=2,
        posemb_grid_size=4, projection_hidden_dim=24, text_hidden_size=32,
    )
    tower = Inkling2LoopVisionModel(config).eval()
    tower.embeddings.patch_embedding.weight.zero_()
    table = tower.embeddings.position_embedding.weight
    table.copy_(torch.arange(16).float()[:, None].expand_as(table))

    # vLLM resamples its learned positions on the row-major image grid.
    # GPT-QModel's patch stream stores each 2x2 block together.
    positions = F.interpolate(table.T.reshape(1, 16, 4, 4), size=(2, 4), mode="bilinear")
    expected = positions.flatten(2)[0].T[[0, 1, 4, 5, 2, 3, 6, 7]]

    def check_embeddings(module, args, result):
        torch.testing.assert_close(result[1:], expected, atol=0, rtol=0)

    handle = tower.embeddings.register_forward_hook(check_embeddings)
    try:
        output = tower(torch.zeros(2, 4, 4, 3), torch.tensor([[1, 1, 2]]))
    finally:
        handle.remove()
    assert output.pooler_output.shape == (2, config.text_hidden_size)
