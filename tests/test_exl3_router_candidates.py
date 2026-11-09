import pytest
import torch
from torch import nn

from gptqmodel.utils.exl3_router_candidates import (
    bind_sigmoid_grouped_router_recovery,
    learned_router_ranked_choices,
)


class _Router(nn.Module):
    def __init__(
        self,
        *,
        experts: int = 8,
        groups: int = 1,
        topk_groups: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer("e_score_correction_bias", torch.zeros(experts))
        self.num_group = groups
        self.topk_group = topk_groups


def test_learned_router_ranking_fails_without_explicit_score_function() -> None:
    router = _Router()
    with pytest.raises(RuntimeError, match="explicit score function"):
        learned_router_ranked_choices(
            router,
            torch.randn(3, 8),
            rank_max=8,
        )


def test_sigmoid_grouped_router_ranking_matches_selected_topk() -> None:
    router = _Router(groups=2, topk_groups=1)
    bind_sigmoid_grouped_router_recovery(router)
    logits = torch.tensor(
        [
            [8.0, 7.0, 6.0, 5.0, 1.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 2.0, 3.0, 8.0, 7.0, 6.0, 5.0],
        ]
    )
    selected = torch.tensor([[0, 1], [4, 5]])

    scores, indices = learned_router_ranked_choices(
        router,
        logits,
        rank_max=4,
        selected_indices=selected,
    )

    assert scores.shape == indices.shape == (2, 4)
    assert torch.equal(indices[:, :2], selected)


def test_sigmoid_group_policy_uses_top2_sum_instead_of_group_maximum() -> None:
    router = _Router(groups=2, topk_groups=1)
    bind_sigmoid_grouped_router_recovery(router)
    logits = torch.tensor(
        [[10.0, -10.0, -10.0, -10.0, 0.5, 0.5, -10.0, -10.0]]
    )

    _scores, indices = learned_router_ranked_choices(
        router,
        logits,
        rank_max=4,
        selected_indices=torch.tensor([[4, 5]], dtype=torch.int64),
    )

    # Group zero has the single largest expert, but group one's two strongest
    # sigmoid scores have the larger sum and therefore win GLM's group gate.
    assert set(indices[0].tolist()) == {4, 5, 6, 7}


def test_learned_router_ranking_rejects_live_topk_mismatch() -> None:
    router = _Router()
    bind_sigmoid_grouped_router_recovery(router)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        learned_router_ranked_choices(
            router,
            torch.arange(8, dtype=torch.float32).reshape(1, 8),
            rank_max=8,
            selected_indices=torch.tensor([[0, 1]]),
        )


def test_learned_router_ranking_rejects_non_integer_live_indices() -> None:
    router = _Router()
    bind_sigmoid_grouped_router_recovery(router)
    with pytest.raises(RuntimeError, match="invalid geometry"):
        learned_router_ranked_choices(
            router,
            torch.arange(8, dtype=torch.float32).reshape(1, 8),
            rank_max=8,
            selected_indices=torch.tensor([[6.0, 7.0]]),
        )


def test_live_topk_is_authoritative_across_tied_candidate_widths() -> None:
    router = _Router(experts=256)
    bind_sigmoid_grouped_router_recovery(router)
    logits = torch.zeros((1, 256), dtype=torch.float32)
    live = torch.topk(
        torch.sigmoid(logits),
        k=2,
        dim=-1,
        sorted=False,
    ).indices

    _scores, indices = learned_router_ranked_choices(
        router,
        logits,
        rank_max=4,
        selected_indices=live,
    )

    assert set(indices[0, :2].tolist()) == set(live[0].tolist())
    assert not set(indices[0, :2].tolist()).intersection(
        indices[0, 2:].tolist()
    )
