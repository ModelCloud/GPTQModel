# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU ownership and integrity tests for V4.1 calibration state."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.utils.ple_mmap import MappedPLETable
from gptqmodel.utils.v41_checkpoint import (
    load_frontier,
    load_routed_batch,
    save_frontier,
    save_routed_batch,
)
from gptqmodel.utils.v41_dspark import V41DSparkInput
from gptqmodel.utils.v41_replay import V41ReplayBatch
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch


@pytest.mark.parametrize("separate_shards", [False, True])
def test_mapped_ple_gathers_owned_rows_in_order(tmp_path, separate_shards):
    prefix = "layers.0.engram.embed"
    weight = torch.arange(96).reshape(3, 32).to(torch.float8_e4m3fn)
    scale = torch.ones((3, 1), dtype=torch.float8_e8m0fnu)
    weight_path = tmp_path / "weight.safetensors"
    scale_path = tmp_path / "scale.safetensors" if separate_shards else weight_path
    if separate_shards:
        save_file({prefix + ".weight": weight}, weight_path)
        save_file({prefix + ".scale": scale}, scale_path)
    else:
        save_file({prefix + ".weight": weight, prefix + ".scale": scale}, weight_path)

    with MappedPLETable(weight_path, prefix, scale_path=scale_path, max_gather_rows=3) as table:
        assert (table.rows, table.width) == (3, 32)
        table.prefetch([2, 0, 2])
        gathered = table.gather([2, 0, 2])
        table.release()
        with pytest.raises(ValueError, match="bounded row count"):
            table.gather([0, 1, 2, 0])
        with pytest.raises(IndexError, match="outside table"):
            table.gather([3])

    expected_weight = b"".join(weight.view(torch.uint8)[row].numpy().tobytes() for row in (2, 0, 2))
    expected_scale = b"".join(scale.view(torch.uint8)[row].numpy().tobytes() for row in (2, 0, 2))
    assert gathered == (expected_weight, expected_scale)
    with pytest.raises(RuntimeError, match="closed"):
        table.gather([0])


def _frontier():
    return V41ReplayBatch(
        next_layer=2,
        hidden=torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),
        pre_mix=torch.zeros(1, 2, 6),
        shared={"kv": torch.tensor([1.0, 2.0])},
        kwargs={"position_ids": torch.tensor([[0, 1]])},
        engram_rows={3: torch.ones(1, 2, 4)},
        target_layer_ids=(3,),
    )


def test_frontier_roundtrip_is_owned_and_journal_bound(tmp_path):
    path = tmp_path / "frontier.safetensors"
    provenance = {"source": "fixture", "recipe": 1}
    source = _frontier()
    digest = save_frontier(source, path, provenance=provenance)
    loaded = load_frontier(path, expected_sha256=digest, expected_provenance=provenance)

    torch.testing.assert_close(loaded.hidden, source.hidden, rtol=0, atol=0)
    torch.testing.assert_close(loaded.shared["kv"], source.shared["kv"], rtol=0, atol=0)
    assert loaded.engram_rows[3].shape == (1, 2, 4)
    loaded.hidden[0, 0, 0] = -999
    reloaded = load_frontier(path, expected_sha256=digest, expected_provenance=provenance)
    assert reloaded.hidden[0, 0, 0] == 0
    assert source.hidden[0, 0, 0] == 0

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frontier(path, expected_sha256="0" * 64, expected_provenance=provenance)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_frontier(path, expected_sha256=digest, expected_provenance={"source": "other"})

    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frontier(path, expected_sha256=digest, expected_provenance=provenance)


def test_routed_frontier_kind_and_geometry_are_checked(tmp_path):
    path = tmp_path / "routed.safetensors"
    provenance = {"source": "fixture"}
    batch = V41RoutedBatch(
        hidden=torch.ones(2, 3),
        logits=torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
        weights=torch.ones(2, 1),
        indices=torch.tensor([[0], [1]]),
    )
    digest = save_routed_batch(batch, path, provenance=provenance)
    loaded = load_routed_batch(path, expected_sha256=digest, expected_provenance=provenance)
    torch.testing.assert_close(loaded.indices, batch.indices, rtol=0, atol=0)
    with pytest.raises(ValueError, match="kind mismatch"):
        load_frontier(path, expected_sha256=digest, expected_provenance=provenance)

    batch.indices[1, 0] = 2
    with pytest.raises(ValueError, match="out of range"):
        save_routed_batch(batch, path, provenance=provenance)


def test_joint_dspark_replay_uses_known_tokens_and_only_target_prefix():
    config = SimpleNamespace(
        dspark_target_layer_ids=(1, 2),
        dspark_block_size=3,
        dspark_noise_token_id=0,
        hc_mult=2,
    )
    embedding = nn.Embedding(16, 2)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(32).reshape(16, 2))
    projection = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
    adapter = V41DSparkInput(embedding, projection, nn.LayerNorm(2), config).eval()
    target_features = {
        1: torch.arange(6, dtype=torch.float32).reshape(1, 6, 1),
        2: (torch.arange(6, dtype=torch.float32) + 2).reshape(1, 6, 1),
    }
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8]])
    positions = torch.tensor([1, 3])

    replay = adapter.prepare_joint(target_features, tokens, positions=positions)
    assert replay.hidden.shape == (2, 3, 2, 2)
    assert replay.kwargs["anchor_positions"].tolist() == [1, 3]
    torch.testing.assert_close(replay.hidden[:, 0, 0], embedding(torch.tensor([5, 7])))
    torch.testing.assert_close(replay.hidden[:, 1:, 0], embedding(torch.zeros(2, 2, dtype=torch.long)))

    changed_future = {key: value.clone() for key, value in target_features.items()}
    for value in changed_future.values():
        value[:, 4:] += 1000
    replay_after_future_change = adapter.prepare_joint(changed_future, tokens, positions=positions)
    torch.testing.assert_close(replay.kwargs["main_x"], replay_after_future_change.kwargs["main_x"], rtol=0, atol=0)
