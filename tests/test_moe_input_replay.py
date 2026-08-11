from types import SimpleNamespace
from unittest.mock import Mock

import torch

from gptqmodel.models.moe_input_replay import RoutedMoEInputReplayAttachment, routed_projection_roles
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks


def _module(*flags):
    return SimpleNamespace(state={"module_tree_flags": frozenset(flags)})


def test_routed_input_replay_uses_explicit_projection_tags_and_exact_batches():
    gate_up = {
        "arbitrary_a": _module("routed", "gate"),
        "arbitrary_b": _module("routed", "up"),
    }
    down = {"unrelated_name": _module("routed", "down")}
    replay = RoutedMoEInputReplayAttachment(maximum_retained_bytes=1024)
    first = torch.randn(2, 4)
    second = torch.randn(2, 4)

    assert routed_projection_roles(gate_up) == frozenset({"gate", "up"})
    replay.prepare_subset(gate_up, batch_count=2)
    replay.retain(1, second)
    replay.retain(0, first)
    replay.prepare_subset(down, batch_count=2)

    first_replay = replay.take(0)
    second_replay = replay.take(1)
    assert first_replay.data_ptr() == first.data_ptr()
    assert second_replay.data_ptr() == second.data_ptr()
    torch.testing.assert_close(first_replay, first, atol=0, rtol=0)
    torch.testing.assert_close(second_replay, second, atol=0, rtol=0)
    assert replay.take(0) is None


def test_routed_input_replay_fails_closed_when_incomplete_or_over_budget():
    gate_up = {"a": _module("routed", "gate"), "b": _module("routed", "up")}
    down = {"c": _module("routed", "down")}
    replay = RoutedMoEInputReplayAttachment(maximum_retained_bytes=16)

    replay.prepare_subset(gate_up, batch_count=2)
    replay.retain(0, torch.zeros(8, dtype=torch.float32))
    replay.prepare_subset(down, batch_count=2)

    assert replay.take(0) is None


def test_routed_input_replay_requires_exact_batch_indices_and_matching_consumer_count():
    gate_up = {"a": _module("routed", "gate"), "b": _module("routed", "up")}
    down = {"c": _module("routed", "down")}
    replay = RoutedMoEInputReplayAttachment(maximum_retained_bytes=1024)

    replay.prepare_subset(gate_up, batch_count=2)
    replay.retain(1, torch.ones(2))
    replay.retain(2, torch.ones(2))
    replay.prepare_subset(down, batch_count=2)
    assert replay.take(1) is None

    replay.prepare_subset(gate_up, batch_count=2)
    replay.retain(0, torch.ones(2))
    replay.retain(1, torch.ones(2))
    replay.prepare_subset(down, batch_count=3)
    assert replay.take(0) is None

    replay.prepare_subset(gate_up, batch_count=0)
    replay.prepare_subset(down, batch_count=0)
    assert replay.take(0) is None


def test_routed_input_replay_ignores_duplicates_and_clears_on_unrelated_subset():
    gate_up = {"a": _module("routed", "gate"), "b": _module("routed", "up")}
    unrelated = {"c": _module("shared", "down")}
    replay = RoutedMoEInputReplayAttachment(maximum_retained_bytes=1024)
    value = torch.arange(8)

    replay.prepare_subset(gate_up, batch_count=1)
    replay.retain(None, value)
    replay.retain(0, value)
    replay.retain(0, value + 1)
    replay.prepare_subset(unrelated, batch_count=1)

    assert replay.take(0) is None
    replay.abort()


def test_routed_input_replay_deduplicates_backing_storage_for_budget():
    gate_up = {"a": _module("routed", "gate"), "b": _module("routed", "up")}
    down = {"c": _module("routed", "down")}
    source = torch.arange(16, dtype=torch.float32)
    replay = RoutedMoEInputReplayAttachment(maximum_retained_bytes=source.untyped_storage().nbytes())

    replay.prepare_subset(gate_up, batch_count=2)
    replay.retain(0, source[:8])
    replay.retain(1, source[8:])
    replay.prepare_subset(down, batch_count=2)

    torch.testing.assert_close(replay.take(0), source[:8], atol=0, rtol=0)
    torch.testing.assert_close(replay.take(1), source[8:], atol=0, rtol=0)


def test_moe_lifecycle_input_replay_methods_delegate_to_attachment():
    hooks = GateUpDownMoELifecycleHooks()
    hooks.input_replay = Mock()
    hidden = torch.randn(1, 2, 3)
    hooks.input_replay.take.return_value = hidden
    subset = {"module": object()}

    hooks.prepare_input_replay(subset, 7)
    actual = hooks.take_input_replay(3)

    hooks.input_replay.prepare_subset.assert_called_once_with(subset, 7)
    hooks.input_replay.take.assert_called_once_with(3)
    assert actual is hidden
