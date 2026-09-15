# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Future

import pytest
import torch

from gptqmodel.looper.checkpoint import (
    CheckpointConfig,
    CheckpointExtension,
    CheckpointStopped,
    LoopPlan,
)
from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.looper.extension import LoopExtensions, LoopStep
from gptqmodel.looper.input_cache import InputCache

PLAN = LoopPlan(tuple(LoopStep("layer", i, f"model.layers.{i}") for i in range(3)))


def test_complete_input_cache_roundtrip():
    source = torch.arange(24).reshape(4, 6)
    cache = InputCache(
        [[source[:, ::2]]],
        [{"nested": (None, {7: source})}],
        [source[:1]],
        [torch.ones(4, dtype=torch.bool)],
        [[source + 5]],
    )
    state = {
        "cache": cache,
        "shared_kv": {2: (source, source + 1)},
        "rng": torch.random.get_rng_state(),
        "empty": (),
        "flag": False,
    }
    restored = ContinuationCodec.loads(ContinuationCodec.dumps(state))
    actual = restored["cache"]
    assert torch.equal(actual.layer_inputs[0][0], source[:, ::2])
    assert torch.equal(actual.src_inputs[0][0], source + 5)
    assert torch.equal(actual.position_ids[0], source[:1])
    assert torch.equal(actual.attention_masks[0], cache.attention_masks[0])
    assert actual.layer_input_kwargs[0]["nested"][0] is None
    assert torch.equal(actual.layer_input_kwargs[0]["nested"][1][7], source)
    assert torch.equal(restored["rng"], state["rng"])
    assert type(restored["shared_kv"][2]) is tuple
    assert restored["empty"] == () and restored["flag"] is False


@pytest.mark.parametrize(
    "state", [object(), float("nan"), {1.5: "invalid"}, torch.empty(1, device="meta")]
)
def test_codec_rejects_unsupported_state(state):
    with pytest.raises(TypeError):
        ContinuationCodec.dumps(state)


def test_codec_rejects_cycle():
    state = []
    state.append(state)
    with pytest.raises(TypeError, match="cyclic"):
        ContinuationCodec.dumps(state)


class Adapter:
    def __init__(self):
        self.value = torch.tensor([1.0])
        self.artifacts = {}
        self.captures = 0

    def capture(self):
        self.captures += 1
        return {"value": self.value}, self.artifacts

    def restore(self, state, artifacts):
        self.value = state["value"]
        self.artifacts = {name: path.read_bytes() for name, path in artifacts.items()}

    def execute(self, index):
        self.value = self.value * 2 + index
        self.artifacts[str(index)] = ContinuationCodec.dumps(self.value)


def test_resume_installs_continuation_without_replaying(tmp_path):
    uninterrupted = Adapter()
    for index in range(3):
        uninterrupted.execute(index)
    interrupted = Adapter()
    with CheckpointExtension(CheckpointConfig(tmp_path), interrupted) as extension:
        assert extension.prepare(PLAN, {"model": "test"}) == 0
        interrupted.execute(0)
        extension.request_stop()
        with pytest.raises(CheckpointStopped):
            LoopExtensions([extension]).publish(PLAN.steps[0])
    resumed = Adapter()
    with CheckpointExtension(
        CheckpointConfig(tmp_path, resume="required"), resumed
    ) as extension:
        assert extension.prepare(PLAN, {"model": "test"}) == 1
        dispatch = LoopExtensions([extension])
        for index in range(extension.next_step, len(PLAN.steps)):
            resumed.execute(index)
            dispatch.publish(PLAN.steps[index])
    assert torch.equal(resumed.value, uninterrupted.value)
    assert resumed.artifacts == uninterrupted.artifacts


def test_checkpoint_interval_and_final_boundary(tmp_path):
    adapter = Adapter()
    with CheckpointExtension(
        CheckpointConfig(tmp_path, interval="layer:2"), adapter
    ) as extension:
        extension.prepare(PLAN, {})
        dispatch = LoopExtensions([extension])
        for index, step in enumerate(PLAN.steps):
            adapter.execute(index)
            dispatch.publish(step)
        assert adapter.captures == 2
        assert extension.store.load(extension.identity)["cursor"] == 3


def test_failed_finalizer_never_captures(tmp_path):
    adapter = Adapter()
    failed = Future()
    failed.set_exception(ValueError("finalizer failed"))
    with CheckpointExtension(CheckpointConfig(tmp_path), adapter) as extension:
        extension.prepare(PLAN, {})
        with pytest.raises(ValueError, match="finalizer failed"):
            LoopExtensions([extension]).publish(PLAN.steps[0], [failed])
        assert adapter.captures == 0
        assert extension.store.load(extension.identity) is None


def test_cursor_rejects_missing_boundary(tmp_path):
    with CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension:
        extension.prepare(PLAN, {})
        with pytest.raises(CheckpointError, match="execution cursor"):
            LoopExtensions([extension]).publish(PLAN.steps[1])


def test_changed_plan_rejected(tmp_path):
    with CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension:
        extension.prepare(PLAN, {})
        LoopExtensions([extension]).publish(PLAN.steps[0])
    changed = LoopPlan(
        (LoopStep("input_embedding", 0, "model.embed_tokens"), *PLAN.steps)
    )
    with (
        CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension,
        pytest.raises(CheckpointError, match="execution_plan"),
    ):
        extension.prepare(changed, {})
