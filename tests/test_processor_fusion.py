# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.looper.processor_fusion import (
    ArtifactScope,
    ExecutionStage,
    InlineConfig,
    ProcessorFusionPlan,
    ProcessorStages,
    compose_forward_hooks,
    plan_processor_fusion,
)


class FakeProcessor:
    emits = ()
    stages = ()
    inline = None

    def __init__(self, name, *, emits=(), stages=(), inline=None, batches=4):
        self._name = name
        self.emits = tuple(emits)
        self.stages = tuple(stages)
        self.inline = inline
        self.num_batches = batches
        self.inline_processors = ()

    def name(self):
        return self._name


def _inline_config(producer=("gptq",), consumes=("wq", "w")):
    return InlineConfig(
        producer=frozenset(producer),
        stage=ProcessorStages.POST_QUANTIZE,
        forward_hooks=True,
        consumes=consumes,
    )


def test_plan_processor_fusion_attaches_consumer_to_earlier_producer():
    producer = FakeProcessor(
        "gptq", emits=("wq", "w"), stages=(ProcessorStages.POST_QUANTIZE,)
    )
    consumer = FakeProcessor("eora", inline=_inline_config())

    plan = plan_processor_fusion([producer, consumer])

    assert isinstance(plan, ProcessorFusionPlan)
    assert plan.fused == (consumer,)
    assert len(plan.stages) == 1
    assert plan.stages[0].processor is producer
    assert plan.stages[0].inline == (consumer,)
    assert producer.inline_processors == (consumer,)


def test_plan_processor_fusion_falls_back_when_artifact_is_not_emitted():
    producer = FakeProcessor("gptq", emits=("wq",), stages=(ProcessorStages.POST_QUANTIZE,))
    consumer = FakeProcessor("eora", inline=_inline_config())

    plan = plan_processor_fusion([producer, consumer])

    assert plan.fused == ()
    assert [stage.processor for stage in plan.stages] == [producer, consumer]
    assert producer.inline_processors == ()


def test_plan_processor_fusion_falls_back_when_slot_is_not_opened():
    producer = FakeProcessor(
        "gptq", emits=("wq", "w"), stages=(ProcessorStages.POST_INSTALL,)
    )
    consumer = FakeProcessor("eora", inline=_inline_config())

    plan = plan_processor_fusion([producer, consumer])

    assert plan.fused == ()
    assert len(plan.stages) == 2


def test_plan_processor_fusion_respects_consumer_compatibility():
    class BatchSensitive(FakeProcessor):
        def can_fuse_with(self, producer):
            return self.num_batches == producer.num_batches

    producer = FakeProcessor(
        "gptq", emits=("wq", "w"), stages=(ProcessorStages.POST_QUANTIZE,), batches=8
    )
    consumer = BatchSensitive("eora", inline=_inline_config(), batches=4)

    plan = plan_processor_fusion([producer, consumer])

    assert plan.fused == ()
    assert len(plan.stages) == 2


def test_plan_processor_fusion_disabled_keeps_every_processor_separate():
    producer = FakeProcessor(
        "gptq", emits=("wq", "w"), stages=(ProcessorStages.POST_QUANTIZE,)
    )
    consumer = FakeProcessor("eora", inline=_inline_config())

    plan = plan_processor_fusion([producer, consumer], enabled=False)

    assert plan.fused == ()
    assert plan.stages == (ExecutionStage(producer), ExecutionStage(consumer))
    assert producer.inline_processors == ()


def test_scope_is_lazy_and_materializes_once():
    calls = []

    def recipe():
        calls.append(1)
        return object()

    scope = ArtifactScope("producer", None, ProcessorStages.POST_QUANTIZE)
    scope.publish("w", recipe)

    assert calls == []
    first = scope.get("w")
    second = scope.get("w")
    assert first is second
    assert calls == [1]
    assert scope.materialized == 1


def test_scope_replace_wins_over_recipe():
    scope = ArtifactScope("producer", None, ProcessorStages.POST_QUANTIZE)
    scope.publish("wq", lambda: "original")
    assert scope.get("wq") == "original"
    scope.replace("wq", "corrected")
    assert scope.get("wq") == "corrected"


def test_scope_rejects_access_after_exit():
    scope = ArtifactScope("producer", None, ProcessorStages.POST_QUANTIZE)
    with scope:
        scope.publish("wq", lambda: "value")
        assert scope.get("wq") == "value"
    with pytest.raises(RuntimeError, match="closed"):
        scope.get("wq")


def test_scope_unknown_artifact_is_explicit():
    scope = ArtifactScope("producer", None, ProcessorStages.POST_QUANTIZE)
    with pytest.raises(KeyError, match="not published"):
        scope.get("missing")


def test_scope_detach_is_counted_and_independent():
    class Value:
        def __init__(self, data):
            self.data = list(data)

        def clone(self):
            return Value(self.data)

    scope = ArtifactScope("producer", None, ProcessorStages.POST_QUANTIZE)
    scope.publish("wq", lambda: Value([1, 2]))
    kept = scope.detach("wq")
    assert kept.data == [1, 2]
    assert scope.detached == 1


def test_processor_contracts_are_declared():
    from gptqmodel.looper.eora_processor import EoraProcessor
    from gptqmodel.looper.gptq_processor import GPTQProcessor

    # GPTQ publishes only the weights it owns; the residual is EoRA's.
    assert GPTQProcessor.emits == ("wq", "w")
    assert "w_wq_diff" not in GPTQProcessor.emits
    assert ProcessorStages.POST_QUANTIZE in GPTQProcessor.stages
    assert EoraProcessor.inline is not None
    assert EoraProcessor.inline.stage is ProcessorStages.POST_QUANTIZE
    assert EoraProcessor.inline.consumes == ("wq", "w")
    assert EoraProcessor.inline.forward_hooks is True


def test_eora_fuses_only_with_the_same_calibration_dataset():
    from gptqmodel.looper.eora_processor import EoraProcessor

    class Calibration:
        def __init__(self, dataset):
            self.dataset = dataset

    dataset = [{"input_ids": [1, 2, 3]}]
    consumer = type("Consumer", (), {"calibration": Calibration(dataset)})()
    same = type("Producer", (), {"calibration": Calibration(dataset)})()
    other = type("Producer", (), {"calibration": Calibration([{"input_ids": [4]}])})()

    assert EoraProcessor.can_fuse_with(consumer, same) is True
    assert EoraProcessor.can_fuse_with(consumer, other) is False


def _hook(events, label):
    def hook(module, inputs, output):
        events.append(label)

    return hook


def test_composed_hook_captures_before_the_early_stop():
    """HookedLinear raises the stop from inside forward(), after forward_hook."""

    events = []

    class Producer:
        name = "gptq"
        inline = None

        def pre_process_fwd_hook(self, name):
            return _hook(events, "producer")

    class Consumer:
        name = "eora"
        inline = InlineConfig(
            producer=frozenset({"gptq"}),
            stage=ProcessorStages.POST_QUANTIZE,
            forward_hooks=True,
            consumes=("wq", "w"),
        )

        def is_skipped(self, module):
            return False

        def pre_process_fwd_hook(self, name):
            return _hook(events, "consumer")

    producer, consumer = Producer(), Consumer()
    combined = compose_forward_hooks(
        producer,
        producer.pre_process_fwd_hook("proj"),
        object(),
        "proj",
        (consumer,),
    )

    class StopForward(Exception):
        pass

    def hooked_linear_forward():
        combined(None, (object(),), object())
        raise StopForward

    with pytest.raises(StopForward):
        hooked_linear_forward()

    assert events == ["producer", "consumer"]


def test_composed_hook_skips_disabled_consumers():
    events = []

    class Producer:
        name = "gptq"
        inline = None

        def pre_process_fwd_hook(self, name):
            return _hook(events, "producer")

    class Consumer:
        name = "eora"
        inline = InlineConfig(
            producer=frozenset({"gptq"}),
            stage=ProcessorStages.POST_QUANTIZE,
            forward_hooks=True,
            consumes=("wq", "w"),
        )

        def is_skipped(self, module):
            return True

        def pre_process_fwd_hook(self, name):
            return _hook(events, "consumer")

    combined = compose_forward_hooks(
        Producer(), _hook(events, "producer"), object(), "proj", (Consumer(),)
    )
    combined(None, (object(),), object())

    assert events == ["producer"]


def test_composed_hook_copies_the_producer_batch_index():
    seen = []

    class Producer:
        name = "gptq"
        inline = None

        def current_batch_index(self):
            return 3

    class Consumer:
        name = "eora"
        inline = InlineConfig(
            producer=frozenset({"gptq"}),
            stage=ProcessorStages.POST_QUANTIZE,
            forward_hooks=True,
            consumes=("wq", "w"),
        )

        def is_skipped(self, module):
            return False

        def _set_current_batch_index(self, index):
            seen.append(index)

        def pre_process_fwd_hook(self, name):
            return _hook([], "consumer")

    combined = compose_forward_hooks(
        Producer(), None, object(), "proj", (Consumer(),)
    )
    combined(None, (object(),), object())

    assert seen == [3]
