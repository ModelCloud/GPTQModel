from types import MethodType

import torch

import gptqmodel.utils.looper_helpers as looper_helpers


def test_rebinds_shallow_copied_instance_method_to_replica_node():
    class Router(torch.nn.Module):
        pass

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = Router()

    def patched_forward(self, value):
        return self, value

    source = Layer()
    source.router.forward = MethodType(patched_forward, source.router)
    replica = Layer()
    replica.router.forward = source.router.forward

    looper_helpers._rebind_copied_instance_methods(source, replica)

    owner, value = replica.router.forward("ok")
    assert owner is replica.router
    assert value == "ok"


class _DummyProcessor:
    def __init__(self):
        self.current_batch_index = None
        self._mask_tls = None

    def _set_current_batch_index(self, batch_index):
        self.current_batch_index = batch_index


class _DynamicConfig:
    lm_head = False

    def dynamic_get(self, *, layer_name):
        return False if layer_name.startswith(("layers.1.", "layers.2.")) else None


def test_find_last_quantized_layer_index_uses_dynamic_exclusions():
    assert looper_helpers.find_last_quantized_layer_index(
        _DynamicConfig(),
        layer_modules=[["linear#capture_only"]],
        layer_names=["layers.0", "layers.1", "layers.2"],
        layer_count=3,
    ) == 0


def test_plan_layer_count_truncates_trailing_excluded_layers():
    assert looper_helpers.plan_layer_count(4, 1) == 2


def test_plan_layer_count_keeps_every_layer_without_exclusions():
    assert looper_helpers.plan_layer_count(4, None) == 4


def test_plan_layer_count_covers_replayed_layers_for_embedding_modes():
    # `run_layer_stage()` replays and publishes every trailing decoder layer
    # when an embedding endpoint is requantized, so the plan must not truncate.
    assert looper_helpers.plan_layer_count(4, 1, replay_all_layers=True) == 4


def test_plan_accepts_replayed_boundary_after_last_quantized_layer():
    from gptqmodel.looper.extension import LoopPlan, LoopStep

    # Two-layer model where layer 1 is fully excluded, plus an output embedding
    # step. Layer 1 is still replayed and published, so `plan.after()` must
    # resolve it instead of raising `ValueError: tuple.index(x): x not in tuple`.
    steps = tuple(
        LoopStep("layer", index, f"model.layers.{index}")
        for index in range(
            looper_helpers.plan_layer_count(2, 0, replay_all_layers=True)
        )
    ) + (LoopStep("output_embeddings", 2, "output_embeddings"),)
    plan = LoopPlan(steps)

    assert plan.after(LoopStep("layer", 1, "model.layers.1")) == 2


class _RequiresAttentionMask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, attention_mask, use_cache=False):
        assert attention_mask is None
        return self.proj(hidden_states)


class _IgnoresAttentionMask(torch.nn.Module):
    def forward(self, hidden_states, use_cache=False):
        return hidden_states + 1


class _ReturnsAuxState(torch.nn.Module):
    def forward(self, hidden_states, use_cache=False):
        aux_state = hidden_states + 2
        return hidden_states + 1, aux_state


class _RecordsForward(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, hidden_states, use_cache=False):
        self.events.append(("forward", hidden_states.device))
        return hidden_states + 1


def test_forward_batch_worker_waits_for_device_before_future_completion(monkeypatch):
    events = []
    processor = _DummyProcessor()
    module = _RecordsForward(events)
    hidden_states = torch.randn(1, 2, 4)
    monkeypatch.setattr(
        looper_helpers,
        "torch_sync",
        lambda device=None: events.append(("sync", device)),
    )

    looper_helpers.forward_batch_worker(
        module=module,
        processor=processor,
        batch_index=0,
        layer_input=[hidden_states],
        layer_input_kwargs={},
        attention_mask=None,
        position_ids=None,
        support_batch_quantize=True,
        is_lm_head_module=False,
        need_output=True,
        reuse_kv=False,
        prev_kv=None,
    )

    assert events == [
        ("sync", hidden_states.device),
        ("forward", hidden_states.device),
        ("sync", hidden_states.device),
    ]


def test_forward_batch_worker_passes_none_attention_mask_when_module_requires_it():
    processor = _DummyProcessor()
    module = _RequiresAttentionMask()
    hidden_states = torch.randn(1, 2, 4)

    batch_index, module_output, kv_next = looper_helpers.forward_batch_worker(
        module=module,
        processor=processor,
        batch_index=3,
        layer_input=[hidden_states],
        layer_input_kwargs={},
        attention_mask=None,
        position_ids=None,
        support_batch_quantize=True,
        is_lm_head_module=False,
        need_output=True,
        reuse_kv=False,
        prev_kv=None,
    )

    assert batch_index == 3
    assert module_output.shape == hidden_states.shape
    assert kv_next is None
    assert processor.current_batch_index is None


def test_forward_batch_worker_skips_attention_mask_for_modules_without_the_kwarg():
    processor = _DummyProcessor()
    module = _IgnoresAttentionMask()
    hidden_states = torch.randn(1, 2, 4)

    batch_index, module_output, kv_next = looper_helpers.forward_batch_worker(
        module=module,
        processor=processor,
        batch_index=1,
        layer_input=[hidden_states],
        layer_input_kwargs={},
        attention_mask=None,
        position_ids=None,
        support_batch_quantize=True,
        is_lm_head_module=False,
        need_output=True,
        reuse_kv=False,
        prev_kv=None,
    )

    assert batch_index == 1
    torch.testing.assert_close(module_output, hidden_states + 1)
    assert kv_next is None
    assert processor.current_batch_index is None


def test_forward_batch_worker_only_returns_primary_tensor_for_tuple_outputs():
    processor = _DummyProcessor()
    module = _ReturnsAuxState()
    hidden_states = torch.randn(1, 2, 4)

    batch_index, module_output, kv_next = looper_helpers.forward_batch_worker(
        module=module,
        processor=processor,
        batch_index=2,
        layer_input=[hidden_states],
        layer_input_kwargs={},
        attention_mask=None,
        position_ids=None,
        support_batch_quantize=True,
        is_lm_head_module=False,
        need_output=True,
        reuse_kv=True,
        prev_kv=None,
    )

    assert batch_index == 2
    assert isinstance(module_output, torch.Tensor)
    torch.testing.assert_close(module_output, hidden_states + 1)
    torch.testing.assert_close(kv_next, hidden_states + 2)
    assert processor.current_batch_index is None
