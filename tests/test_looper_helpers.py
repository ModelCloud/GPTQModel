import pytest
import torch

import gptqmodel.utils.looper_helpers as looper_helpers


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_clone_module_for_cuda_devices_avoids_nccl_broadcast(monkeypatch):
    """Replica staging must not enter torch replicate/NCCL broadcast_coalesced."""

    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    torch.manual_seed(17)
    module = (
        torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32, bias=False),
        )
        .to(devices[0])
        .eval()
    )
    held_out = torch.randn(4, 32)
    with torch.inference_mode():
        reference = module(held_out.to(devices[0])).cpu()
    broadcast_attempts = []

    def reject_broadcast(*args, **kwargs):
        broadcast_attempts.append((args, kwargs))
        raise AssertionError("NCCL broadcast_coalesced must not be used for module cloning")

    monkeypatch.setattr(torch._C, "_broadcast_coalesced", reject_broadcast)
    clones = looper_helpers.clone_module_for_devices(module, devices)

    assert broadcast_attempts == []
    assert list(clones) == devices
    for device, replica in clones.items():
        assert replica is not module
        assert getattr(replica, "_gptqmodule_device_hint") == device
        assert all(parameter.device == device for parameter in replica.parameters())
        with torch.inference_mode():
            actual = replica(held_out.to(device)).cpu()
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)

    for source_parameter, first_parameter, second_parameter in zip(
        module.parameters(), clones[devices[0]].parameters(), clones[devices[1]].parameters()
    ):
        assert source_parameter.data_ptr() != first_parameter.data_ptr()
        assert first_parameter.data_ptr() != second_parameter.data_ptr()

    torch.cuda.synchronize()


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
    assert (
        looper_helpers.find_last_quantized_layer_index(
            _DynamicConfig(),
            layer_modules=[["linear#capture_only"]],
            layer_names=["layers.0", "layers.1", "layers.2"],
            layer_count=3,
        )
        == 0
    )


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
        is_embeddings_module=False,
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
        is_embeddings_module=False,
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
        is_embeddings_module=False,
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
        is_embeddings_module=False,
        need_output=True,
        reuse_kv=True,
        prev_kv=None,
    )

    assert batch_index == 2
    assert isinstance(module_output, torch.Tensor)
    torch.testing.assert_close(module_output, hidden_states + 1)
    torch.testing.assert_close(kv_next, hidden_states + 2)
    assert processor.current_batch_index is None
