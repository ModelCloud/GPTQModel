import types

import torch

from gptqmodel.looper.module_looper import ModuleLooper


class _DummyModel:
    def __init__(self, compute_device_filter):
        self.support_batch_quantize = False
        self.quantize_config = types.SimpleNamespace(
            device=torch.device("cpu"),
            vram_strategy="exclusive",
            compute_device_filter=compute_device_filter,
            auto_forward_data_parallel=True,
            moe_routing_bypass=lambda: False,
        )
        self.layer_callback = None


def test_compute_device_filter_applies_to_quant_devices(monkeypatch):
    devices = [torch.device("cpu"), torch.device("meta")]
    captured = {}

    def fake_select_forward_devices(_base_device):
        return devices

    def compute_device_filter(candidates):
        captured["candidates"] = list(candidates)
        return [candidates[0]]

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    looper = ModuleLooper(model=_DummyModel(compute_device_filter), processors=[])

    assert captured["candidates"] == devices
    assert looper._quant_devices == [devices[0]]


def test_compute_device_filter_applies_to_forward_devices(monkeypatch):
    devices = [torch.device("cpu"), torch.device("meta")]

    def compute_device_filter(candidates):
        return [candidates[0]]

    looper = ModuleLooper(model=_DummyModel(compute_device_filter), processors=[])

    class DummyProcessor:
        num_batches = 2

        def _set_current_batch_index(self, _index):
            return None

    def fake_clone_module_for_devices(_module, target_devices, progress_callback=None):
        return {device: object() for device in target_devices}

    def fake_forward_batch_worker(*args, **kwargs):
        batch_idx = args[2]
        return batch_idx, None, None

    called_devices = []

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    def fake_submit(device, fn, *args, **kwargs):
        called_devices.append(device)
        return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.clone_module_for_devices",
        fake_clone_module_for_devices,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.forward_batch_worker",
        fake_forward_batch_worker,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit",
        fake_submit,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit_serial",
        fake_submit,
    )

    module = torch.nn.Linear(1, 1)
    processor = DummyProcessor()

    outputs = looper._run_forward_batches_parallel(
        module=module,
        processor=processor,
        layer_inputs=[[torch.zeros(1, 1)], [torch.zeros(1, 1)]],
        layer_input_kwargs=[{}, {}],
        position_ids=[],
        attention_masks=[torch.zeros(1, 1), torch.zeros(1, 1)],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=False,
        reuse_kv=False,
        devices=devices,
    )

    assert outputs == []
    assert called_devices
    assert torch.device("meta") not in called_devices
    assert all(device.type == "cpu" for device in called_devices)
