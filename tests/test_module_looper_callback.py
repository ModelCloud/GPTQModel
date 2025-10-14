import types

import pytest

from gptqmodel.looper.module_looper import ModuleLooper, StopMainLoop


class DummyQModel:
    def __init__(self):
        self.support_batch_quantize = False
        self.quantize_config = types.SimpleNamespace(device=None)
        self.layer_callback = None


def make_looper(layer_callback=None):
    model = DummyQModel()
    if layer_callback is not None:
        model.layer_callback = layer_callback
    processors = [types.SimpleNamespace()]
    return ModuleLooper(model=model, processors=processors)


def test_callbackup_invokes_model_layer_callback():
    calls = []

    class Recorder:
        def layer_complete(self, *, layer_idx, submodule_finalized):
            calls.append((layer_idx, submodule_finalized))

    looper = make_looper(layer_callback=Recorder())

    looper.callbackup(layer_idx=3, submodule_finalized=False)
    looper.callbackup(layer_idx=3, submodule_finalized=True)

    assert calls == [(3, False), (3, True)]


def test_callbackup_stop_request_via_returning_class():
    def stopper(**_):
        return StopMainLoop

    looper = make_looper(layer_callback=stopper)

    with pytest.raises(StopMainLoop):
        looper.callbackup(layer_idx=1, submodule_finalized=False)


def test_callbackup_stop_request_via_instance():
    def stopper(**_):
        return StopMainLoop("stop")

    looper = make_looper(layer_callback=stopper)

    with pytest.raises(StopMainLoop):
        looper.callbackup(layer_idx=1, submodule_finalized=False)


def test_emit_layer_complete_records_stop(monkeypatch):
    err = ValueError("boom")

    def raising_callback(*, layer_idx, submodule_finalized):
        raise err

    looper = make_looper(layer_callback=raising_callback)

    looper._emit_layer_complete(
        layer_idx=7,
        submodule_finalized=False,
        raise_in_place=False,
    )

    assert looper._loop_stop_exc is err
    assert looper._loop_stop_event.is_set()

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.wait",
        lambda *_, **__: None,
    )

    with pytest.raises(ValueError) as exc:
        looper._check_loop_stop()

    assert exc.value is err


def test_emit_layer_complete_propagates_when_requested():
    err = RuntimeError("direct")

    def raising_callback(*, layer_idx, submodule_finalized):
        raise err

    looper = make_looper(layer_callback=raising_callback)

    with pytest.raises(RuntimeError) as exc:
        looper._emit_layer_complete(
            layer_idx=2,
            submodule_finalized=True,
            raise_in_place=True,
        )

    assert exc.value is err


def test_emit_layer_complete_stops_cleanly_on_stop_main_loop(monkeypatch):
    class Stopper:
        def layer_complete(self, *, layer_idx, submodule_finalized):
            raise StopMainLoop()

    looper = make_looper(layer_callback=Stopper())

    looper._emit_layer_complete(
        layer_idx=0,
        submodule_finalized=True,
        raise_in_place=True,
    )

    assert looper._loop_stop_exc is None
    assert looper._loop_stop_event.is_set()

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.wait",
        lambda *_, **__: None,
    )

    assert looper._check_loop_stop() is True
