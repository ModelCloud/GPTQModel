from types import SimpleNamespace

import pytest

from gptqmodel.looper.qqq_processor import QQQProcessor


def test_qqq_processor_preserves_diagnostics_before_free():
    diagnostics = {"before": 1.0, "after": 0.9, "history": [1.0, 0.9]}
    module = SimpleNamespace(name="projection", state={})

    class StopAfterCleanup(Exception):
        pass

    class Quantizer:
        gsq_diagnostics = diagnostics

        def quantize(self):
            return (None,) * 9

        def free(self):
            del self.gsq_diagnostics
            raise StopAfterCleanup

    quantizer = Quantizer()
    quantizer.gsq_diagnostics = diagnostics
    processor = object.__new__(QQQProcessor)
    processor.tasks = {module.name: quantizer}
    processor.draw_progress = lambda *args: None
    with pytest.raises(StopAfterCleanup):
        processor.process(module)
    assert module.state["gsq_diagnostics"] == diagnostics
