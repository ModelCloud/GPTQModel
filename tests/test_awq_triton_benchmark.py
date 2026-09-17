from gptqmodel.quantization.awq.modules.triton.scheduler import AwqTritonPlan
from scripts.benchmark_awq_triton_fp32_ab import _module_forward


class _FakeModule:
    def __init__(self):
        self.calls = []

    def __call__(self, entry):
        self.calls.append((self.awq_triton_schedule_mode, self.awq_triton_schedule))
        return entry


def test_module_benchmark_sets_and_clears_explicit_schedule():
    module = _FakeModule()
    candidate = AwqTritonPlan("fused", 16, 64, 64, 1, 4, 2)

    assert _module_forward(module, "selected", "auto", candidate) == "selected"
    assert _module_forward(module, "legacy", "legacy", None) == "legacy"
    assert module.calls == [("auto", candidate), ("legacy", None)]
