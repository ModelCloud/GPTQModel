# SPDX-License-Identifier: Apache-2.0
import threading

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.looper.eora_processor import EoraProcessor
from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.loop_processor import LoopProcessor


def processor(cls):
    obj = object.__new__(cls)
    obj.lock = threading.Lock()
    obj._results_lock = threading.Lock()
    obj._input_cache_lock = threading.RLock()
    obj._results = {}
    obj.receive_input_cache(InputCache([], [], [], []))
    obj.log = [{"layer": 0}]
    obj.num_batches = 4
    obj.total_calibration_tokens = 128
    obj.tasks = {"unfinished": object()}
    return obj


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_eora_completed_results_exact_roundtrip(device):
    if device.startswith("cuda") and torch.cuda.device_count() <= int(device[-1]):
        pytest.skip("CUDA device unavailable")
    original = processor(EoraProcessor)
    a = torch.tensor([[0.0, -0.0], [1.0, -2.0]], device=device)
    b = torch.arange(4, device=device, dtype=torch.float16).reshape(2, 2)
    original.result_save("model.layers.0.proj", Lora(rank=2, lora_A=a, lora_B=b))
    saved = ContinuationCodec.dumps(original.continuation_state_dict())
    restored = processor(EoraProcessor)
    restored.load_continuation_state_dict(ContinuationCodec.loads(saved))
    adapter = restored.results()["model.layers.0.proj"]
    for expected, actual in [(a, adapter.lora_A), (b, adapter.lora_B)]:
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert restored.num_batches == 4
    assert restored.total_calibration_tokens == 128
    assert restored.log == original.log
    assert adapter.rank == 2
    # Runtime objects are not serialized, copied, or resurrected.
    assert restored.tasks["unfinished"] is not original.tasks["unfinished"]


def test_base_processor_does_not_serialize_runtime_objects():
    original = processor(LoopProcessor)
    original.result_save("completed", torch.tensor([42]))
    state = ContinuationCodec.loads(
        ContinuationCodec.dumps(original.continuation_state_dict())
    )
    assert "tasks" not in state
    restored = processor(LoopProcessor)
    restored.load_continuation_state_dict(state)
    assert torch.equal(restored.results()["completed"], torch.tensor([42]))
