import threading

import pytest
import torch

from gptqmodel.eora.eora import eora_process_input
from gptqmodel.looper.eora_processor import EoraProcessor


def _processor_for_uniformity():
    processor = object.__new__(EoraProcessor)
    processor.lock = threading.Lock()
    processor._segment_accumulators = {"layer.proj": {}}
    processor._module_target_devices = {"layer.proj": torch.device("cpu")}
    processor.num_batches = 16
    processor._batch_tls = threading.local()
    processor.eora_process_input = eora_process_input
    return processor


class _FakeModule:
    def __init__(self, device):
        self.weight = type("_Weight", (), {"data": torch.empty(0, device=device)})()


def test_eora_covariance_weights_every_batch_uniformly():
    torch.manual_seed(0)
    cols = 8
    batches = [
        torch.randn(1, 3, cols, dtype=torch.float32)
        for _ in range(16)
    ]
    processor = _processor_for_uniformity()
    hook = processor.pre_process_fwd_hook("layer.proj")
    module = _FakeModule(torch.device("cpu"))

    for batch_index, sample in enumerate(batches):
        processor._set_current_batch_index(batch_index)
        hook(module, (sample,), None)

    actual = processor._finalize_eigen_scaling_matrix("layer.proj")
    tokens = torch.cat(batches, dim=1).reshape(-1, cols).double()
    oracle = tokens.T @ tokens / tokens.shape[0]

    torch.testing.assert_close(actual.double(), oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_eora_process_input_preserves_device():
    device = torch.device("cuda", 0)
    sample = torch.randn(1, 4, 6, device=device, dtype=torch.float16)

    batch, contribution, sample_count = eora_process_input(
        input=(sample,),
        name="test",
        sample_size=32,
        device=device,
    )

    assert batch == 1
    assert sample_count == 4
    assert contribution.device == device
    assert contribution.dtype == torch.float32
