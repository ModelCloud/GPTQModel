# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.quantization import QVQConfig


_STAGED_QVQ_KEYS = {"trellis", "SU", "SV", "bias", "_qvq_original_weight", "_qvq_runtime_config"}


def _prepared_calibration(**kwargs):
    return kwargs["calibration_dataset"]


def _processor():
    calibration = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }
    ]
    return QVQProcessor(
        tokenizer=None,
        qcfg=QVQConfig(bits=2, rounding="block_ldlq", device="cpu", offload_to_disk=False),
        calibration=calibration,
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )


def _captured_linear():
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float32)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    processor = _processor()
    processor.preprocess(named)
    source = torch.randn((1, 4, 16), dtype=torch.float32)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.ones((1, 4), dtype=torch.bool)
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook("proj")(root.proj, (source,), root.proj(source))
    return root, named, processor, processor.tasks["proj"]["capture"]


def test_qvq_encode_failure_releases_hessian_and_does_not_leave_staged_payload():
    torch.manual_seed(20260812)
    root, named, processor, capture = _captured_linear()
    original_weight = root.proj.weight.detach().clone()

    with patch("gptqmodel.looper.qvq_processor.quantize_qvq_linear", side_effect=RuntimeError("encode failed")):
        with pytest.raises(RuntimeError, match="encode failed"):
            processor.process(named, device=torch.device("cpu"))

    assert not hasattr(capture, "H")
    assert not hasattr(capture, "quantizer")
    assert not _STAGED_QVQ_KEYS.intersection(named.state)
    torch.testing.assert_close(root.proj.weight, original_weight, rtol=0, atol=0)


def test_qvq_stream_failure_removes_partial_payload_and_releases_hessian():
    root, named, processor, capture = _captured_linear()
    result = SimpleNamespace(
        trellis=torch.zeros((1, 16), dtype=torch.int32),
        SU=torch.ones(16),
        SV=torch.ones(16),
        bias=None,
        weight=root.proj.weight.detach().clone(),
        proxy_loss=torch.tensor(0.0),
        output_scale_optimized_channels=0,
        hessian_viterbi_selected=False,
    )
    result.serialized_tensors = lambda: {
        "trellis": result.trellis,
        "SU": result.SU,
        "SV": result.SV,
        "bias": result.bias,
    }

    def fail_after_partial_store(self, tensors):
        self.state.update(tensors)
        raise RuntimeError("host staging failed")

    with (
        patch("gptqmodel.looper.qvq_processor.quantize_qvq_linear", return_value=result),
        patch.object(NamedModule, "stream_state_payload_to_cpu", fail_after_partial_store),
        pytest.raises(RuntimeError, match="host staging failed"),
    ):
        processor.process(named, device=torch.device("cpu"))

    assert not hasattr(capture, "H")
    assert not hasattr(capture, "quantizer")
    assert not _STAGED_QVQ_KEYS.intersection(named.state)


def test_qvq_cleanup_failures_do_not_mask_original_quantization_error():
    _, named, processor, capture = _captured_linear()

    with (
        patch("gptqmodel.looper.qvq_processor.quantize_qvq_linear", side_effect=RuntimeError("encode failed")),
        patch.object(NamedModule, "stream_sync", side_effect=RuntimeError("sync cleanup failed")),
        patch.object(QVQProcessor, "_restore_module_weight", side_effect=RuntimeError("restore cleanup failed")),
        pytest.raises(RuntimeError, match="encode failed"),
    ):
        processor.process(named, device=torch.device("cpu"))

    assert not hasattr(capture, "H")
    assert not hasattr(capture, "quantizer")
    assert not _STAGED_QVQ_KEYS.intersection(named.state)


@pytest.mark.parametrize("invalid_state", ("missing_hessian", "zero_samples"))
def test_qvq_prequantization_validation_still_releases_capture(invalid_state):
    _, named, processor, capture = _captured_linear()

    def leave_invalid_state(*, target_device):
        del target_device
        if invalid_state == "missing_hessian":
            capture.H = None
        else:
            capture.H = torch.eye(16)
            capture.nsamples = 0

    with (
        patch.object(capture, "finalize_hessian", side_effect=leave_invalid_state),
        pytest.raises(RuntimeError, match="failed to capture Hessian|no calibration activations"),
    ):
        processor.process(named, device=torch.device("cpu"))

    assert not hasattr(capture, "H")
    assert not hasattr(capture, "quantizer")
