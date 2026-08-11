# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""End-to-end hook/Hessian contracts for padded calibration batches."""

import threading

import pytest
import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig


def _processor_for(module: torch.nn.Module, name: str, device: torch.device) -> tuple[GPTQProcessor, NamedModule]:
    """Create one real GPTQ hook task without model-loading lifecycle noise."""

    qcfg = QuantizeConfig(
        bits=4,
        group_size=8,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        hessian=HessianConfig(staging_dtype=torch.float32),
        enable_shared_hessian_cache=False,
    )
    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    # These focused hook tests do not enter the full looper lifecycle, whose
    # finalize step normally closes telemetry handles.
    processor._close_device_smi_handles()
    named = NamedModule(module.to(device), name=name, full_name=f"model.layers.0.{name}", layer_index=0)
    processor.preprocess(named)
    processor.prepare_subset({name: named}, subset_index=0, subset_total=1)
    processor._mask_tls = threading.local()
    return processor, named


def _capture(
    processor: GPTQProcessor,
    named: NamedModule,
    inputs: torch.Tensor,
    keep: torch.Tensor,
) -> None:
    processor._mask_tls.value = keep
    output = named.module(inputs)
    processor.pre_process_fwd_hook(named.name)(named.module, (inputs,), output)


def _assert_masked_batched_hessian_matches_same_samples_serially(device: torch.device, seed: int) -> None:
    torch.manual_seed(seed)
    activations = torch.randn((4, 5, 8), dtype=torch.float32, device=device)
    keep = torch.tensor(
        [
            [True, True, True, False, False],
            [False, False, True, True, True],
            [True, False, True, False, True],
            [False, False, False, False, False],
        ]
    )
    activations[~keep] = 10_000.0
    keep = keep.to(device)
    weight = torch.randn((8, 8), dtype=torch.float32, device=device)

    batched, batched_named = _processor_for(torch.nn.Linear(8, 8, bias=False), "proj", device)
    serial, serial_named = _processor_for(torch.nn.Linear(8, 8, bias=False), "proj", device)
    batched_named.module.weight.data.copy_(weight)
    serial_named.module.weight.data.copy_(weight)

    _capture(batched, batched_named, activations, keep)
    for sample_index in range(activations.shape[0]):
        _capture(
            serial,
            serial_named,
            activations[sample_index : sample_index + 1],
            keep[sample_index : sample_index + 1],
        )

    batched_hessian = batched.tasks["proj"].finalize_hessian(target_device=device)
    serial_hessian = serial.tasks["proj"].finalize_hessian(target_device=device)
    selected = torch.cat([activations[index, keep[index]] for index in range(4)], dim=0)
    reference = selected.T @ selected * (2.0 / selected.shape[0])

    assert batched.tasks["proj"].nsamples == int(keep.sum().item())
    assert serial.tasks["proj"].nsamples == int(keep.sum().item())
    torch.testing.assert_close(batched_hessian, serial_hessian, rtol=0, atol=0)
    # CUDA addmm and the separately expressed matmul reference can select
    # different kernels and differ by a few FP32 ULPs. The batch-vs-serial
    # assertion above remains exact because both traverse the production path.
    reference_atol = 4 * torch.finfo(torch.float32).eps if device.type == "cuda" else 0
    torch.testing.assert_close(batched_hessian, reference, rtol=2e-6, atol=reference_atol)


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("seed", [0, 123, 997])
def test_masked_batched_hessian_matches_same_samples_serially(device_name, seed):
    """Partitioning identical activations into a batch must preserve exact FP32 XtX math."""

    _assert_masked_batched_hessian_matches_same_samples_serially(torch.device(device_name), seed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("seed", [0, 123, 997])
def test_masked_batched_hessian_is_stable_across_repeated_cuda_launches(seed):
    """Repeated free-threaded CUDA launches must preserve the exact batching contract."""

    for _ in range(10):
        _assert_masked_batched_hessian_matches_same_samples_serially(torch.device("cuda"), seed)


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_embedding_hessian_excludes_masked_padding_when_pad_equals_eos(device_name):
    """Embedding frequency must count a valid EOS but not identical masked pad IDs."""

    device = torch.device(device_name)
    processor, named = _processor_for(torch.nn.Embedding(16, 8), "embed_tokens", device)
    token_ids = torch.tensor([[3, 7, 7, 7], [4, 5, 7, 7]], device=device)
    keep = torch.tensor([[True, True, False, False], [True, True, False, False]], device=device)

    _capture(processor, named, token_ids, keep)

    task = processor.tasks["embed_tokens"]
    counts = task._device_embedding_counts[token_ids.device]
    assert task.nsamples == 4
    assert counts[3].item() == 1
    assert counts[4].item() == 1
    assert counts[5].item() == 1
    assert counts[7].item() == 1
    assert counts.sum().item() == 4


def test_all_masked_embedding_records_zero_samples_for_rtn_fallback():
    processor, named = _processor_for(torch.nn.Embedding(16, 8), "embed_tokens", torch.device("cpu"))
    token_ids = torch.tensor([[7, 7, 7]])
    keep = torch.zeros_like(token_ids, dtype=torch.bool)

    _capture(processor, named, token_ids, keep)

    task = processor.tasks["embed_tokens"]
    assert task.nsamples == 0
    assert task._device_embedding_counts == {}
    quantized_weight, *_rest, nsamples = task.quantize()
    assert nsamples == 0
    assert quantized_weight.shape == named.module.weight.shape
    assert torch.isfinite(quantized_weight).all()


def test_flattened_linear_input_uses_unmasked_hessian_path():
    processor, named = _processor_for(torch.nn.Linear(8, 8, bias=False), "proj", torch.device("cpu"))
    activations = torch.randn((5, 8), dtype=torch.float32)

    _capture(processor, named, activations, torch.ones((1, 5), dtype=torch.bool))

    assert processor.tasks["proj"].nsamples == 5
