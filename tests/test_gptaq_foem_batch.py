# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ/FOEM consume one native calibration entry per sample.

`NativeProcessor` captures one entry per forward *batch*, while `GPTQProcessor`
calls `add_batch` once per *sample*. Before the split, sample 0 consumed the
whole batch tensor and later samples hit an empty list. These tests pin the
per-sample split against the batch_size=1 flow over the same samples.
"""

import threading

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY, NativeProcessor
from gptqmodel.quantization.config import CalibrationConfig, FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


def _qcfg(kind):
    if kind == "gptaq":
        return QuantizeConfig(bits=4, group_size=8, gptaq=GPTAQConfig())
    return QuantizeConfig(bits=4, group_size=8, foem=FOEMConfig(alpha=0.25))


def _cls(kind):
    return GPTAQ if kind == "gptaq" else FOEM


def _run(kind, module, batches):
    """Capture ``batches`` forwards, then replay them through the quantizer."""

    qcfg = _qcfg(kind)
    named = NamedModule(module, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor = NativeProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(dataset=None, batch_size=2),
    )
    processor.preprocess(named)
    hook = processor.pre_process_fwd_hook(named.name)
    for x, keep_mask in batches:
        # ModuleLooper stores the active [B, S] keep-mask on the processor's
        # thread-local storage before every forward pass.
        processor._mask_tls = threading.local()
        processor._mask_tls.value = keep_mask
        hook(module, (x,), None)
    processor.process(named)

    quantizer = _cls(kind)(named, qcfg)
    for x, keep_mask in batches:
        for sample_index in range(x.shape[0]):
            if not keep_mask[sample_index].any():
                continue
            # GPTQProcessor slices the same per-sample keep-mask before it
            # calls add_batch.
            sample = x[sample_index : sample_index + 1][:, keep_mask[sample_index]]
            quantizer.add_batch(sample, None)
    return quantizer


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_batched_native_capture_matches_per_sample_reference(kind):
    torch.manual_seed(0)
    module = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 3, 8)
    keep_mask = torch.tensor([[True, True, True], [True, True, False]])

    batched = _run(kind, module, [(x, keep_mask)])
    reference = _run(
        kind,
        module,
        [(x[0:1], keep_mask[0:1]), (x[1:2], keep_mask[1:2])],
    )

    torch.testing.assert_close(batched.H, reference.H, rtol=1e-6, atol=1e-6)
    if kind == "gptaq" or batched.gptaq:
        assert batched.dXXT is not None
        torch.testing.assert_close(batched.dXXT, reference.dXXT, rtol=1e-6, atol=1e-6)
    assert batched.nsamples == reference.nsamples
    assert torch.isfinite(batched.H).all()


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_native_capture_skips_empty_samples_like_gptq_processor(kind):
    torch.manual_seed(1)
    module = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 2, 8)
    keep_mask = torch.tensor([[True, True], [False, False], [True, False]])

    quantizer = _run(kind, module, [(x, keep_mask)])

    # GPTQProcessor skips samples with no kept tokens, so the native list must
    # skip them too or every later sample would read the wrong entry.
    assert torch.isfinite(quantizer.H).all()
    assert quantizer.H.shape == (8, 8)


def _run_flattened(kind, module, batches):
    """Capture flattened ``[B*S, H]`` forwards, then replay GPTQProcessor's 2-D split.

    HF qwen2_moe calls its ``shared_expert`` on ``hidden_states.view(-1, H)``.
    GPTQProcessor's hook then slices ``inp[b*S:(b+1)*S][keep[b]]`` and calls
    ``add_batch`` once per kept sample, so the native capture must produce the
    same per-sample entries.
    """

    qcfg = _qcfg(kind)
    named = NamedModule(module, name="shared_expert.gate_proj", full_name="model.layers.0.mlp.shared_expert.gate_proj", layer_index=0)
    processor = NativeProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(dataset=None, batch_size=4),
    )
    processor.preprocess(named)
    hook = processor.pre_process_fwd_hook(named.name)
    for batch_index, (x, keep_mask) in enumerate(batches):
        processor._mask_tls = threading.local()
        processor._mask_tls.value = keep_mask
        processor._set_current_batch_index(batch_index)
        hook(module, (x.reshape(-1, x.shape[-1]),), None)
    processor.process(named)

    quantizer = _cls(kind)(named, qcfg)
    for batch_index, (x, keep_mask) in enumerate(batches):
        flat = x.reshape(-1, x.shape[-1])
        seq_len = keep_mask.shape[1]
        processor._set_current_batch_index(batch_index)
        for sample_index in range(x.shape[0]):
            if not keep_mask[sample_index].any():
                continue
            start = sample_index * seq_len
            sample = flat[start:start + seq_len][keep_mask[sample_index]]
            quantizer.add_batch(sample, None, batch_index=batch_index)
    return quantizer


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_flattened_native_capture_matches_per_sample_reference(kind):
    """A ``[B*S, H]`` module input splits per sample like a ``[B, S, H]`` one (#4457)."""

    torch.manual_seed(4)
    module = nn.Linear(8, 4, bias=False)
    # Variable-length rows padded to S=4, plus one fully padded sample.
    x = torch.randn(4, 4, 8)
    keep_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [False, False, False, False],
            [True, True, True, False],
        ]
    )

    batched = _run_flattened(kind, module, [(x, keep_mask)])
    reference = _run(
        kind,
        module,
        [(x[i:i + 1], keep_mask[i:i + 1]) for i in range(x.shape[0]) if keep_mask[i].any()],
    )

    assert batched.nsamples == reference.nsamples == int(keep_mask.sum())
    torch.testing.assert_close(batched.H, reference.H, rtol=1e-6, atol=1e-6)
    assert batched.dXXT is not None
    torch.testing.assert_close(batched.dXXT, reference.dXXT, rtol=1e-6, atol=1e-6)


def test_flattened_native_entries_drop_padding_rows():
    """Native entries for flattened rows keep exactly the masked rows of each sample."""

    torch.manual_seed(5)
    module = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 5, 8)
    keep_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [False, False, False, False, False],
            [True, True, True, True, True],
        ]
    )
    named = NamedModule(module, name="shared_expert.up_proj", full_name="model.layers.0.mlp.shared_expert.up_proj", layer_index=0)
    processor = NativeProcessor(
        tokenizer=None,
        qcfg=_qcfg("gptaq"),
        calibration=CalibrationConfig(dataset=None, batch_size=3),
    )
    processor.preprocess(named)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = keep_mask
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook(named.name)(module, (x.reshape(-1, 8),), None)
    processor.process(named)

    entries = named.state[NATIVE_INPUTS_STATE_KEY]
    assert [batch for batch, _ in entries] == [0, 0]
    torch.testing.assert_close(entries[0][1].cpu(), x[0][keep_mask[0]])
    torch.testing.assert_close(entries[1][1].cpu(), x[2][keep_mask[2]])


def _foem_quantize(blocksize, *, group_size=-1, beta=0.2, columns=512, rows=16):
    """Quantize one synthetic linear with FOEM and return Q plus the avg loss."""

    torch.manual_seed(0)
    linear = nn.Linear(columns, rows, bias=False)
    named = NamedModule(linear, name="proj", full_name="model.layers.0.proj", layer_index=0)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        foem=FOEMConfig(alpha=0, beta=beta),
    )
    task = FOEM(named, qcfg)
    task.quantizer.configure(perchannel=True)
    x = torch.randn(64, columns)
    task.add_batch(x, linear(x))
    Q, _scale, _zero, _g_idx, _duration, avg_loss, _damp, _nsamples = task.quantize(
        blocksize=blocksize
    )
    return Q.clone(), avg_loss


def test_foem_block_boundary_beta_correction_is_blocksize_invariant():
    """The beta correction must reach the first column of every block.

    ``FOEM.quantize`` pulls the next column back toward its raw value before
    that column is quantized. The block-boundary form targeted
    ``W[:, i2 + 1]`` after the inner loop had already ended, so it never ran
    and every block after the first skipped beta for its first column. With
    ``group_size=-1`` the quantization grid is fixed before the loop, so the
    only difference between block sizes is the correction itself.
    """

    q128, loss128 = _foem_quantize(128)
    q512, loss512 = _foem_quantize(512)

    torch.testing.assert_close(q128, q512, rtol=0, atol=1e-5)
    torch.testing.assert_close(
        torch.tensor(loss128), torch.tensor(loss512), rtol=0, atol=1e-4
    )


def _hessian_task(kind, batches, *, columns=16):
    """Accumulate ``batches`` into a routed-expert task (``dXXT`` disabled)."""

    named = NamedModule(
        nn.Linear(columns, 8, bias=False),
        name="proj",
        full_name="model.layers.0.proj",
        layer_index=0,
    )
    named.state["is_routed_expert"] = True
    named.state[NATIVE_INPUTS_STATE_KEY] = []
    task = _cls(kind)(named, _qcfg(kind))
    task.quantizer.configure(perchannel=True)
    for batch in batches:
        task.add_batch(batch, None)
    return task


def _normalized_error(actual, expected):
    actual = actual.to(device="cpu", dtype=torch.float64)
    expected = expected.to(device="cpu", dtype=torch.float64)
    return float(
        torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    )


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_hessian_normalisation_matches_documented_contract(kind):
    """FOEM normalizes by tokens, GPTAQ by calibration requests.

    ``docs/quantizer-moe.md`` records the FOEM contract from 60b420d: its
    ``nsamples`` and loss denominator are calibration tokens, matching GPTQ.
    A Hessian normalized by requests instead would change with the number of
    calibration sequences rather than the number of rows, so the same 12 rows
    grouped as 4x3 and as 12x1 must produce the same FOEM Hessian.
    """

    torch.manual_seed(0)
    rows = torch.randn(12, 16)
    grouped = [rows[i * 3:(i + 1) * 3] for i in range(4)]
    single = [rows[i:i + 1] for i in range(12)]
    samples = rows.to(torch.float64)
    sum_xx = samples.T @ samples

    grouped_task = _hessian_task(kind, grouped)
    single_task = _hessian_task(kind, single)

    assert grouped_task.nsamples == single_task.nsamples == rows.shape[0]

    if kind == "foem":
        # Token count: independent of how rows are grouped into forwards.
        assert _normalized_error(grouped_task.H, single_task.H) <= 1e-6
        oracle = (2.0 / rows.shape[0]) * sum_xx
    else:
        # GPTAQ keeps its upstream per-request normalization.
        oracle = (2.0 / len(grouped)) * sum_xx

    assert _normalized_error(grouped_task.H, oracle) <= 1e-6


def test_foem_token_normalisation_matches_gptq_hessian_and_loss():
    """FOEM(alpha=0, beta=0) is documented as equivalent to GPTQ.

    That equivalence depends on FOEM's Hessian and loss denominator using the
    token count. With a request denominator, ``H`` and ``avg_loss`` would differ
    from GPTQ's by the average sequence length. This is the contract the
    routed-expert oracle must encode for FOEM.
    """

    torch.manual_seed(1)
    weight = torch.randn(8, 16)
    rows = torch.randn(12, 16)
    batches = [rows[i * 3:(i + 1) * 3] for i in range(4)]

    def build(cls, qcfg):
        linear = nn.Linear(16, 8, bias=False)
        with torch.no_grad():
            linear.weight.copy_(weight)
        named = NamedModule(
            linear, name="proj", full_name="model.layers.0.proj", layer_index=0
        )
        named.state["is_routed_expert"] = True
        named.state[NATIVE_INPUTS_STATE_KEY] = []
        task = cls(named, qcfg)
        task.quantizer.configure(perchannel=True)
        for batch in batches:
            task.add_batch(batch, None)
        return task

    gptq = build(GPTQ, QuantizeConfig(bits=4, group_size=8, desc_act=False))
    gptq.materialize_global_hessian()
    foem = build(
        FOEM,
        QuantizeConfig(
            bits=4,
            group_size=8,
            desc_act=False,
            foem=FOEMConfig(alpha=0.0, beta=0.0),
        ),
    )

    assert _normalized_error(foem.H, gptq.H) <= 1e-6

    foem_out = foem.quantize()
    gptq_out = gptq.quantize()

    assert foem_out[7] == gptq_out[7] == rows.shape[0]
    torch.testing.assert_close(
        torch.tensor(float(foem_out[5])),
        torch.tensor(float(gptq_out[5])),
        rtol=1e-6,
        atol=0,
    )
    torch.testing.assert_close(foem_out[0], gptq_out[0], rtol=1e-6, atol=1e-6)
