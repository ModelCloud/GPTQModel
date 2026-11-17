# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`CalibrationConfig.exclude_attention_sinks` (issue #4393).

Every BOS token is an attention sink whose activation norm can dwarf the text
tokens (Llama-3.2-1B: ~500-1100 vs ~1.8 at ``layers.1.mlp.down_proj``), so a
handful of rows dominates every calibration statistic. The option drops the
attended BOS positions once, in the shared calibration keep-mask, so every
processor family sees the same retained rows. A tokenizer without a BOS token
(Qwen2/2.5/3) has no sink to drop and its run is unchanged (issue #4459). These
tests pin:

* the sink positions and the mask edit (left/right padding, a concatenated
  window with a mid-row BOS, rows without BOS, a tokenizer without BOS);
* each processor family's statistic against a float64 oracle over the
  retained rows;
* that the default path is the historical one, object for object;
* that forward-visible consumers keep using the padding-only mask;
* the public API and the checkpoint provenance record.
"""

import os
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.fp8_gsq_processor import FP8GSQProcessor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.loop_processor import LoopProcessor, active_padding_mask
from gptqmodel.looper.macaq_processor import MacaqProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.looper.qqq_processor import QQQProcessor
from gptqmodel.looper.vptq_processor import VPTQProcessor
from gptqmodel.quantization.config import (
    META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS,
    CalibrationConfig,
    FOEMConfig,
    GPTAQConfig,
    GPTQConfig,
    MacaqConfig,
    QQQConfig,
    QuantizeConfig,
    VPTQConfig,
)
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.macaq import Macaq
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.utils.attn_mask import bos_token_positions, exclude_attention_sinks


BOS = 7
HIDDEN = 8


class _Processor(LoopProcessor):
    def name(self) -> str:
        return "sink-test"


def _identity_prepare(calibration_dataset, **_kwargs):
    return calibration_dataset


def _batch():
    """BOS + right pad, left pad + BOS, and a concatenated window that starts
    mid-document and carries a document-boundary BOS."""

    input_ids = torch.tensor(
        [
            [BOS, 11, 12, 13, 0],
            [0, 0, BOS, 21, 22],
            [31, 32, BOS, 34, 35],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    return input_ids, attention_mask


def _expected_keep(attention_mask: torch.Tensor) -> torch.Tensor:
    keep = attention_mask.bool().clone()
    keep[0, 0] = False  # leading BOS of a right-padded row
    keep[1, 2] = False  # leading BOS of a left-padded row
    keep[2, 2] = False  # document-boundary BOS inside the window
    # keep[2, 0] stays: a window that starts mid-document leads with text.
    return keep


def _flagged_processor(cls, *, exclude: bool = True, **attrs):
    """A processor of ``cls`` whose sink flags come from the shared helper."""

    processor = object.__new__(cls)
    processor.calibration = CalibrationConfig(exclude_attention_sinks=exclude)
    processor.tokenizer = SimpleNamespace(bos_token_id=BOS)
    input_ids, attention_mask = _batch()
    processor.attention_sink_positions = (
        processor._compute_attention_sink_positions(
            [{"input_ids": input_ids, "attention_mask": attention_mask}], BOS
        )[0]
        if exclude
        else None
    )
    processor._batch_tls = SimpleNamespace(index=0)
    processor.lock = threading.Lock()
    for key, value in attrs.items():
        setattr(processor, key, value)
    return processor


def _bind_masks(processor, layer_input: torch.Tensor):
    """Bind the masks exactly as ``ForwardExecutor`` does for one batch."""

    _, attention_mask = _batch()
    padding = attention_mask.bool()
    keep = processor.calibration_keep_mask(padding, batch_index=0, layer_input=[layer_input])
    ModuleLooper.__new__(ModuleLooper)._set_processor_mask(processor, keep, padding_mask=padding)
    return keep, padding


def _wrapped_hook(processor, name: str):
    """``processor``'s hook behind the looper's mask wrapper, as registered."""

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = None
    return looper._masked_hook_wrapper(processor, processor.pre_process_fwd_hook(name), name)


def _inputs() -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(3, 5, HIDDEN, dtype=torch.float32)
    # Make the sinks dominate, as they do in a real model.
    x[0, 0] *= 300.0
    x[1, 2] *= 300.0
    x[2, 2] *= 300.0
    return x


def _retained_rows(x: torch.Tensor) -> torch.Tensor:
    _, attention_mask = _batch()
    return x[_expected_keep(attention_mask)]


def _assert_proportional_to_oracle(hessian: torch.Tensor, rows: torch.Tensor) -> None:
    """``hessian`` must be ``c * X^T X`` over exactly ``rows`` (float64 oracle)."""

    rows64 = rows.reshape(-1, rows.shape[-1]).to(torch.float64)
    oracle = rows64.T @ rows64
    hessian64 = hessian.detach().to(device="cpu", dtype=torch.float64)
    scale = hessian64.trace() / oracle.trace()
    torch.testing.assert_close(hessian64, scale * oracle, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# Row flags and the mask edit
# --------------------------------------------------------------------------


def test_bos_token_positions_marks_every_attended_bos():
    input_ids, attention_mask = _batch()
    positions = bos_token_positions(input_ids, attention_mask, BOS)
    assert torch.nonzero(positions).tolist() == [[0, 0], [1, 2], [2, 2]]

    # Pre-tokenized python lists and a missing mask behave the same way.
    assert bos_token_positions([[BOS, 1, 2], [3, BOS, 4]], None, BOS).tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    # A padded BOS is not attended and never counts.
    assert bos_token_positions(torch.tensor([[BOS, 1]]), torch.tensor([[0, 1]]), BOS) is None


def test_bos_token_positions_is_none_without_bos():
    input_ids, attention_mask = _batch()
    # A tokenizer without a BOS token (Qwen2/2.5) has none to mark.
    assert bos_token_positions(input_ids, attention_mask, None) is None
    # Pre-tokenized rows that never contain BOS mark nothing either.
    assert bos_token_positions([[31, 32, 33]], None, BOS) is None
    # Float "ids" (embeddings) are not token ids.
    assert bos_token_positions(torch.full((1, 3), float(BOS)), None, BOS) is None


def test_exclude_attention_sinks_clears_only_bos_positions():
    input_ids, attention_mask = _batch()
    padding = attention_mask.bool()
    positions = bos_token_positions(input_ids, attention_mask, BOS)

    keep = exclude_attention_sinks(padding, positions)
    assert torch.equal(keep, _expected_keep(attention_mask))
    # The padding mask itself is never mutated.
    assert torch.equal(padding, attention_mask.bool())

    # Without BOS positions nothing is dropped: the same mask object comes back.
    assert exclude_attention_sinks(padding, None) is padding
    assert exclude_attention_sinks(padding, torch.zeros_like(padding)) is padding
    assert exclude_attention_sinks(None, None, batch=2, seq_len=3) is None
    # Without a captured attention mask every position is attended.
    lead = torch.tensor([[True, False, False], [False, True, False]])
    assert exclude_attention_sinks(None, lead, batch=2, seq_len=3).tolist() == [
        [False, True, True],
        [True, False, True],
    ]
    # BOS positions that do not fit the replayed sequence (a VLM whose forward
    # expands image placeholders) cannot be placed and are ignored.
    assert exclude_attention_sinks(padding, positions[:, :4]) is padding
    assert exclude_attention_sinks(None, lead, batch=2, seq_len=4) is None


# --------------------------------------------------------------------------
# Shared calibration stage
# --------------------------------------------------------------------------


def _loop_processor(*, exclude: bool, bos_token_id=BOS, dataset=None) -> _Processor:
    input_ids, attention_mask = _batch()
    if dataset is None:
        dataset = [
            {"input_ids": input_ids, "attention_mask": attention_mask},
            {"input_ids": torch.tensor([[BOS, 1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])},
            {"input_ids": torch.tensor([[4, 5, 6]]), "attention_mask": torch.tensor([[1, 1, 1]])},
        ]
    return _Processor(
        tokenizer=SimpleNamespace(bos_token_id=bos_token_id),
        qcfg=QuantizeConfig(bits=4, group_size=8),
        calibration=CalibrationConfig(dataset=dataset, exclude_attention_sinks=exclude).bind(_identity_prepare),
    )


def test_calibration_token_total_matches_retained_rows():
    off = _loop_processor(exclude=False)
    on = _loop_processor(exclude=True)

    # 4 + 3 + 5 attended tokens in batch 0, 4 in batch 1, 3 in batch 2.
    assert off.total_calibration_tokens == 19
    assert off.attention_sink_positions is None
    # Three BOS in batch 0 and one in batch 1 go; the fallback threshold and
    # averages use the 15 retained rows. Batch 2 holds no BOS.
    assert on.total_calibration_tokens == 15
    positions = on.attention_sink_positions
    assert [torch.nonzero(p).tolist() for p in positions[:2]] == [
        [[0, 0], [1, 2], [2, 2]],
        [[0, 0]],
    ]
    assert positions[2] is None


def test_tokenizer_without_bos_changes_nothing():
    off = _loop_processor(exclude=False, bos_token_id=None)
    on = _loop_processor(exclude=True, bos_token_id=None)
    # No BOS token, no sink: a BOS-less tokenizer (Qwen2/2.5/3) keeps its
    # first real text token (issue #4459).
    assert on.total_calibration_tokens == off.total_calibration_tokens == 19
    assert on.attention_sink_positions is None
    padding = _batch()[1].bool()
    assert on.calibration_keep_mask(padding, batch_index=0, layer_input=[_inputs()]) is padding
    assert on.calibration_keep_mask(None, batch_index=1, layer_input=[_inputs()]) is None


def test_batch_without_bos_keeps_its_mask_object():
    processor = _loop_processor(exclude=True)
    x = torch.randn(1, 3, HIDDEN)
    assert processor.calibration_keep_mask(None, batch_index=2, layer_input=[x]) is None
    padding = torch.ones(1, 3, dtype=torch.bool)
    assert processor.calibration_keep_mask(padding, batch_index=2, layer_input=[x]) is padding


def test_default_calibration_keep_mask_is_the_padding_mask_object():
    processor = _loop_processor(exclude=False)
    padding = _batch()[1].bool()
    assert processor.calibration_keep_mask(padding, batch_index=0, layer_input=[_inputs()]) is padding
    assert processor.calibration_keep_mask(None, batch_index=0, layer_input=[_inputs()]) is None


def test_calibration_keep_mask_without_attention_mask_uses_layer_shape():
    processor = _loop_processor(exclude=True)
    x = torch.randn(1, 4, HIDDEN)
    keep = processor.calibration_keep_mask(None, batch_index=1, layer_input=[x])
    assert keep.tolist() == [[False, True, True, True]]
    # A replayed sequence longer than ``input_ids`` (VLM expansion) is left alone.
    longer = torch.randn(1, 6, HIDDEN)
    assert processor.calibration_keep_mask(None, batch_index=1, layer_input=[longer]) is None


def test_set_processor_mask_keeps_padding_separate():
    processor = _flagged_processor(GPTQProcessor)
    keep, padding = _bind_masks(processor, _inputs())
    assert processor._mask_tls.value is keep
    assert active_padding_mask(processor) is padding
    assert processor._mask_tls.sample_has_tokens == [True, True, True]
    assert processor._mask_tls.mask_has_padding is True

    # A sink-only mask over an unpadded batch reports no padding.
    ModuleLooper.__new__(ModuleLooper)._set_processor_mask(
        processor, torch.tensor([[False, True]]), padding_mask=None
    )
    assert processor._mask_tls.mask_has_padding is False


def test_calibration_config_rejects_non_bool():
    with pytest.raises(TypeError, match="exclude_attention_sinks"):
        CalibrationConfig(exclude_attention_sinks=1)


# --------------------------------------------------------------------------
# Every processor family accumulates over the retained rows only
# --------------------------------------------------------------------------


def _linear() -> nn.Linear:
    torch.manual_seed(1)
    return nn.Linear(HIDDEN, 4, bias=False, dtype=torch.float32).eval()


def test_gptq_hessian_matches_float64_oracle_on_retained_rows():
    layer = _linear()
    task = GPTQ(layer, qcfg=QuantizeConfig(bits=4, group_size=4))
    processor = _flagged_processor(GPTQProcessor, tasks={"proj": task}, qcfg=task.qcfg)
    x = _inputs()
    _bind_masks(processor, x)
    processor.pre_process_fwd_hook("proj")(layer, (x,), layer(x))

    rows = _retained_rows(x)
    assert task.nsamples == rows.shape[0] == 9
    rows64 = rows.to(torch.float64)
    oracle = (2.0 * rows64.T @ rows64 / rows64.shape[0]).to(torch.float32)
    torch.testing.assert_close(task.finalize_hessian(), oracle, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_gptaq_foem_hessian_and_dxxt_use_the_retained_rows(kind):
    """Native capture and the quantized-path hook drop the same sink rows."""

    layer = _linear()
    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=4,
        **({"gptaq": GPTAQConfig(device="cpu")} if kind == "gptaq" else {"foem": FOEMConfig(alpha=0.25, device="cpu")}),
    )
    cls = GPTAQ if kind == "gptaq" else FOEM
    x_native = _inputs()
    x_quant = x_native + 0.01 * torch.randn_like(x_native)

    native = NativeProcessor(tokenizer=SimpleNamespace(bos_token_id=BOS), qcfg=qcfg,
                             calibration=CalibrationConfig(exclude_attention_sinks=True))
    native.attention_sink_positions = native._compute_attention_sink_positions(
        [{"input_ids": _batch()[0], "attention_mask": _batch()[1]}], BOS
    )[0]
    native._batch_tls = SimpleNamespace(index=0)
    native.preprocess(named)
    _bind_masks(native, x_native)
    native.pre_process_fwd_hook(named.name)(layer, (x_native,), None)
    native.process(named)

    task = cls(named, qcfg)
    processor = _flagged_processor(GPTQProcessor, tasks={"proj": task}, qcfg=qcfg)
    _bind_masks(processor, x_quant)
    processor.pre_process_fwd_hook("proj")(layer, (x_quant,), layer(x_quant))

    # Reference: feed the explicitly retained rows of every sample.
    ref_named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    keep = _expected_keep(_batch()[1])
    ref_named.state["native_inp"] = [
        (0, x_native[i][keep[i]]) for i in range(x_native.shape[0])
    ]
    from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY

    ref_named.state[NATIVE_INPUTS_STATE_KEY] = ref_named.state.pop("native_inp")
    reference = cls(ref_named, qcfg)
    for i in range(x_quant.shape[0]):
        reference.add_batch(x_quant[i : i + 1, keep[i]], None, batch_index=0)

    rows = _retained_rows(x_quant)
    assert task.nsamples == reference.nsamples == rows.shape[0]
    torch.testing.assert_close(task.H, reference.H, rtol=0, atol=0)
    torch.testing.assert_close(task.dXXT, reference.dXXT, rtol=0, atol=0)
    _assert_proportional_to_oracle(task.H, rows)


@pytest.mark.parametrize(
    "processor_cls,make_task,hessian",
    [
        (QQQProcessor, lambda layer: QQQ(layer, QQQConfig(bits=4, group_size=4)), lambda t: t.materialize_hessian()),
        (MacaqProcessor, lambda layer: Macaq(layer, MacaqConfig()), lambda t: t.materialize_hessian()),
    ],
    ids=["qqq", "macaq"],
)
def test_masked_hooks_accumulate_retained_rows(processor_cls, make_task, hessian):
    layer = _linear()
    task = make_task(layer)
    processor = _flagged_processor(processor_cls, tasks={"proj": task})
    x = _inputs()
    _bind_masks(processor, x)
    _wrapped_hook(processor, "proj")(layer, (x,), layer(x))

    rows = _retained_rows(x)
    assert task.nsamples == rows.shape[0]
    _assert_proportional_to_oracle(hessian(task), rows)


def test_exl3_capture_accumulates_retained_rows():
    layer = _linear()
    capture = GPTQ(layer, qcfg=QuantizeConfig(bits=4, group_size=4))
    processor = _flagged_processor(EXL3Processor, tasks={"proj": {"capture": capture}})
    x = _inputs()
    _bind_masks(processor, x)
    _wrapped_hook(processor, "proj")(layer, (x,), layer(x))

    rows = _retained_rows(x)
    assert capture.nsamples == rows.shape[0]
    _assert_proportional_to_oracle(capture.finalize_hessian(), rows)


def test_calibration_rows_is_the_input_when_nothing_is_excluded():
    """Option off, or a batch without BOS: the hook input object passes through."""

    x = _inputs()
    processor = _flagged_processor(QQQProcessor, exclude=False)
    _bind_masks(processor, x)
    assert processor.calibration_rows(x) is x
    flat = x.reshape(-1, HIDDEN)
    assert processor.calibration_rows(flat) is flat

    processor = _flagged_processor(QQQProcessor)
    padding = _batch()[1].bool()
    ModuleLooper.__new__(ModuleLooper)._set_processor_mask(processor, padding, padding_mask=padding)
    assert processor.calibration_rows(flat) is flat


def test_unmasked_rows_lose_only_the_bos_sinks():
    """Inputs the wrapper cannot mask keep their padding rows, as before."""

    processor = _flagged_processor(QQQProcessor)
    x = _inputs()
    _bind_masks(processor, x)
    input_ids, attention_mask = _batch()
    not_bos = ~bos_token_positions(input_ids, attention_mask, BOS)
    torch.testing.assert_close(processor.calibration_rows(x), x[not_bos])
    flat = x.reshape(-1, HIDDEN)
    torch.testing.assert_close(processor.calibration_rows(flat), flat[not_bos.reshape(-1)])
    # Natively routed `[routed_rows, H]` inputs have no sequence position.
    routed = flat[:6]
    assert processor.calibration_rows(routed) is routed


def test_gsq_capture_receives_the_calibration_mask():
    seen = []

    class _Task:
        def add_batch(self, source, *, source_weight=1.0, mask=None):
            seen.append(mask)

    processor = _flagged_processor(FP8GSQProcessor, tasks={"proj": _Task()}, calibration_dataset=[{}])
    x = _inputs()
    keep, _ = _bind_masks(processor, x)
    processor.pre_process_fwd_hook("proj")(None, (x,), None)
    assert torch.equal(seen[0], keep)


def test_awq_features_and_token_row_structure_drop_the_sink():
    processor = _flagged_processor(AWQProcessor)
    x = _inputs()
    keep, padding = _bind_masks(processor, x)
    feature = processor._apply_active_keep_mask_to_feature(x)
    torch.testing.assert_close(feature, _retained_rows(x))

    # The replay structure maps the retained rows back to their true cells.
    sinks = processor.attention_sink_positions_for_batch(0)
    documents, positions, allowed, form = AWQProcessor._token_row_source_structure(
        padding.to(torch.long), feature.shape[0], sinks, exclude_sinks=True
    )
    expected = torch.nonzero(keep)
    assert documents.tolist() == expected[:, 0].tolist()
    assert positions.tolist() == expected[:, 1].tolist()
    assert form is not None

    # No captured mask: the unpadded grid is rebuilt from the sink positions'
    # shape, retained rows keep their true positions and no mask form is
    # invented.
    lead_bos = torch.tensor([[True, False, False, False]])
    documents, positions, allowed, form = AWQProcessor._token_row_source_structure(
        None, 3, lead_bos, exclude_sinks=True
    )
    assert positions.tolist() == [1, 2, 3]
    assert form is None
    assert torch.equal(allowed, torch.tril(torch.ones(3, 3, dtype=torch.bool)))
    mid_bos = torch.tensor([[False, False, True, False, False]])
    _, positions, _, _ = AWQProcessor._token_row_source_structure(None, 4, mid_bos, exclude_sinks=True)
    assert positions.tolist() == [0, 1, 3, 4]
    # A batch without BOS keeps the historical "no structure" answer.
    assert AWQProcessor._token_row_source_structure(None, 3, None, exclude_sinks=True) is None
    # With the option off the historical "no structure" answer is unchanged.
    assert AWQProcessor._token_row_source_structure(None, 3) is None


def test_awq_siblings_share_one_masked_host_copy():
    """q/k/v hooks read one activation; the masked host copy is made once."""

    processor = _flagged_processor(AWQProcessor, _capture_host_cache={}, _capture_host_batch=None)
    x = _inputs()
    keep, _ = _bind_masks(processor, x)
    first = processor._materialize_host_feature(x)
    assert processor._materialize_host_feature(x) is first
    torch.testing.assert_close(first, _retained_rows(x).unsqueeze(0))

    # A different keep-mask object (another batch binding) never aliases it.
    ModuleLooper.__new__(ModuleLooper)._set_processor_mask(processor, keep.clone(), padding_mask=None)
    assert processor._materialize_host_feature(x) is not first


_LLAMA_TOKENIZER = "/monster/data/model/Llama-3.2-1B-Instruct"


@pytest.mark.skipif(not os.path.isdir(_LLAMA_TOKENIZER), reason="Llama-3.2 tokenizer not available")
def test_real_llama_bos_rows_are_excluded_after_the_dataset_is_released():
    """#4568: the BOS decision must survive ``release_calibration_dataset``.

    ``ModuleLooper`` deletes ``calibration_dataset`` on every processor before
    the first layer's hooks run, so a decision that reads it from the hook is
    a silent no-op. The positions are resolved from the real tokenizer's
    ``input_ids`` when the processor is built, and the Hessian of a real GPTQ
    task must then lose exactly the BOS rows.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_LLAMA_TOKENIZER)
    bos = tokenizer.bos_token_id
    assert bos is not None
    single = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    first = tokenizer("A first calibration document.", return_tensors="pt")["input_ids"]
    second = tokenizer("A second one, packed behind it.", return_tensors="pt")["input_ids"]
    window = torch.cat([first, second], dim=1)  # concatenated calibration window
    assert int(single["input_ids"][0, 0]) == bos
    assert int(window[0, first.shape[1]]) == bos
    dataset = [
        {"input_ids": single["input_ids"], "attention_mask": single["attention_mask"]},
        {"input_ids": window, "attention_mask": torch.ones_like(window)},
    ]

    layer = _linear()
    qcfg = QuantizeConfig(bits=4, group_size=4)
    processor = GPTQProcessor(
        tokenizer=tokenizer,
        qcfg=qcfg,
        calibration=CalibrationConfig(dataset=dataset, exclude_attention_sinks=True).bind(_identity_prepare),
    )
    attended = sum(int(row["attention_mask"].sum()) for row in dataset)
    assert processor.total_calibration_tokens == attended - 3
    processor.release_calibration_dataset()
    assert not hasattr(processor, "calibration_dataset")

    task = GPTQ(layer, qcfg=qcfg)
    processor.tasks = {"proj": task}
    hook = _wrapped_hook(processor, "proj")
    looper = ModuleLooper.__new__(ModuleLooper)
    retained = []
    for batch_index, row in enumerate(dataset):
        torch.manual_seed(batch_index)
        x = torch.randn(1, row["input_ids"].shape[1], HIDDEN)
        padding = row["attention_mask"].bool()
        keep = processor.calibration_keep_mask(padding, batch_index=batch_index, layer_input=[x])
        assert torch.equal(~keep, row["input_ids"] == bos)
        looper._set_processor_mask(processor, keep, padding_mask=padding)
        processor._set_current_batch_index(batch_index)
        hook(layer, (x,), layer(x))
        retained.append(x[keep])

    rows = torch.cat(retained)
    assert task.nsamples == rows.shape[0] == attended - 3
    _assert_proportional_to_oracle(task.finalize_hessian(), rows)


# --------------------------------------------------------------------------
# Forward-visible consumers keep the padding-only mask
# --------------------------------------------------------------------------


def test_moe_padding_zeroing_never_touches_the_sink_row():
    processor = _flagged_processor(GPTQProcessor, preserve_batch_keep_mask=True)
    x = _inputs()
    _bind_masks(processor, x)
    block = nn.Identity()
    handles = []
    assert processor.register_moe_root_capture_hook(block, "mlp", handles)
    try:
        out = block(x)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(out[0, 0], x[0, 0])  # leading BOS kept
    torch.testing.assert_close(out[1, 2], x[1, 2])  # leading BOS kept
    torch.testing.assert_close(out[2, 2], x[2, 2])  # mid-row BOS kept
    assert torch.count_nonzero(out[0, 4]) == 0  # right pad zeroed
    assert torch.count_nonzero(out[1, :2]) == 0  # left pad zeroed


# --------------------------------------------------------------------------
# Public API and provenance
# --------------------------------------------------------------------------


def test_gptq_config_no_longer_accepts_the_option():
    with pytest.raises(TypeError, match="exclude_attention_sinks"):
        GPTQConfig(bits=4, group_size=128, exclude_attention_sinks=True)


@pytest.mark.parametrize("cls", [VPTQProcessor, ParoQuantProcessor], ids=["vptq", "paroquant"])
def test_methods_that_cannot_use_the_option_reject_it(cls):
    kwargs = {"gptq_model": None, "model": None} if cls is ParoQuantProcessor else {}
    qcfg = VPTQConfig() if cls is VPTQProcessor else QuantizeConfig(method="paroquant", bits=4, group_size=128)
    with pytest.raises(ValueError, match=r"CalibrationConfig\.exclude_attention_sinks"):
        cls(
            tokenizer=SimpleNamespace(bos_token_id=BOS),
            qcfg=qcfg,
            calibration=CalibrationConfig(exclude_attention_sinks=True),
            **kwargs,
        )


def test_calibration_config_carries_the_sink_option():
    import inspect

    from gptqmodel.models.base import BaseQModel

    parameters = inspect.signature(BaseQModel.quantize).parameters
    assert "calibration_exclude_attention_sinks" not in parameters
    assert "CalibrationConfig" in str(parameters["calibration"].annotation)
    assert CalibrationConfig().exclude_attention_sinks is False
    config = CalibrationConfig(exclude_attention_sinks=True).bind(_identity_prepare, batch_size=2)
    assert config.exclude_attention_sinks is True
    assert config.batch_size == 2


def test_rejects_a_per_module_sink_rule():
    from gptqmodel.models.base import BaseQModel

    # Construction rejects the rule, before any module is matched.
    with pytest.raises(ValueError, match=r"CalibrationConfig\(exclude_attention_sinks=True\)"):
        GPTQConfig(
            bits=4,
            group_size=128,
            dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
        )

    # A rule attached after construction is still refused before calibration.
    model = object.__new__(BaseQModel)
    model.quantize_config = GPTQConfig(bits=4, group_size=128)
    model.quantize_config.dynamic = {r"+:.*down_proj.*": {"exclude_attention_sinks": True}}
    with pytest.raises(ValueError, match=r"CalibrationConfig\(exclude_attention_sinks=True\)"):
        model.quantize()


def test_provenance_meta_round_trips():
    qcfg = GPTQConfig(bits=4, group_size=128)
    qcfg.meta_set(key=META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS, value=True)
    payload = qcfg.to_dict()
    assert payload["meta"][META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS] is True
    restored = GPTQConfig.from_quant_config(payload)
    assert restored.meta_get(META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS) is True

    # The default policy writes nothing.
    default_payload = GPTQConfig(bits=4, group_size=128).to_dict()
    assert META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS not in default_payload.get("meta", {})
