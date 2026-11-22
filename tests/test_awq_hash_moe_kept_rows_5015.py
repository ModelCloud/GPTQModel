# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5015: a kept-row index is not a padded-grid index.

#4996 packed `input_ids` with the replayed AWQ token rows so DeepSeek-V4
`hash_moe` routing (`tid2eid[input_ids.reshape(-1)]`) sees one id per hidden
row. The provenance `row_index` counts *kept* rows (the capture hook strips
padding and excluded sinks), so reading `input_ids.reshape(-1)[row_index]`
handed the router another document's id for every row after the first padded
one. The row count still matched, so the #4996 guard never fired and the
routed-expert scales were searched on activations the model never produces.

These nodes check the id *values* against the kept grid, not just the count.
"""

from __future__ import annotations

import torch

from gptqmodel import GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import FORMAT, AWQConfig, CalibrationConfig
from gptqmodel.utils.attn_mask import normalize_seq_mask
from tests.test_awq_deepseek_v4_hash_moe_4880 import _build_tiny_model, _enable_hash_moe_scale_search
from tests.tiny_remote_models import build_wordlevel_tokenizer


_WORDS = "the quick brown fox jumps over lazy dog alpha beta gamma delta epsilon".split()


def _calibration(lengths):
    """Rows long enough to differ, each starting with the tokenizer's BOS."""

    tokenizer = build_wordlevel_tokenizer()
    bos = tokenizer.bos_token_id
    rows = []
    for length in lengths:
        text = " ".join(_WORDS[index % len(_WORDS)] for index in range(length))
        encoded = tokenizer(text, return_tensors="pt")
        # The bare word-level tokenizer has no post-processor, so prepend the
        # BOS explicitly: `exclude_attention_sinks` keys off the BOS position.
        ids = torch.cat([torch.tensor([[bos]]), encoded["input_ids"]], dim=-1)
        rows.append(
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }
        )
    return rows


def _kept_grid_truth(processor, provenance, batch_indices):
    """The id each replayed row must carry, derived from the capture's grid.

    Independent of the code under test: the kept set is the padding keep-mask
    from the cached attention mask, minus the processor's own BOS sink
    positions when sink exclusion is on. That is exactly the set the capture
    hook stored a feature for, so kept index ``r`` must read grid row
    ``nonzero(keep)[r]``.
    """

    cache = processor.inputs_cache
    exclude_sinks = bool(
        getattr(getattr(processor, "calibration", None), "exclude_attention_sinks", False)
    )
    truth = []
    for source_index, row_index in provenance:
        batch_index = batch_indices[source_index]
        keep = normalize_seq_mask(cache.attention_masks[batch_index].cpu())
        if exclude_sinks:
            sinks = processor.attention_sink_positions_for_batch(batch_index)
            if torch.is_tensor(sinks) and tuple(sinks.shape) == tuple(keep.shape):
                keep = keep & ~sinks.to(dtype=torch.bool)
        grid = torch.nonzero(keep.reshape(-1), as_tuple=False).reshape(-1)
        ids = cache.layer_input_kwargs[batch_index]["input_ids"].reshape(-1)
        truth.append(int(ids[grid[int(row_index)]]))
    return truth


def _run_awq(tmp_path, monkeypatch, *, batch_size, exclude_sinks=False):
    _enable_hash_moe_scale_search(monkeypatch)
    model_dir = tmp_path / "dsv4-hash-moe"
    _build_tiny_model(model_dir)

    seen = {"rows": 0, "mismatch": 0, "calls": 0, "packed": []}
    original = AWQProcessor._pack_token_aligned_kwargs

    def spy(self, provenance, batch_indices, *rest, **kwargs):
        out = original(self, provenance, batch_indices, *rest, **kwargs)
        ids = out.get("input_ids")
        if ids is not None:
            truth = _kept_grid_truth(self, provenance, batch_indices)
            seen["calls"] += 1
            seen["rows"] += len(truth)
            seen["packed"].append(ids.reshape(-1).cpu())
            seen["mismatch"] += int(
                (torch.tensor(truth) != ids.reshape(-1).cpu()).sum()
            )
        return out

    monkeypatch.setattr(AWQProcessor, "_pack_token_aligned_kwargs", spy)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=AWQConfig(bits=4, group_size=32, format=FORMAT.GEMM, device="cpu"),
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(
            dataset=_calibration((9, 14, 6, 11, 17, 8, 13, 10)),
            batch_size=batch_size,
            min_length=1,
            exclude_attention_sinks=exclude_sinks,
        )
    )
    return seen


def test_packed_input_ids_follow_the_padded_kept_grid(tmp_path, monkeypatch):
    """Variable-length rows in one batch: kept indices are not grid indices."""

    seen = _run_awq(tmp_path, monkeypatch, batch_size=4)

    assert seen["calls"] > 0, "the hash_moe replay must pack input_ids"
    assert seen["rows"] > 0
    assert seen["mismatch"] == 0, (
        f"{seen['mismatch']}/{seen['rows']} replayed hash_moe tokens were routed "
        "with another token's id"
    )


def test_unpadded_single_row_batches_are_unaffected(tmp_path, monkeypatch):
    """The no-padding case is the one #4996 already got right."""

    seen = _run_awq(tmp_path, monkeypatch, batch_size=1)

    assert seen["rows"] > 0
    assert seen["mismatch"] == 0


def test_packed_input_ids_follow_the_kept_grid_with_sinks_excluded(
    tmp_path, monkeypatch
):
    """Dropping the BOS sink also shifts the kept index away from the grid."""

    seen = _run_awq(tmp_path, monkeypatch, batch_size=4, exclude_sinks=True)

    packed = torch.cat(seen["packed"]) if seen["packed"] else torch.empty(0, dtype=torch.long)
    assert packed.numel() > 0
    # The fixture prepends the tokenizer's BOS, and the sink exclusion removes
    # it from every captured row; a replayed BOS id would mean the packer read
    # the un-excluded grid.
    assert int(build_wordlevel_tokenizer().bos_token_id) not in set(packed.tolist()), (
        "an excluded attention-sink token was replayed"
    )
    assert seen["rows"] > 0
    assert seen["mismatch"] == 0, (
        f"{seen['mismatch']}/{seen['rows']} replayed tokens were misrouted with "
        "attention sinks excluded"
    )
