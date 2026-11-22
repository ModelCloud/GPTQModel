# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Black-box checks of the LocateAnything localization sampler against the golden fixture."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.ports.locateanything.generate_utils import (
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    build_token_ids,
    classify_box_frame,
    decode_bbox_avg,
    decode_ref,
    handle_pattern,
    select_multi_position,
)


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/cleanroom/locateanything/sampler.json").read_text()
)
TOKEN_IDS = FIXTURE["token_ids"]
CASES = FIXTURE["cases"]
PROCESSORS = FIXTURE["processors"]


def _as_list(value):
    return None if value is None else value.tolist()


def test_token_table_from_config():
    config = SimpleNamespace(
        box_start_token_id=32, box_end_token_id=33, coord_start_token_id=16, coord_end_token_id=31,
        ref_start_token_id=34, ref_end_token_id=35, none_token_id=36,
        text_config=SimpleNamespace(null_token_id=61, eos_token_id=2, switch_token_id=62, text_mask_token_id=60),
    )
    assert build_token_ids(config) == TOKEN_IDS


def test_token_table_defaults():
    table = build_token_ids(SimpleNamespace(text_config=SimpleNamespace()))
    assert table == {
        "box_start_token_id": 151668, "box_end_token_id": 151669, "coord_start_token_id": 151677,
        "coord_end_token_id": 152677, "ref_start_token_id": 151672, "ref_end_token_id": 151673,
        "none_token_id": 4064, "null_token_id": 152678, "im_end_token_id": 151645,
        "switch_token_id": 152679, "default_mask_token_id": 151676,
    }


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_frame_decoders_match_fixture(case):
    logits = torch.tensor(case["logits"])
    probs = torch.softmax(logits, dim=-1)
    assert classify_box_frame(probs, TOKEN_IDS) == case["frame_class"]
    assert _as_list(decode_ref(logits, probs, TOKEN_IDS)) == case["ref_decode"]
    for mode in ("hybrid", "fast"):
        assert _as_list(decode_bbox_avg(logits, probs, TOKEN_IDS, keep_k=4, generation_mode=mode)) == \
            case[f"decode_bbox_avg_{mode}_keep4"]
        assert _as_list(decode_bbox_avg(logits, probs, TOKEN_IDS, generation_mode=mode)) == \
            case[f"decode_bbox_avg_{mode}_default_keep5"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
@pytest.mark.parametrize("mode", ["hybrid", "fast"])
def test_multi_position_selection_matches_fixture(case, mode):
    logits = torch.tensor(case["logits"])[None]
    probs, confidence, ids, box = select_multi_position(logits, None, TOKEN_IDS, generation_mode=mode)
    expected = case[f"select_{mode}"]
    torch.testing.assert_close(probs, torch.softmax(logits, dim=-1), rtol=0, atol=0)
    assert ids.tolist() == expected["argmax_ids"]
    torch.testing.assert_close(confidence, torch.tensor(expected["confidence"]), rtol=0, atol=1e-7)
    assert box.dtype == torch.long and box.tolist() == expected["box"]


@pytest.mark.parametrize("entry", FIXTURE["handle_pattern"], ids=lambda entry: ",".join(map(str, entry["candidate"])))
@pytest.mark.parametrize("mode", ["hybrid", "fast"])
def test_handle_pattern_matches_fixture(entry, mode):
    assert handle_pattern(entry["candidate"], TOKEN_IDS, generation_mode=mode) == entry[mode]
    assert handle_pattern(torch.tensor(entry["candidate"]), TOKEN_IDS, mode) == entry[mode]


def test_logit_processors_match_fixture():
    logits = torch.tensor(PROCESSORS["logits"])
    previous = torch.tensor(PROCESSORS["prev_ids"])
    assert torch.equal(apply_repetition_penalty(logits, previous, 1.3), torch.tensor(PROCESSORS["repetition_penalty_1.3"]))
    assert torch.equal(
        apply_repetition_penalty(logits[:, 0], previous, 0.7), torch.tensor(PROCESSORS["repetition_penalty_0.7_2d"])
    )
    assert torch.equal(apply_top_p(logits, 0.5), torch.tensor(PROCESSORS["top_p_0.5"]))
    assert torch.equal(apply_top_k(logits, 3), torch.tensor(PROCESSORS["top_k_3"]))
    assert torch.equal(apply_top_k(logits, 1000), torch.tensor(PROCESSORS["top_k_1000"]))


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("select_single_greedy", {}),
        ("select_single_rp1.3_topk5_topp0.9", {"repetition_penalty": 1.3, "top_k": 5, "top_p": 0.9}),
        # Recorded probabilities equal a greedy top-2 filter of the raw logits.
        ("select_single_temperature0.5_greedy_topk", {"top_k": 2}),
    ],
)
def test_single_position_selection_matches_fixture(name, kwargs):
    logits = torch.tensor(PROCESSORS["logits"])[:, :1]
    previous = torch.tensor(PROCESSORS["prev_ids"])
    probs, confidence, ids, box = select_multi_position(logits, previous, TOKEN_IDS, **kwargs)
    expected = PROCESSORS[name]
    assert box is None
    assert ids.tolist() == expected["ids"]
    torch.testing.assert_close(confidence, torch.tensor(expected["confidence"]), rtol=0, atol=1e-7)
    torch.testing.assert_close(probs, torch.tensor(expected["probs"]), rtol=0, atol=1e-7)


# Handcrafted behavior checks ----------------------------------------------------------------


def test_handle_pattern_shapes():
    t = TOKEN_IDS
    bs, be, null = t["box_start_token_id"], t["box_end_token_id"], t["null_token_id"]
    assert handle_pattern([bs, 16, 17, 18, 19, be], t) == {
        "type": "coord_box", "tokens": [bs, 16, 17, 18, 19, be], "need_switch_to_ar": False, "is_terminal": False,
    }
    assert handle_pattern([bs, 20, 21, be, null, null], t)["tokens"] == [bs, 20, 21, be]
    assert handle_pattern([bs, 20, 21, be, null, null], t)["type"] == "point_box"
    error = handle_pattern([bs, 20, 5, 6, 7, 8], t)
    assert error == {"type": "error_box", "tokens": [bs, 20], "need_switch_to_ar": True, "is_terminal": False}
    assert handle_pattern([bs, 20, 5, 6, 7, 8], t, "fast")["type"] == "coord_box"
    assert handle_pattern([null, 1, 2, 3, 4, 5], t) == {
        "type": "im_end", "tokens": [t["im_end_token_id"]], "need_switch_to_ar": False, "is_terminal": True,
    }
    empty = handle_pattern([bs, t["none_token_id"], be, null, null, null], t)
    assert empty["type"] == "empty_box" and empty["tokens"] == [bs, t["none_token_id"], be]
    ref_end = t["ref_end_token_id"]
    ref = handle_pattern([t["ref_start_token_id"], 7, ref_end, ref_end, null, 9], t)
    assert ref["type"] == "ref_object" and ref["tokens"] == [t["ref_start_token_id"], 7, ref_end]


def _frame(vocab, picks):
    logits = torch.full((6, vocab), -20.0)
    for position, entries in enumerate(picks):
        for token, value in entries:
            logits[position, token] = value
    return logits


def test_hybrid_rejects_ambiguous_wide_coordinates_fast_keeps_top1():
    table = dict(TOKEN_IDS, coord_start_token_id=100, coord_end_token_id=300,
                 box_start_token_id=1, box_end_token_id=2, none_token_id=3, null_token_id=4, im_end_token_id=5)
    vocab = 320
    sure = [(150, 10.0)]
    # Two coordinates 150 apart with similar mass: neither reaches 0.9.
    wide = [(120, 5.0), (270, 4.8)]
    close = [(120, 5.0), (130, 4.8)]
    picks = [[(1, 10.0)], wide, sure, close, sure, [(2, 10.0)]]
    logits = _frame(vocab, picks)
    probs = torch.softmax(logits, dim=-1)
    assert decode_bbox_avg(logits, probs, table, keep_k=4).tolist() == [1, 0, 150, 120, 150, 2]
    assert decode_bbox_avg(logits, probs, table, keep_k=4, generation_mode="fast").tolist() == [1, 120, 150, 120, 150, 2]


def test_box_decode_requires_coordinate_in_topk_and_end_mass():
    t = TOKEN_IDS
    picks = [[(32, 10.0)], [(16, 10.0)], [(5, 10.0)], [(18, 10.0)], [(19, 10.0)], [(33, 10.0)]]
    logits = _frame(64, picks)
    assert decode_bbox_avg(logits, torch.softmax(logits, -1), t) is None
    picks[2] = [(17, 10.0)]
    picks[5] = [(7, 10.0)]
    logits = _frame(64, picks)
    assert classify_box_frame(torch.softmax(logits, -1), t) == "illegal_box"
    assert decode_bbox_avg(logits, torch.softmax(logits, -1), t) is None
