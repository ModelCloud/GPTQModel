# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Windowing rules for calibration text.

The Tenstorrent block-float path once collected its Hessian from
``ids[:, :seqlen]`` on every sample, so ``nsamples`` multiplied one passage and
GPTQ's error feedback was tuned to it. That produced a Llama-3.2-1B checkpoint
with perplexity 354 against 9.03 dense while every per-layer metric looked fine.
These tests pin the properties that make that impossible to do silently.
"""

import importlib

import pytest
import torch


def _calibration():
    return importlib.import_module("gptqmodel.utils.calibration")


class _Tok:
    """Tokenizer stub whose ids are ``0..n-1``, so windows are easy to compare."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, _text, return_tensors=None):
        class _Out:
            input_ids = None

        out = _Out()
        out.input_ids = torch.arange(self.n).unsqueeze(0)
        return out


class _Recorder:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        pass


def test_windows_stride_and_are_distinct():
    cal = _calibration()
    seqlen = 16
    windows = list(
        cal.iter_calibration_windows(
            _Tok(4 * seqlen), "ignored", seqlen=seqlen, nsamples=4
        )
    )

    assert len(windows) == 4
    assert all(window.shape == (1, seqlen) for window in windows)
    starts = [int(window[0, 0]) for window in windows]
    # span = 64 - 16 + 1 = 49, so the four windows sit at 0, 12, 24, 36.
    assert starts == [0, 12, 24, 36]
    assert len({tuple(window[0].tolist()) for window in windows}) == 4


def test_windows_cover_the_whole_corpus():
    """The sample budget must reach the end of the text, not just its start.

    Packing windows at ``index * seqlen`` caps the reach at
    ``nsamples * seqlen``: 32 windows of 256 only ever read the first 8k tokens,
    so a 100k-token calibration corpus was 92 % unused while the run reported the
    full record count.
    """

    cal = _calibration()
    seqlen, nsamples = 256, 32
    total = 100_000
    windows = list(
        cal.iter_calibration_windows(
            _Tok(total), "ignored", seqlen=seqlen, nsamples=nsamples
        )
    )

    starts = [int(window[0, 0]) for window in windows]
    assert len(set(starts)) == nsamples
    # The old rule would have stopped at (nsamples - 1) * seqlen == 7936.
    assert starts[-1] > 10 * (nsamples - 1) * seqlen
    # The last window must still end inside the text.
    assert starts[-1] + seqlen <= total
    # Consecutive windows stay ordered and non-overlapping.
    assert starts == sorted(starts)
    assert all(b - a >= seqlen for a, b in zip(starts, starts[1:]))


def test_overlapping_windows_stay_distinct():
    cal = _calibration()
    seqlen = 16
    total = 2 * seqlen + 5
    windows = list(
        cal.iter_calibration_windows(
            _Tok(total), "ignored", seqlen=seqlen, nsamples=5
        )
    )

    # When the budget exceeds the text the windows overlap, but every one must
    # still start at a different offset and stay full length.
    assert all(window.shape == (1, seqlen) for window in windows)
    starts = [int(window[0, 0]) for window in windows]
    assert len(set(starts)) == 5
    assert max(starts) <= total - seqlen


def test_short_text_warns_instead_of_silently_using_one_passage(monkeypatch):
    cal = _calibration()
    recorder = _Recorder()
    monkeypatch.setattr(cal, "setup_logger", lambda: recorder)

    windows = list(
        cal.iter_calibration_windows(_Tok(20), "ignored", seqlen=256, nsamples=3)
    )

    assert len(windows) == 3
    assert all(torch.equal(windows[0], window) for window in windows)
    assert recorder.warnings, "a single-passage Hessian must be reported"
    assert "same window" in recorder.warnings[0]


def test_text_shorter_than_a_window_is_an_error():
    cal = _calibration()
    with pytest.raises(ValueError, match="meaningless"):
        list(cal.iter_calibration_windows(_Tok(3), "ignored", seqlen=16, nsamples=2))


def test_invalid_window_arguments_are_rejected():
    cal = _calibration()
    with pytest.raises(ValueError, match="seqlen"):
        list(cal.iter_calibration_windows(_Tok(64), "ignored", seqlen=4, nsamples=2))
    with pytest.raises(ValueError, match="nsamples"):
        list(cal.iter_calibration_windows(_Tok(64), "ignored", seqlen=16, nsamples=0))


class _CountingTok:
    """Tokenizer stub yielding globally increasing ids, one block per call."""

    def __init__(self):
        self.next_id = 0

    def __call__(self, text, return_tensors=None):
        class _Out:
            input_ids = None

        start = self.next_id
        self.next_id += len(text)
        out = _Out()
        out.input_ids = torch.arange(start, self.next_id).unsqueeze(0)
        return out


def test_windows_accept_records_from_a_dataset():
    cal = _calibration()
    tok = _CountingTok()
    windows = list(
        cal.iter_calibration_windows(
            tok, ["a" * 8, "b" * 12, "c" * 8], seqlen=8, nsamples=2
        )
    )

    # 8 + 12 + 8 = 28 ids concatenated, so the windows stride across records.
    assert [tuple(window.shape) for window in windows] == [(1, 8), (1, 8)]
    # span = 28 - 8 + 1 = 21, so the second window starts at 21 // 2 = 10.
    assert [int(window[0, 0]) for window in windows] == [0, 10]


def test_load_calibration_records_reads_json_gz(tmp_path):
    import gzip
    import json

    cal = _calibration()
    path = tmp_path / "calibration.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump([{"text": "first"}, {"text": "second"}, {"text": "third"}], handle)

    assert cal.load_calibration_records(str(path)) == ["first", "second", "third"]
    assert cal.load_calibration_records(str(path), rows=2) == ["first", "second"]


def test_load_calibration_records_reads_jsonl(tmp_path):
    import json

    cal = _calibration()
    path = tmp_path / "calibration.jsonl"
    path.write_text("\n".join(json.dumps({"text": t}) for t in ("a", "b")) + "\n")

    assert cal.load_calibration_records(str(path)) == ["a", "b"]


def test_load_calibration_records_rejects_records_without_text(tmp_path):
    import json

    cal = _calibration()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps([{"nope": 1}]))

    with pytest.raises(ValueError, match="no calibration text"):
        cal.load_calibration_records(str(path))


def test_missing_calibration_file_is_reported_not_silently_skipped(tmp_path):
    cal = _calibration()
    missing = str(tmp_path / "absent.json.gz")
    with pytest.raises(FileNotFoundError, match="calibration file not found"):
        cal.load_calibration_records(missing)
