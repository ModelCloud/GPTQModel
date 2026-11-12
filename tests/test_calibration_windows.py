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
    assert starts == [0, seqlen, 2 * seqlen, 3 * seqlen]
    assert len({tuple(window[0].tolist()) for window in windows}) == 4


def test_windows_wrap_around_without_repeating():
    cal = _calibration()
    seqlen = 16
    total = 2 * seqlen + 5
    windows = list(
        cal.iter_calibration_windows(
            _Tok(total), "ignored", seqlen=seqlen, nsamples=5
        )
    )

    # Wraparound keeps every window full-length and every start distinct.
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
