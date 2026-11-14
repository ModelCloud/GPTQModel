# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Audio/video calibration fetches must use the bounded, protocol-filtered path."""

import os

import pytest
import torch

from gptqmodel.utils import audio


def _conversation(kind: str, url: str):
    return [[{"content": [{"type": kind, kind: url}]}]]


class _FakeLibrosa:
    def __init__(self, seen):
        self.seen = seen

    def load(self, path, **kwargs):
        self.seen["path"] = path
        self.seen["kwargs"] = kwargs
        assert isinstance(path, str)
        with open(path, "rb") as handle:
            self.seen["data"] = handle.read()
        return torch.zeros(4), kwargs["sr"]


def test_remote_audio_uses_bounded_worker_and_a_local_path(monkeypatch):
    calls = {}

    def fake_load_bytes(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return b"audio-bytes"

    seen = {}
    monkeypatch.setattr(audio, "load_bytes_from_url", fake_load_bytes)
    monkeypatch.setattr(audio, "LIBROSA_AVAILABLE", True)
    monkeypatch.setattr(audio, "librosa", _FakeLibrosa(seen), raising=False)

    result = audio.process_audio_info(
        _conversation("audio", "https://example.com/a.wav"),
        use_audio_in_video=False,
    )

    assert calls["url"] == "https://example.com/a.wav"
    assert calls["kwargs"]["max_bytes"] == audio.DEFAULT_MEDIA_FETCH_MAX_BYTES
    assert calls["kwargs"]["allow_private_addresses"] is False
    assert seen["data"] == b"audio-bytes"
    assert not os.path.exists(seen["path"])
    assert len(result) == 1


def test_remote_video_uses_protocol_whitelist_and_never_opens_the_url(monkeypatch):
    seen = {}

    def fake_load_bytes(url, **kwargs):
        return b"video-bytes"

    class _Stream:
        type = "audio"

    class _Container:
        streams = [_Stream()]

    class _FakeAV:
        @staticmethod
        def open(path, **kwargs):
            seen["av"] = (path, kwargs)
            assert os.path.exists(path)
            return _Container()

    monkeypatch.setattr(audio, "load_bytes_from_url", fake_load_bytes)
    monkeypatch.setattr(audio, "AV_AVAILABLE", True)
    monkeypatch.setattr(audio, "av", _FakeAV, raising=False)
    monkeypatch.setattr(audio, "LIBROSA_AVAILABLE", True)
    monkeypatch.setattr(audio, "librosa", _FakeLibrosa({}), raising=False)

    audio.process_audio_info(
        _conversation("video", "https://example.com/v.mp4"),
        use_audio_in_video=True,
    )

    assert seen["av"][1]["options"]["protocol_whitelist"] == "file,pipe"
    assert not seen["av"][0].startswith("http")


def test_local_media_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("gptqmodel.utils.audio.DEFAULT_MEDIA_FETCH_MAX_BYTES", 4)
    path = tmp_path / "a.wav"
    path.write_bytes(b"12345")

    with pytest.raises(ValueError, match="exceeding"):
        audio.process_audio_info(
            _conversation("audio", f"file://{path}"),
            use_audio_in_video=False,
        )
