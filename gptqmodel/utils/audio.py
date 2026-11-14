# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import base64
import os
import tempfile
from contextlib import contextmanager
from io import BytesIO

from .image import load_bytes_from_url


try:
    import audioread
    AUDIOREAD_AVAILABLE = True
except ImportError:
    AUDIOREAD_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

AUDIOREAD_INSTALL_HINT = "audioread not installed. Please install via `pip install audioread>=3.1.0`."
AV_INSTALL_HINT = "av not installed. Please install via `pip install av>=16.0.1`."
LIBROSA_INSTALL_HINT = "librosa not installed. Please install via `pip install librosa>=0.11.0`."

# Audio/video calibration payloads are larger than images but still bounded:
# the same worker enforces the timeout, total budget, byte cap and redirect cap.
DEFAULT_MEDIA_FETCH_MAX_BYTES = 128 * 1024 * 1024

# PyAV/ffmpeg must not be allowed to open nested protocols (`concat:`,
# `subfile:`, HLS playlists that reference `file://`, ...).
MEDIA_PROTOCOL_WHITELIST = "file,pipe"


def _is_remote_media(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _resolve_media_source(path: str, *, allow_private_addresses: bool):
    """Return ``(bytes_or_local_path, is_remote)`` for one calibration entry."""

    if _is_remote_media(path):
        data = load_bytes_from_url(
            path,
            max_bytes=DEFAULT_MEDIA_FETCH_MAX_BYTES,
            allow_private_addresses=allow_private_addresses,
        )
        return data, True
    local_path = path[len("file://") :] if path.startswith("file://") else path
    try:
        size = os.path.getsize(local_path)
    except OSError:
        size = None
    if size is not None and size > DEFAULT_MEDIA_FETCH_MAX_BYTES:
        raise ValueError(
            f"Media file {local_path!r} is {size} bytes, exceeding the "
            f"{DEFAULT_MEDIA_FETCH_MAX_BYTES} byte limit."
        )
    return local_path, False


@contextmanager
def _materialized_media(source):
    """Give ffmpeg/librosa a local path for remote bytes and remove it after use."""

    if not isinstance(source, (bytes, bytearray)):
        yield source
        return
    descriptor, temporary = tempfile.mkstemp(prefix="gptqmodel-media-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source)
        yield temporary
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _load_audio(local_path, sample_rate: int, audio_start: float, audio_end):
    if not LIBROSA_AVAILABLE:
        raise ValueError(LIBROSA_INSTALL_HINT)
    return librosa.load(
        local_path,
        sr=sample_rate,
        offset=audio_start,
        duration=(audio_end - audio_start) if audio_end is not None else None,
    )[0]

def _check_if_video_has_audio(video_path):
    if not AV_AVAILABLE:
        raise ValueError(AV_INSTALL_HINT)
    container = av.open(video_path, options={"protocol_whitelist": MEDIA_PROTOCOL_WHITELIST})
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]],
                       use_audio_in_video: bool,
                       sample_rate: int=16000,
                       allow_private_addresses: bool=False,
                      ):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        # Accept any array-like sample buffer (NumPy, Torch,
                        # array.array, ...) without importing NumPy here.
                        if not isinstance(path, (str, bytes)) and hasattr(path, "ndim"):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(sample_rate * audio_start) : None if audio_end is None else int(sample_rate * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            media = BytesIO(base64.b64decode(base64_data))
                        else:
                            # Remote audio is downloaded by the bounded worker
                            # before ffmpeg/librosa sees it, so no URL reaches
                            # ffmpeg's protocol handlers.
                            media, _ = _resolve_media_source(
                                path, allow_private_addresses=allow_private_addresses
                            )
                        with _materialized_media(media) as local_path:
                            audios.append(_load_audio(local_path, sample_rate, audio_start, audio_end))
                        continue
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        media, _ = _resolve_media_source(
                            path, allow_private_addresses=allow_private_addresses
                        )
                        with _materialized_media(media) as local_path:
                            assert _check_if_video_has_audio(
                                local_path
                            ), "Video must has audio track when use_audio_in_video=True"
                            audios.append(_load_audio(local_path, sample_rate, audio_start, audio_end))
                        continue
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
    if len(audios) == 0:
        audios = None
    return audios
